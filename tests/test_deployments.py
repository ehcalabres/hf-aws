import json
from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import Mock, call
from uuid import UUID

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.response import StreamingBody
from botocore.stub import Stubber

from hf_aws import _serve, recipes
from hf_aws.deployments import Deployments, quotas
from hf_aws.planning import deployment_plan

REGION = "us-west-2"
NAME = "example-endpoint"
ROLE_ARN = "arn:aws:iam::123456789012:role/sagemaker"
ENDPOINT = {"EndpointName": NAME}
MISSING_ENDPOINT = f'Could not find endpoint "arn:aws:sagemaker:{REGION}:123456789012:endpoint/{NAME}".'


def stubbed(service):
    session = boto3.Session(aws_access_key_id="fake", aws_secret_access_key="fake", region_name=REGION)
    client = session.client(service)
    return client, Stubber(client)


def test_endpoints_list_paginates_and_passes_filter():
    client, stubber = stubbed("sagemaker")
    endpoint = {
        **ENDPOINT,
        "EndpointArn": f"arn:aws:sagemaker:{REGION}:123456789012:endpoint/{NAME}",
        "CreationTime": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "LastModifiedTime": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "EndpointStatus": "InService",
    }
    stubber.add_response("list_endpoints", {"Endpoints": [], "NextToken": "next"}, {"StatusEquals": "InService"})
    stubber.add_response(
        "list_endpoints", {"Endpoints": [endpoint]}, {"StatusEquals": "InService", "NextToken": "next"}
    )
    with stubber:
        rows = Deployments(Mock(client=Mock(return_value=client))).list("InService")
    assert [row["name"] for row in rows] == [NAME]
    stubber.assert_no_pending_responses()


def test_delete_already_missing_is_idempotent_and_other_errors_surface():
    client, stubber = stubbed("sagemaker")
    stubber.add_client_error("delete_endpoint", "ValidationException", MISSING_ENDPOINT, expected_params=ENDPOINT)
    stubber.add_client_error("delete_endpoint", "AccessDeniedException", "access denied", expected_params=ENDPOINT)
    with stubber:
        deployments = Deployments(Mock(client=Mock(return_value=client)))
        assert deployments.delete(NAME, wait=True)["status"] == "Deleted"
        with pytest.raises(ClientError, match="access denied"):
            deployments.delete(NAME, wait=True)
    stubber.assert_no_pending_responses()


def test_invoke_forwards_custom_json_and_target_variant():
    runtime, stubber = stubbed("sagemaker-runtime")
    payload = {"prompt": "Hello", "max_tokens": 32}
    response = b'{"choices":[{"text":" world"}]}'
    stubber.add_response(
        "invoke_endpoint",
        {"Body": StreamingBody(BytesIO(response), len(response))},
        {**ENDPOINT, "Body": json.dumps(payload), "ContentType": "application/json", "TargetVariant": "AllTraffic"},
    )
    session = Mock(client=Mock(side_effect={"sagemaker": Mock(), "sagemaker-runtime": runtime}.__getitem__))
    with stubber:
        assert Deployments(session).invoke(NAME, payload, variant="AllTraffic") == json.loads(response)
    stubber.assert_no_pending_responses()


def test_logs_return_latest_events_in_timestamp_order():
    logs, stubber = stubbed("logs")
    group = f"/aws/sagemaker/Endpoints/{NAME}"
    streams = [{"logStreamName": "newest"}, {"logStreamName": "older"}]
    stubber.add_response(
        "describe_log_streams",
        {"logStreams": streams},
        {"logGroupName": group, "orderBy": "LastEventTime", "descending": True, "limit": 5},
    )
    stubber.add_response(
        "get_log_events",
        {"events": [{"timestamp": 2, "message": "two"}, {"timestamp": 4, "message": "four"}]},
        {"logGroupName": group, "logStreamName": "newest", "startFromHead": False, "limit": 2},
    )
    stubber.add_response(
        "get_log_events",
        {"events": [{"timestamp": 1, "message": "one"}, {"timestamp": 3, "message": "three"}]},
        {"logGroupName": group, "logStreamName": "older", "startFromHead": False, "limit": 2},
    )
    session = Mock(client=Mock(side_effect={"sagemaker": Mock(), "logs": logs}.__getitem__))
    with stubber:
        assert Deployments(session).logs(NAME, 2) == ["three", "four"]
    stubber.assert_no_pending_responses()


def test_quotas_hide_non_endpoint_limits():
    client, stubber = stubbed("service-quotas")
    quota = {"QuotaCode": "L-11111111", "Value": 0.0, "Unit": "None", "Adjustable": True}
    stubber.add_response(
        "list_service_quotas",
        {
            "Quotas": [
                {**quota, "QuotaName": "ml.g6.2xlarge for endpoint usage"},
                {**quota, "QuotaName": "ml.g6.2xlarge for training job usage"},
            ]
        },
        {"ServiceCode": "sagemaker"},
    )
    session = Mock(client=Mock(return_value=client), region_name=REGION)
    with stubber:
        rows = quotas(session, None)
    assert [(row["instance_type"], row["quota"]) for row in rows] == [("ml.g6.2xlarge", 0.0)]
    stubber.assert_no_pending_responses()


@pytest.mark.parametrize("optimized,wait", [(False, True), (True, False)])
def test_apply_plan_creates_endpoint_from_model_package(monkeypatch, tmp_path, optimized, wait):
    model_id = "example/model" if optimized else "openai-community/gpt2"
    source = None
    if optimized:
        source = "hf://datasets/example/recipes"
        image = f"123456789012.dkr.ecr.{REGION}.amazonaws.com/custom:1"
        package = recipes.default_package(
            "example",
            model_id,
            image,
            "ml.p5e.48xlarge",
            {
                "SM_VLLM_MODEL": model_id,
                "SM_VLLM_TENSOR_PARALLEL_SIZE": "8",
                "SM_VLLM_TOKENIZER_MODE": "auto",
                "SM_VLLM_MAX_MODEL_LEN": "8192",
            },
        )
        alternative = deepcopy(package["InferenceSpecification"])
        alternative["Name"] = "long-context"
        package["AdditionalInferenceSpecifications"] = [alternative]
        package["InferenceSpecification"]["Containers"][0]["Environment"]["SM_VLLM_MAX_MODEL_LEN"] = "2048"
        path = tmp_path / "recipe.json"
        path.write_text(json.dumps(package), encoding="utf-8")
        monkeypatch.setattr(recipes, "hf_hub_download", Mock(return_value=path))
    plan = deployment_plan(
        model_id,
        name=NAME,
        region=REGION,
        profile="test",
        role=ROLE_ARN,
        instance_type=None if optimized else "ml.g5.12xlarge",
        count=2,
        tensor_parallel_size=None if optimized else 2,
        environment={"HF_TASK": "text-generation"},
        token="fake-token",
        wait=wait,
        recipe_source=source,
        recipe_name="long-context" if optimized else None,
    )
    client, stubber = stubbed("sagemaker")
    stubber.add_client_error("describe_endpoint", "ValidationException", MISSING_ENDPOINT, expected_params=ENDPOINT)
    monkeypatch.setattr(_serve, "uuid4", lambda: UUID("12345678-1234-1234-1234-123456789abc"))
    resource = f"{NAME}-12345678"
    arn = f"arn:aws:sagemaker:{REGION}:123456789012"
    spec = plan.public()["model_package"]["InferenceSpecification"]
    container = spec["Containers"][0]
    assert container["Environment"]["SM_VLLM_MODEL"] == model_id
    assert container["Environment"]["SM_VLLM_TENSOR_PARALLEL_SIZE"] == ("8" if optimized else "2")
    assert container["Environment"]["SM_VLLM_HOST"] == "0.0.0.0"
    if optimized:
        assert container["Environment"]["SM_VLLM_TOKENIZER_MODE"] == "auto"
        assert container["Environment"]["SM_VLLM_MAX_MODEL_LEN"] == "8192"
        assert container["Image"] == image
    stubber.add_response(
        "create_model",
        {"ModelArn": f"{arn}:model/{resource}"},
        {
            "ModelName": resource,
            "ExecutionRoleArn": ROLE_ARN,
            "PrimaryContainer": {
                **container,
                "Environment": {
                    **container["Environment"],
                    "HF_TOKEN": "fake-token",
                    "HUGGING_FACE_HUB_TOKEN": "fake-token",
                },
            },
        },
    )
    variant = {
        "VariantName": "AllTraffic",
        "ModelName": resource,
        "InstanceType": "ml.p5e.48xlarge" if optimized else "ml.g5.12xlarge",
        "InitialInstanceCount": 2,
        "ContainerStartupHealthCheckTimeoutInSeconds": 3600 if optimized else 1800,
    }
    if "-cu130-" in container["Image"]:
        variant["InferenceAmiVersion"] = "al2023-ami-sagemaker-inference-gpu-4-1"
    stubber.add_response(
        "create_endpoint_config",
        {"EndpointConfigArn": f"{arn}:endpoint-config/{resource}"},
        {"EndpointConfigName": resource, "ProductionVariants": [variant]},
    )
    request = {**ENDPOINT, "EndpointConfigName": resource}
    if optimized:
        stubber.add_client_error(
            "create_endpoint",
            "ValidationException",
            f"Could not find endpoint configuration {resource}",
            expected_params=request,
        )
    stubber.add_response("create_endpoint", {"EndpointArn": f"{arn}:endpoint/{NAME}"}, request)
    sleep = Mock()
    monkeypatch.setattr(_serve, "sleep", sleep)
    iam = Mock()
    iam.get_role.return_value = {
        "Role": {
            "Arn": ROLE_ARN,
            "AssumeRolePolicyDocument": {
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "sagemaker.amazonaws.com"},
                        "Action": "sts:AssumeRole",
                    }
                ]
            },
        }
    }
    session = Mock(client=Mock(side_effect={"sagemaker": client, "iam": iam}.__getitem__), region_name=REGION)
    monkeypatch.setattr(_serve.boto3, "Session", Mock(return_value=session))
    waiter = Mock()
    monkeypatch.setattr(_serve, "wait_for_endpoint", waiter)
    with stubber:
        result = _serve.apply_plan(plan.public(), token="fake-token")
    stubber.assert_no_pending_responses()
    iam.get_role.assert_called_once_with(RoleName="sagemaker")
    assert sleep.call_args_list == ([call(1)] if optimized else [])
    if wait:
        waiter.assert_called_once_with(client, NAME)
        assert result["status"] == "InService"
    else:
        waiter.assert_not_called()
        assert result["status"] == "Submitted"


def test_scale_updates_capacity_without_replacing_endpoint_config():
    client, stubber = stubbed("sagemaker")
    arn = f"arn:aws:sagemaker:{REGION}:123456789012:endpoint/{NAME}"
    endpoint = {
        **ENDPOINT,
        "EndpointArn": arn,
        "EndpointStatus": "InService",
        "EndpointConfigName": "existing-config",
        "CreationTime": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "LastModifiedTime": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "ProductionVariants": [{"VariantName": "AllTraffic", "DesiredInstanceCount": 1, "CurrentInstanceCount": 1}],
    }
    stubber.add_response("describe_endpoint", endpoint, ENDPOINT)
    stubber.add_response(
        "update_endpoint_weights_and_capacities",
        {"EndpointArn": arn},
        {**ENDPOINT, "DesiredWeightsAndCapacities": [{"VariantName": "AllTraffic", "DesiredInstanceCount": 2}]},
    )
    session = Mock(client=Mock(return_value=client), region_name=REGION)
    with stubber:
        assert Deployments(session).scale(NAME, 2)["desired_instances"] == 2
    stubber.assert_no_pending_responses()
