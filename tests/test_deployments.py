import json
from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import Mock

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.response import StreamingBody
from botocore.stub import Stubber

from hf_aws import _serve
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


@pytest.mark.parametrize("wait", [True, False])
def test_apply_plan_builds_and_waits_or_submits(monkeypatch, wait):
    import sagemaker.serve
    from sagemaker.core.helper import session_helper

    plan = deployment_plan(
        "openai-community/gpt2",
        name=NAME,
        region=REGION,
        profile="test",
        role=ROLE_ARN,
        instance_type="ml.g5.12xlarge",
        count=2,
        tensor_parallel_size=2,
        environment={"HF_TASK": "text-generation"},
        token="fake-token",
        wait=wait,
    )
    client, stubber = stubbed("sagemaker")
    stubber.add_client_error("describe_endpoint", "ValidationException", MISSING_ENDPOINT, expected_params=ENDPOINT)
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
    monkeypatch.setattr(session_helper, "Session", Mock())
    builder = Mock()
    monkeypatch.setattr(sagemaker.serve, "ModelBuilder", builder)
    waiter = Mock()
    monkeypatch.setattr(_serve, "wait_for_endpoint", waiter)
    with stubber:
        result = _serve.apply_plan(plan.public(), token="fake-token")
    iam.get_role.assert_called_once_with(RoleName="sagemaker")
    assert builder.call_args.kwargs["model_server"] == sagemaker.serve.ModelServer.VLLM
    assert builder.call_args.kwargs["env_vars"]["HF_TOKEN"] == "fake-token"
    assert builder.call_args.kwargs["env_vars"]["SM_VLLM_TENSOR_PARALLEL_SIZE"] == "2"
    builder.return_value.build.assert_called_once()
    builder.return_value.deploy.assert_called_once_with(
        endpoint_name=NAME,
        instance_type="ml.g5.12xlarge",
        initial_instance_count=2,
        container_timeout_in_seconds=1800,
        wait=False,
    )
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
