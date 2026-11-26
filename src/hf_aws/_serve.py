"""SageMaker endpoint creation shared by the extension and generated scripts."""

import json
import os
import re
import sys
from contextlib import ExitStack, redirect_stdout
from fnmatch import fnmatchcase
from time import sleep
from typing import Any
from urllib.parse import unquote
from uuid import uuid4

import boto3
from botocore.exceptions import BotoCoreError, ClientError, WaiterError
from huggingface_hub import get_token

SAGEMAKER_PRINCIPAL = "sagemaker.amazonaws.com"
CONTAINER_FIELDS = ("ContainerHostname", "ModelDataUrl", "ModelDataSource", "AdditionalModelDataSources")
CUDA_13_FAMILIES = {"g4dn", "g5", "g6", "g6e", "p4d", "p4de", "p5", "p5e", "p5en"}


def _values(value: Any) -> list:
    return value if isinstance(value, list) else [value]


def _principal_matches(principal: Any) -> bool:
    return principal == "*" or (
        isinstance(principal, dict)
        and (SAGEMAKER_PRINCIPAL in _values(principal.get("Service")) or "*" in _values(principal.get("AWS")))
    )


def _applies_to_sagemaker(statement: dict) -> bool:
    principal = _principal_matches(statement.get("Principal"))
    if "NotPrincipal" in statement:
        principal = not _principal_matches(statement["NotPrincipal"])
    actions = statement.get("Action", statement.get("NotAction", []))
    action = any(fnmatchcase("sts:assumerole", pattern.lower()) for pattern in _values(actions))
    if "NotAction" in statement:
        action = not action
    return principal and action


def _any_match(matches: list[bool | None]) -> bool | None:
    if True in matches:
        return True
    return None if None in matches else False


def _condition_matches(operator: str, key: str, values: Any, arn: str, region: str) -> bool | None:
    partition, account = arn.split(":")[1], arn.split(":")[4]
    key = key.lower()
    if operator == "Bool" and key == "aws:principalisawsservice":
        return any(str(value).lower() == "true" for value in _values(values))
    operator = operator.removesuffix("IfExists")
    negative = operator in {"StringNotEquals", "StringNotLike", "ArnNotEquals", "ArnNotLike"}
    positive = operator.replace("Not", "")
    if positive not in {"StringEquals", "StringLike", "ArnEquals", "ArnLike"}:
        return None
    matches = []
    for value in _values(values):
        if key == "aws:sourceaccount":
            matches.append(fnmatchcase(account, value) if positive.endswith("Like") else account == value)
        elif key == "aws:sourcearn":
            parts = value.split(":", 5)
            context = ["arn", partition, "sagemaker", region, account]
            compare = (lambda actual, pattern: actual == pattern) if positive == "StringEquals" else fnmatchcase
            if len(parts) != 6 or not all(
                compare(actual, pattern) for actual, pattern in zip(context, parts, strict=True)
            ):
                matches.append(False)
            elif positive == "StringEquals":
                matches.append(False if "*" in parts[5] or "?" in parts[5] else None)
            else:
                # Only a full resource wildcard can be confirmed here.
                matches.append(True if parts[5] == "*" else None)
        else:
            matches.append(None)
    result = _any_match(matches)
    return not result if negative and result is not None else result


def _conditions_match(conditions: dict, arn: str, region: str) -> bool | None:
    results = [
        _condition_matches(operator, key, values, arn, region)
        for operator, entries in conditions.items()
        for key, values in entries.items()
    ]
    if False in results:
        return False
    return None if None in results else True


def validate_role_trust(role: dict, region: str) -> bool:
    """Reject definite trust failures; return False for conditions only AWS can evaluate.

    This validates service trust, not iam:PassRole, SCPs, or runtime permissions.
    STS AssumeRole as the caller cannot test assumption as the SageMaker service.
    """
    arn = role["Arn"]
    try:
        policy = role["AssumeRolePolicyDocument"]
        if isinstance(policy, str):
            policy = json.loads(unquote(policy))
        allowed, denied = [], []
        for statement in _values(policy.get("Statement", [])):
            if _applies_to_sagemaker(statement):
                match = _conditions_match(statement.get("Condition", {}), arn, region)
                if statement.get("Effect") == "Allow":
                    allowed.append(match)
                elif statement.get("Effect") == "Deny":
                    denied.append(match)
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ValueError(f"Cannot read execution role trust policy for {arn}: {error}") from error
    if True in denied or not any(match is not False for match in allowed):
        raise ValueError(f"Execution role {arn} does not allow {SAGEMAKER_PRINCIPAL} to assume it in {region}")
    return True in allowed and None not in denied


def resolve_execution_role(boto_session, identifier: str | None) -> str:
    identifier = identifier or os.environ.get("SAGEMAKER_ROLE_ARN")
    if identifier is None:
        from sagemaker.core.helper.session_helper import Session, get_execution_role

        try:
            identifier = get_execution_role(sagemaker_session=Session(boto_session=boto_session))
        except ValueError:
            identifier = "sagemaker_execution_role"
    is_arn = identifier.startswith("arn:")
    name = identifier.rsplit("/", 1)[-1] if is_arn else identifier
    if (is_arn and not re.fullmatch(r"arn:[^:]+:iam::\d{12}:role/.+", identifier)) or not re.fullmatch(
        r"[A-Za-z0-9_+=,.@-]{1,64}", name
    ):
        raise ValueError("Invalid execution role. Use --role with an IAM role ARN or --role-name with a name")
    try:
        role = boto_session.client("iam").get_role(RoleName=name)["Role"]
    except ClientError as error:
        code = error.response["Error"]["Code"]
        if code == "NoSuchEntity":
            raise ValueError(f"Execution role '{name}' does not exist in this AWS account") from error
        if code in {"AccessDenied", "AccessDeniedException", "UnauthorizedOperation"}:
            raise ValueError(f"Cannot read execution role '{name}'. Grant iam:GetRole to the AWS caller") from error
        raise
    if is_arn and role["Arn"] != identifier:
        raise ValueError(f"IAM resolved '{name}' to a different ARN: {role['Arn']}")
    if not validate_role_trust(role, boto_session.region_name):
        print("AWS will evaluate the execution role's conditional trust during deployment.", file=sys.stderr)
    return role["Arn"]


def endpoint_missing(error: ClientError) -> bool:
    details = error.response.get("Error", {})
    message = details.get("Message", "").lower()
    return details.get("Code") in {"ResourceNotFound", "ResourceNotFoundException"} or (
        details.get("Code") == "ValidationException"
        and ("could not find endpoint" in message or "endpoint does not exist" in message)
    )


def wait_for_endpoint(client, name: str, *, deleted: bool = False) -> None:
    waiter = client.get_waiter("endpoint_deleted" if deleted else "endpoint_in_service")
    try:
        waiter.wait(EndpointName=name, WaiterConfig={"Delay": 30, "MaxAttempts": 120})
    except WaiterError as error:
        status = error.last_response.get("EndpointStatus")
        reason = error.last_response.get("FailureReason") or str(error)
        raise ValueError(f"Endpoint '{name}' waiter failed ({status or 'unknown status'}): {reason}") from error


def deployment_environment(plan: dict, token: str | None, index: int = 0) -> dict[str, str]:
    container = plan["model_package"]["InferenceSpecification"]["Containers"][index]
    env = dict(container.get("Environment", {}))
    for key in plan["container_secret_environment"][index]:
        if key not in env:
            if key not in os.environ:
                raise ValueError(f"Set {key} in the environment before running the exported script")
            env[key] = os.environ[key]
    if index == 0 and plan["use_hf_token"]:
        env["HF_TOKEN"] = token or get_token() or ""
        if not env["HF_TOKEN"]:
            raise ValueError("Model access requires a Hugging Face token. Run hf auth login or set HF_TOKEN")
        env.setdefault("HUGGING_FACE_HUB_TOKEN", env["HF_TOKEN"])
    return env


def deployment_image(spec: dict) -> str:
    image = spec["Image"]
    if digest := spec.get("ImageDigest"):
        repository = image.split("@", 1)[0]
        registry, _, name = repository.rpartition("/")
        return f"{registry}/{name.partition(':')[0]}@{digest.lower()}"
    return image


def deployment_containers(plan: dict, token: str | None) -> list[dict]:
    containers = []
    for index, spec in enumerate(plan["model_package"]["InferenceSpecification"]["Containers"]):
        container = {key: spec[key] for key in CONTAINER_FIELDS if key in spec}
        container["Image"] = deployment_image(spec)
        if environment := deployment_environment(plan, token, index):
            container["Environment"] = environment
        containers.append(container)
    return containers


def _delete_created(delete, **params) -> None:
    try:
        delete(**params)
    except (BotoCoreError, ClientError) as error:
        print(f"Could not clean up deployment resources {params}: {error}", file=sys.stderr)


def _create_endpoint(client, name: str, config: str) -> None:
    # SageMaker can briefly report that a newly created endpoint config is missing.
    for delay in (1, 2, 4, None):
        try:
            client.create_endpoint(EndpointName=name, EndpointConfigName=config)
            return
        except ClientError as error:
            details = error.response.get("Error", {})
            message = details.get("Message", "").lower()
            missing = config.lower() in message and any(
                phrase in message for phrase in ("could not find", "does not exist", "not found")
            )
            if delay is None or details.get("Code") != "ValidationException" or not missing:
                raise
            sleep(delay)


def apply_plan(plan: dict, *, token: str | None = None) -> dict:
    with redirect_stdout(sys.stderr):
        containers = deployment_containers(plan, token)
        boto_session = boto3.Session(region_name=plan["region"], profile_name=plan["profile"])
        client = boto_session.client("sagemaker")
        name = plan["name"]
        try:
            client.describe_endpoint(EndpointName=name)
        except ClientError as error:
            if not endpoint_missing(error):
                raise
        else:
            raise ValueError(f"Endpoint '{name}' already exists in {plan['region']}")

        role = resolve_execution_role(boto_session, plan["role"])
        resource_name = f"{name[:54].rstrip('-')}-{uuid4().hex[:8]}"
        variant = {
            "VariantName": "AllTraffic",
            "ModelName": resource_name,
            "InstanceType": plan["instance_type"],
            "InitialInstanceCount": plan["instance_count"],
            "ContainerStartupHealthCheckTimeoutInSeconds": plan["startup_timeout_seconds"],
        }
        family = plan["instance_type"].split(".")[1]
        if family in CUDA_13_FAMILIES and any(
            "cu130" in container["Image"].rpartition(":")[2].split("-")
            for container in plan["model_package"]["InferenceSpecification"]["Containers"]
        ):
            variant["InferenceAmiVersion"] = "al2023-ami-sagemaker-inference-gpu-4-1"

        # Separate names let concurrent attempts clean up only their own resources.
        definition = {"PrimaryContainer": containers[0]} if len(containers) == 1 else {"Containers": containers}
        with ExitStack() as cleanup:
            try:
                client.create_model(ModelName=resource_name, ExecutionRoleArn=role, **definition)
                cleanup.callback(_delete_created, client.delete_model, ModelName=resource_name)
                client.create_endpoint_config(EndpointConfigName=resource_name, ProductionVariants=[variant])
                cleanup.callback(_delete_created, client.delete_endpoint_config, EndpointConfigName=resource_name)
                _create_endpoint(client, name, resource_name)
            except ClientError as error:
                if error.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 500) >= 500:
                    cleanup.pop_all()
                raise
            except BaseException:
                # A transport failure or interruption can leave submission status unknown.
                cleanup.pop_all()
                raise
            cleanup.pop_all()
        if plan["wait"]:
            wait_for_endpoint(client, name)
            status = "InService"
        else:
            status = "Submitted"
        return {"name": name, "status": status, "region": plan["region"], "model_id": plan["model_id"]}
