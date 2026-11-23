"""SageMaker ModelBuilder execution shared by the extension and generated scripts."""

import json
import os
import re
import sys
from contextlib import redirect_stdout
from fnmatch import fnmatchcase
from typing import Any
from urllib.parse import unquote

import boto3
from botocore.exceptions import ClientError, WaiterError
from huggingface_hub import get_token

SAGEMAKER_PRINCIPAL = "sagemaker.amazonaws.com"


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
                # ModelBuilder chooses model resource names later. Only a full
                # resource wildcard can be confirmed before creating the model.
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


def resolve_execution_role(boto_session, sdk_session, identifier: str | None) -> str:
    from sagemaker.core.helper.session_helper import get_execution_role

    identifier = identifier or os.environ.get("SAGEMAKER_ROLE_ARN")
    if identifier is None:
        try:
            identifier = get_execution_role(sagemaker_session=sdk_session)
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


def deployment_environment(plan: dict, token: str | None) -> dict[str, str]:
    env = dict(plan["environment"])
    for key in plan["secret_environment"]:
        if key not in env:
            if key not in os.environ:
                raise ValueError(f"Set {key} in the environment before running the exported script")
            env[key] = os.environ[key]
    if plan["use_hf_token"]:
        env["HF_TOKEN"] = token or get_token() or ""
        if not env["HF_TOKEN"]:
            raise ValueError("Model access requires a Hugging Face token. Run hf auth login or set HF_TOKEN")
    return env


def apply_plan(plan: dict, *, token: str | None = None) -> dict:
    with redirect_stdout(sys.stderr):
        from sagemaker.core.helper.session_helper import Session
        from sagemaker.serve import ModelBuilder, ModelServer
        from sagemaker.serve.builder.schema_builder import SchemaBuilder

        env = deployment_environment(plan, token)
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

        session = Session(boto_session=boto_session)
        role = resolve_execution_role(boto_session, session, plan["role"])
        server = {"model_server": ModelServer.VLLM} if plan["model_server"] == "vllm" else {}
        builder = ModelBuilder(
            model=plan["model_id"],
            role_arn=role,
            sagemaker_session=session,
            instance_type=plan["instance_type"],
            schema_builder=SchemaBuilder(**plan["schema"]),
            env_vars=env,
            model_metadata={"HF_TASK": plan["task"]},
            **server,
        )
        builder.build()
        builder.deploy(
            endpoint_name=name,
            instance_type=plan["instance_type"],
            initial_instance_count=plan["instance_count"],
            container_timeout_in_seconds=plan["startup_timeout_seconds"],
            wait=False,
        )
        if plan["wait"]:
            wait_for_endpoint(client, name)
            status = "InService"
        else:
            status = "Submitted"
        return {"name": name, "status": status, "region": plan["region"], "model_id": plan["model_id"]}
