"""SageMaker endpoint operations using the AWS account as the source of truth."""

from __future__ import annotations

import json
from contextlib import closing

import boto3
from botocore.exceptions import ClientError

from hf_aws._serve import endpoint_missing, wait_for_endpoint
from hf_aws.errors import AwsError


def summary(endpoint: dict) -> dict:
    return {
        "name": endpoint["EndpointName"],
        "status": endpoint["EndpointStatus"],
        "created_at": endpoint.get("CreationTime"),
        "updated_at": endpoint.get("LastModifiedTime"),
    }


class Deployments:
    def __init__(self, session: boto3.Session):
        self.session = session
        self.client = session.client("sagemaker")

    def list(self, status: str | None) -> list[dict]:
        filters = {"StatusEquals": status} if status else {}
        return [
            summary(endpoint)
            for page in self.client.get_paginator("list_endpoints").paginate(**filters)
            for endpoint in page["Endpoints"]
        ]

    def status(self, name: str) -> dict:
        endpoint = self.client.describe_endpoint(EndpointName=name)
        return {
            **summary(endpoint),
            "arn": endpoint.get("EndpointArn"),
            "region": self.session.region_name,
            "failure_reason": endpoint.get("FailureReason"),
            "endpoint_config": endpoint["EndpointConfigName"],
            "variants": [
                {
                    "name": variant["VariantName"],
                    "current_instances": variant.get("CurrentInstanceCount"),
                    "desired_instances": variant.get("DesiredInstanceCount"),
                }
                for variant in endpoint.get("ProductionVariants", [])
            ],
        }

    def describe(self, name: str) -> dict:
        endpoint = self.status(name)
        config = self.client.describe_endpoint_config(EndpointConfigName=endpoint["endpoint_config"])
        variants = {item["name"]: item for item in endpoint["variants"]}
        for variant in config["ProductionVariants"]:
            name = variant["VariantName"]
            details = variants.setdefault(name, {"name": name})
            model_name = variant.get("ModelName")
            details.update(model_name=model_name, instance_type=variant.get("InstanceType"))
            if model_name:
                model = self.client.describe_model(ModelName=model_name)
                containers = model.get("Containers") or [model.get("PrimaryContainer", {})]
                details["containers"] = []
                for container in containers:
                    env = container.get("Environment", {})
                    served_names = env.get("SM_VLLM_SERVED_MODEL_NAME")
                    details["containers"].append(
                        {
                            "model_id": env.get("SM_VLLM_MODEL") or env.get("HF_MODEL_ID"),
                            "model_server": "vllm" if env.get("SM_VLLM_MODEL") else None,
                            "served_model_name": served_names.split()[0] if served_names else None,
                            "image": container.get("Image"),
                        }
                    )
        endpoint["variants"] = list(variants.values())
        return endpoint

    def wait(self, name: str, *, deleted: bool = False) -> dict:
        wait_for_endpoint(self.client, name, deleted=deleted)
        return {"name": name, "status": "Deleted"} if deleted else self.status(name)

    def scale(self, name: str, count: int, *, variant: str | None = None) -> dict:
        endpoint = self.status(name)
        if endpoint["status"] != "InService":
            raise AwsError(f"Endpoint '{name}' must be InService before scaling")
        variants = [item for item in endpoint["variants"] if variant is None or item["name"] == variant]
        if len(variants) != 1:
            raise AwsError("Choose one existing production variant with --variant")
        target = variants[0]
        if target["desired_instances"] is None:
            raise AwsError(f"Variant '{target['name']}' does not expose a provisioned instance count")
        if target["desired_instances"] != count:
            self.client.update_endpoint_weights_and_capacities(
                EndpointName=name,
                DesiredWeightsAndCapacities=[{"VariantName": target["name"], "DesiredInstanceCount": count}],
            )
        return {"name": name, "variant": target["name"], "desired_instances": count}

    def delete(self, name: str, *, wait: bool) -> dict:
        try:
            self.client.delete_endpoint(EndpointName=name)
        except ClientError as error:
            if not endpoint_missing(error):
                raise
            return {"name": name, "status": "Deleted"}
        return self.wait(name, deleted=True) if wait else {"name": name, "status": "Deleting"}

    def invoke(self, name: str, payload: object, *, variant: str | None = None) -> object:
        runtime = self.session.client("sagemaker-runtime")
        target = {"TargetVariant": variant} if variant else {}
        response = runtime.invoke_endpoint(
            EndpointName=name, ContentType="application/json", Body=json.dumps(payload, allow_nan=False), **target
        )
        with closing(response["Body"]) as stream:
            data = stream.read().decode("utf-8")
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return data

    def input_payload(
        self, name: str, text: str, api: str, model_id: str | None, max_tokens: int, variant: str | None
    ) -> dict:
        if api == "auto":
            endpoint = self.describe(name)
            variants = [item for item in endpoint["variants"] if variant is None or item["name"] == variant]
            if len(variants) != 1 or len(variants[0].get("containers", [])) != 1:
                raise AwsError("Select --variant and --api, or provide a JSON request with --payload")
            container = variants[0]["containers"][0]
            api = container.get("model_server") or "huggingface"
            model_id = model_id or container.get("served_model_name") or container.get("model_id")
        if api == "vllm":
            payload = {"prompt": text, "max_tokens": max_tokens}
            if model_id:
                payload["model"] = model_id
            return payload
        return {"inputs": text, "parameters": {"max_new_tokens": max_tokens}}

    def logs(self, name: str, tail: int) -> list[str]:
        client = self.session.client("logs")
        group = f"/aws/sagemaker/Endpoints/{name}"
        try:
            streams = client.describe_log_streams(
                logGroupName=group, orderBy="LastEventTime", descending=True, limit=5
            )["logStreams"]
            events = [
                event
                for stream in streams
                for event in client.get_log_events(
                    logGroupName=group, logStreamName=stream["logStreamName"], startFromHead=False, limit=tail
                )["events"]
            ]
        except client.exceptions.ResourceNotFoundException:
            return []
        events.sort(key=lambda event: event["timestamp"])
        return [event["message"] for event in events[-tail:]]


def quotas(session: boto3.Session, instance_type: str | None) -> list[dict]:
    client = session.client("service-quotas")
    rows = []
    for page in client.get_paginator("list_service_quotas").paginate(ServiceCode="sagemaker"):
        for entry in page["Quotas"]:
            sku, _, suffix = entry["QuotaName"].partition(" ")
            if not sku.startswith("ml.") or suffix != "for endpoint usage":
                continue
            if instance_type is None or sku == instance_type:
                rows.append(
                    {
                        "instance_type": sku,
                        "quota": entry["Value"],
                        "unit": entry.get("Unit"),
                        "adjustable": entry["Adjustable"],
                        "quota_code": entry["QuotaCode"],
                        "region": session.region_name,
                    }
                )
    return sorted(rows, key=lambda row: row["instance_type"])
