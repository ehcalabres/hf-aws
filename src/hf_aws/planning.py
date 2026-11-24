"""SageMaker instance estimates and deployment plans."""

import json
import math
import re
from dataclasses import asdict, dataclass
from importlib.resources import files
from uuid import uuid4

from huggingface_hub import HfApi, get_token
from huggingface_hub.utils import validate_repo_id

from hf_aws.environment import validate_environment
from hf_aws.errors import AwsError
from hf_aws.hf_mem import HF_MEM_VERSION, EstimateOptions, InstanceData, InstanceInfo, run_hf_mem_batch
from hf_aws.schemas import get_schema_samples

ENDPOINT_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?")
INSTANCE_TYPE = re.compile(r"ml\.[a-z0-9]+\.[a-z0-9]+")
SECRET_NAME = re.compile(r"TOKEN|PASSWORD|SECRET|CREDENTIAL|API_?KEY|ACCESS_?KEY|PRIVATE_?KEY", re.IGNORECASE)


def load_catalog() -> InstanceData:
    data = json.loads(files("hf_aws").joinpath("data", "sagemaker.json").read_text(encoding="utf-8"))
    return InstanceData(
        data["instances"], data["safetyMarginDefault"], data["last_updated"], data.get("pricingRegion")
    )


def projection(instance: InstanceInfo, count: int, hours: float) -> dict:
    if count < 1 or not math.isfinite(hours) or hours <= 0:
        raise AwsError("--instances must be positive and --hours must be finite and positive")
    price = instance.get("pricePerHourUsd")
    if price is None:
        return {"price_per_hour": None, "deployment_price_per_hour": None, "estimated_cost": None}
    try:
        hourly = price * count
        total = hourly * hours
    except OverflowError as error:
        raise AwsError("Projected compute cost is too large") from error
    if not math.isfinite(total):
        raise AwsError("Projected compute cost is too large")
    return {"price_per_hour": price, "deployment_price_per_hour": round(hourly, 6), "estimated_cost": round(total, 6)}


def estimate(
    model_id: str,
    options: EstimateOptions,
    *,
    instance_type: str | None = None,
    tensor_parallel_size: int | None = None,
    safety_margin: float | None = None,
    count: int = 1,
    hours: float = 1,
    token: str | None = None,
) -> dict:
    catalog = load_catalog()
    margin = catalog.safety_margin_default if safety_margin is None else safety_margin
    if not math.isfinite(margin) or not 0 <= margin < 1:
        raise AwsError("--safety-margin must be finite and in [0, 1)")
    if tensor_parallel_size is not None and tensor_parallel_size < 1:
        raise AwsError("--tensor-parallel-size must be positive")
    projection({}, count, hours)
    candidates = [
        item
        for item in catalog.instances
        if (instance_type is None or item["instanceType"] == instance_type)
        and item["numGpu"] >= (tensor_parallel_size or 1)
    ]
    if instance_type and not candidates:
        raise AwsError(f"No catalog GPU layout matches {instance_type} and the requested tensor parallel size")
    layouts = [(tensor_parallel_size or item["numGpu"], item["gpuMemGbPerGpu"]) for item in candidates]
    unique_layouts = list(dict.fromkeys(layouts))
    results = run_hf_mem_batch(
        model_id, unique_layouts, options=options, gpu_memory_utilization=1 - margin, token=token
    )
    by_layout = dict(zip(unique_layouts, results, strict=True))
    matches = []
    for item, layout in zip(candidates, layouts, strict=True):
        result = by_layout[layout]
        if result.fit.fits:
            matches.append(
                {
                    "instance_type": item["instanceType"],
                    "accelerator": item["acceleratorType"],
                    "gpus": item["numGpu"],
                    "gib_per_gpu": item["gpuMemGbPerGpu"],
                    "tensor_parallel_size": layout[0],
                    **projection(item, count, hours),
                    "fit": asdict(result.fit),
                }
            )
    matches.sort(key=lambda row: (row["price_per_hour"] is None, row["price_per_hour"] or 0, row["instance_type"]))
    return {
        "model_id": model_id,
        "estimator": HF_MEM_VERSION,
        "workload": asdict(options),
        "safety_margin": margin,
        "instances": count,
        "hours": hours,
        "currency": "USD",
        "pricing_region": catalog.pricing_region,
        "catalog_date": catalog.last_updated,
        "runtime_verified": False,
        "warnings": list(dict.fromkeys(warning for result in results for warning in result.warnings)),
        "matches": matches,
    }


@dataclass(frozen=True)
class DeploymentPlan:
    model_id: str
    name: str
    region: str
    profile: str | None
    role: str | None
    task: str
    model_server: str
    instance_type: str
    instance_count: int
    tensor_parallel_size: int | None
    startup_timeout_seconds: int
    environment: dict[str, str]
    secret_environment: list[str]
    use_hf_token: bool
    schema: dict
    wait: bool

    def public(self) -> dict:
        plan = asdict(self)
        for key in self.secret_environment:
            del plan["environment"][key]
        return plan


def deployment_plan(
    model_id: str,
    *,
    name: str | None,
    region: str,
    profile: str | None,
    role: str | None,
    instance_type: str | None,
    count: int,
    tensor_parallel_size: int | None,
    environment: dict[str, str],
    token: str | None,
    wait: bool,
) -> DeploymentPlan:
    validate_repo_id(model_id)
    if count < 1:
        raise AwsError("--instances must be positive")
    if name is None:
        prefix = re.sub(r"[^A-Za-z0-9]+", "-", model_id).strip("-")[:54].rstrip("-")
        name = f"{prefix or 'hf'}-{uuid4().hex[:8]}"
    if not ENDPOINT_NAME.fullmatch(name):
        raise AwsError("Endpoint names must be 1–63 ASCII letters, digits, or hyphens with alphanumeric ends")
    if instance_type and not INSTANCE_TYPE.fullmatch(instance_type):
        raise AwsError("Use a SageMaker instance type such as ml.g6.2xlarge")

    custom = validate_environment(environment)
    for key in ("HF_MODEL_ID", "SM_VLLM_MODEL", "SM_VLLM_TENSOR_PARALLEL_SIZE", "HF_TOKEN"):
        if key in custom:
            raise AwsError(f"{key} is set by the deployment plan. Use MODEL_ID, --tensor-parallel-size, or --token")
    task = custom.get("HF_TASK") or None
    use_hf_token = token is not None
    if not task:
        info = HfApi().model_info(model_id, token=token)
        task = info.pipeline_tag or "text-generation"
        use_hf_token = use_hf_token or bool(info.gated or info.private)

    vllm = task == "text-generation"
    if tensor_parallel_size is not None and (not vllm or tensor_parallel_size < 1):
        raise AwsError("--tensor-parallel-size requires text generation and a positive GPU count")
    if instance_type is None:
        matches = estimate(model_id, EstimateOptions(), tensor_parallel_size=tensor_parallel_size, token=token)[
            "matches"
        ]
        if not matches or matches[0]["price_per_hour"] is None:
            raise AwsError("No priced catalog instance fits this model. Run estimate or specify --instance-type")
        instance_type = matches[0]["instance_type"]
    item = next((entry for entry in load_catalog().instances if entry["instanceType"] == instance_type), {})
    if item and tensor_parallel_size and tensor_parallel_size > item["numGpu"]:
        raise AwsError(f"--tensor-parallel-size exceeds the {item['numGpu']} GPUs on {instance_type}")

    parallel = (tensor_parallel_size or item.get("numGpu", 1)) if vllm else None
    defaults = {"HF_TASK": task}
    if vllm:
        defaults.update(SM_VLLM_MODEL=model_id, SM_VLLM_TENSOR_PARALLEL_SIZE=str(parallel))
    else:
        defaults["HF_MODEL_ID"] = model_id
    env = {**custom, **defaults}
    return DeploymentPlan(
        model_id,
        name,
        region,
        profile,
        role,
        task,
        "vllm" if vllm else "auto",
        instance_type,
        count,
        parallel,
        item.get("startupTimeoutSeconds", 900),
        env,
        sorted(key for key in custom if SECRET_NAME.search(key)),
        use_hf_token,
        get_schema_samples(task, model_id),
        wait,
    )


def hub_token(explicit: str | None) -> str | None:
    """Pick up an explicit token or the standard Hugging Face token for gated downloads."""
    return explicit or get_token()
