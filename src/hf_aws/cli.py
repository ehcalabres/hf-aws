"""Command-line entry point for hf aws and hf-aws."""

import argparse
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from rich.text import Text
from rich_argparse import RichHelpFormatter

from hf_aws import display
from hf_aws._serve import apply_plan
from hf_aws.deploy_code import deployment_code
from hf_aws.deployments import Deployments, quotas
from hf_aws.environment import load_environment
from hf_aws.errors import AwsError, redact
from hf_aws.hf_mem import EstimateOptions
from hf_aws.planning import deployment_plan, estimate, hub_token, load_catalog, projection

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def json_default(value: object) -> str:
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def emit(args: argparse.Namespace, data: object) -> None:
    if isinstance(data, str) and args.output != "json":
        print(data)
    else:
        print(json.dumps(data, indent=2, ensure_ascii=False, default=json_default))


def emit_endpoint(args: argparse.Namespace, title: str, data: dict) -> None:
    if args.output == "json":
        emit(args, data)
    else:
        display.endpoint(title, data)


def emit_listing(
    args: argparse.Namespace,
    title: str,
    headers: dict[str, str],
    rows: list[dict],
    *,
    empty: str,
    numeric: tuple[str, ...] = (),
) -> None:
    if args.output == "json":
        emit(args, rows)
    else:
        display.listing(title, headers, rows, empty=empty, numeric=numeric)


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def log_tail(value: str) -> int:
    number = positive(value)
    if number > 10000:
        raise argparse.ArgumentTypeError("expected between 1 and 10000 lines")
    return number


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("formatter_class", RichHelpFormatter)
        super().__init__(*args, **kwargs)

    def error(self, message: str) -> None:
        print_error(message, output_format(sys.argv[1:]), code=2)
        raise SystemExit(2)


def shared_options(parser: argparse.ArgumentParser) -> None:
    # Suppress defaults so options can be passed at any command level.
    parser.add_argument(
        "-o",
        "--output",
        choices=("table", "json"),
        default=argparse.SUPPRESS,
        help="Output format (default: table)",
    )
    parser.add_argument(
        "-j",
        "--json",
        action="store_const",
        const="json",
        dest="output",
        default=argparse.SUPPRESS,
        help="Output machine-readable JSON",
    )
    parser.add_argument("-r", "--region", default=argparse.SUPPRESS, help="AWS region")
    parser.add_argument("--profile", default=argparse.SUPPRESS, help="AWS shared config profile")


def parser() -> argparse.ArgumentParser:
    root = ArgumentParser(
        prog="hf aws",
        usage="%(prog)s [OPTIONS] COMMAND [ARGS]...",
        description="Plan and deploy Hub models on Amazon SageMaker AI.",
    )
    shared_options(root)
    commands = root.add_subparsers(prog=root.prog, dest="command", required=True, title="Commands", metavar="")

    def command(name: str, description: str) -> argparse.ArgumentParser:
        child = commands.add_parser(name, help=description, description=description)
        shared_options(child)
        return child

    models = command("models", "Search the Hugging Face Hub or inspect a model.")
    models.add_argument("model_id", nargs="?", metavar="MODEL_ID", help="Hub model ID, or list to browse the Hub")
    models.add_argument("--search", help="Search Hub model names")
    models.add_argument("--task", help="Hub pipeline tag filter for model search")
    models.add_argument("--limit", type=positive, default=20)

    instances = command("instances", "List or inspect GPU instances in the bundled SageMaker catalog.")
    instances.add_argument(
        "instance_type", nargs="?", metavar="INSTANCE_TYPE", help="Inspect a SageMaker SKU such as ml.g6.2xlarge"
    )
    add_cost_options(instances)

    estimation = command("estimate", "Rank SageMaker GPU instances by memory fit and projected cost.")
    estimation.usage = "%(prog)s [OPTIONS] MODEL"
    estimation.add_argument("model_id", metavar="MODEL", help="Hub model ID, URL, or local model path")
    comparison = estimation.add_argument_group("Comparison")
    comparison.add_argument("--instance-type", "-i", help="Estimate one bundled instance type")
    comparison.add_argument("--tensor-parallel-size", type=positive, metavar="GPUS", help="GPUs used per replica")
    comparison.add_argument(
        "--safety-margin", type=float, metavar="FRACTION", help="Fraction of GPU memory to reserve"
    )
    comparison.add_argument("--limit", type=positive, metavar="ROWS", help="Maximum number of matching rows to show")
    workload = estimation.add_argument_group("Workload")
    workload.add_argument("--max-model-len", type=positive, metavar="TOKENS", help="Context length in tokens")
    workload.add_argument("--batch-size", type=positive, metavar="SEQUENCES", default=1, help="Concurrent sequences")
    workload.add_argument(
        "--kv-cache-dtype", default="auto", metavar="DTYPE", help="KV cache dtype, such as auto or fp8"
    )
    workload.add_argument("--max-num-batched-tokens", type=positive, metavar="TOKENS", help="Engine tokens per batch")
    weights = estimation.add_argument_group("Model selection")
    weights.add_argument("--revision", help="Hub branch, tag, or commit")
    weights.add_argument("--file", help="Specific model weights file")
    weights.add_argument("--format", choices=("safetensors", "gguf", "onnx"), help="Model weights format")
    weights.add_argument("--token", help="Hugging Face token for private or gated repositories")
    add_cost_options(estimation)

    deploy = command("deploy", "Deploy a Hub model or preview the deployment.")
    deploy.usage = "%(prog)s [OPTIONS] MODEL_ID"
    deploy.add_argument("model_id", metavar="MODEL_ID", help="Hugging Face Hub model ID")
    deploy.add_argument("--name", "-n", help="Endpoint name (generated by default)")
    deploy.add_argument("--instance-type", "-i", help="SageMaker SKU (defaults to the cheapest catalog fit)")
    deploy.add_argument("--instances", "--instance-count", "-c", type=positive, default=1, help="Endpoint replicas")
    deploy.add_argument("--tensor-parallel-size", type=positive, metavar="GPUS", help="vLLM GPUs used per replica")
    auth = deploy.add_argument_group("Authentication")
    role = auth.add_mutually_exclusive_group()
    role.add_argument("--role", help="IAM execution role ARN or name")
    role.add_argument("--role-name", help="IAM execution role name")
    auth.add_argument("--token", help="Hugging Face token for private or gated repositories")
    runtime = deploy.add_argument_group("Serving runtime")
    runtime.add_argument(
        "--env", action="append", metavar="KEY=VALUE", help="Set a runtime variable. Repeat as needed."
    )
    runtime.add_argument(
        "--env-file", action="append", type=Path, metavar="PATH", help="Load runtime variables from a file."
    )
    execution = deploy.add_argument_group("Execution")
    execution.add_argument("--no-wait", action="store_true", help="Return after submission")
    preview = execution.add_mutually_exclusive_group()
    preview.add_argument("--dry-run", action="store_true", help="Print the resolved plan as JSON")
    preview.add_argument("--print-code", action="store_true", help="Export standalone SageMaker SDK Python code")

    quota = command("quota", "Show applied AWS quotas for real-time SageMaker endpoint instances.")
    quota.add_argument("--instance-type", "-i", help="Show quota for one SageMaker instance type")

    endpoints = command("deployments", "Manage real-time SageMaker endpoints.")
    endpoints.usage = "%(prog)s [OPTIONS] COMMAND [ARGS]..."
    actions = endpoints.add_subparsers(prog=endpoints.prog, dest="action", required=True, title="Commands", metavar="")
    for action, description in {
        "list": "List SageMaker endpoints in the selected region.",
        "status": "Check endpoint status and see any failure reason.",
        "describe": "Inspect the endpoint, production variants, and models.",
        "wait": "Wait until an endpoint is ready or has been deleted.",
        "logs": "Read recent CloudWatch endpoint logs.",
        "invoke": "Send an inference request to an endpoint.",
        "scale": "Set the desired instance count for a production variant.",
        "delete": "Delete a SageMaker endpoint.",
    }.items():
        child = actions.add_parser(action, help=description, description=description)
        shared_options(child)
        if action != "list":
            child.add_argument("name", metavar="ENDPOINT", help="SageMaker endpoint name")
    actions.choices["list"].add_argument("--status", help="AWS endpoint status, such as InService")
    actions.choices["wait"].add_argument(
        "--deleted", action="store_true", help="Wait for deletion instead of readiness"
    )
    actions.choices["logs"].add_argument(
        "--tail", type=log_tail, metavar="LINES", default=100, help="Recent lines to show (1–10000, default: 100)"
    )
    actions.choices["logs"].add_argument("--raw", action="store_true", help="Preserve ANSI escapes")
    scale = actions.choices["scale"]
    scale.add_argument("--instances", type=positive, required=True)
    scale.add_argument("--variant", help="Production variant when the endpoint has more than one")
    delete = actions.choices["delete"]
    delete.add_argument("--yes", "--force", action="store_true", help="Skip confirmation")
    delete.add_argument("--wait", action="store_true", help="Wait until the endpoint is deleted")
    invoke = actions.choices["invoke"]
    request = invoke.add_mutually_exclusive_group(required=True)
    request.add_argument("--input", help="Text input")
    request.add_argument("--payload", help="JSON body, @FILE, or - to read stdin")
    invoke.add_argument("--api", choices=("auto", "vllm", "huggingface"), default="auto")
    invoke.add_argument("--model", help="Served model name for vLLM")
    invoke.add_argument("--max-tokens", type=positive, default=100)
    invoke.add_argument("--variant", help="Target one production variant")
    return root


def add_cost_options(command: argparse.ArgumentParser) -> None:
    cost = command.add_argument_group("Cost projection")
    cost.add_argument(
        "--instances",
        "--instance-count",
        type=positive,
        default=1,
        metavar="COUNT",
        help="Number of endpoint replicas",
    )
    cost.add_argument("--hours", type=float, default=1, metavar="HOURS", help="Continuous running time in hours")


def run_models(args: argparse.Namespace) -> None:
    api = HfApi()
    if args.model_id and args.model_id != "list":
        if args.search or args.task:
            raise AwsError("--search and --task apply to model listing")
        models = [api.model_info(args.model_id)]
    else:
        models = api.list_models(search=args.search, pipeline_tag=args.task, limit=args.limit)
    rows = [
        {"model_id": model.id, "task": model.pipeline_tag, "library": model.library_name, "gated": model.gated}
        for model in models
    ]
    emit_listing(
        args,
        "Hugging Face Hub Models",
        {"model_id": "Model", "task": "Task", "library": "Library", "gated": "Gated"},
        rows,
        empty="No Hub models matched the search.",
    )


def catalog_output(args: argparse.Namespace) -> None:
    if args.command == "instances":
        catalog = load_catalog()
        rows = [
            {
                "instance_type": item["instanceType"],
                "accelerator": item["acceleratorType"],
                "gpus": item["numGpu"],
                "gib_per_gpu": item["gpuMemGbPerGpu"],
                "startup_timeout_seconds": item.get("startupTimeoutSeconds"),
                **projection(item, args.instances, args.hours),
            }
            for item in catalog.instances
            if args.instance_type is None or args.instance_type == item["instanceType"]
        ]
        if args.instance_type and not rows:
            raise AwsError(f"{args.instance_type} is absent from the bundled SageMaker catalog")
        result = {
            "catalog_date": catalog.last_updated,
            "pricing_region": catalog.pricing_region,
            "currency": "USD",
            "hours": args.hours,
            "instances": args.instances,
            "matches": rows,
        }
    else:
        options = EstimateOptions(
            revision=args.revision,
            file=args.file,
            format=args.format,
            max_model_len=args.max_model_len,
            batch_size=args.batch_size,
            kv_cache_dtype=args.kv_cache_dtype,
            max_num_batched_tokens=args.max_num_batched_tokens,
        )
        with display.activity(f"Estimating GPU memory for {args.model_id}…", enabled=args.output != "json"):
            result = estimate(
                args.model_id,
                options,
                instance_type=args.instance_type,
                tensor_parallel_size=args.tensor_parallel_size,
                safety_margin=args.safety_margin,
                count=args.instances,
                hours=args.hours,
                token=hub_token(args.token),
            )
        result["total_matches"] = len(result["matches"])
        result["matches"] = result["matches"][: args.limit]
    if args.output == "json":
        emit(args, result)
    else:
        display.catalog(result, region=args.region, profile=args.profile)


def run_deploy(args: argparse.Namespace, session: boto3.Session) -> None:
    if args.role_name and (args.role_name.startswith("arn:") or "/" in args.role_name):
        raise AwsError("--role-name accepts a plain IAM role name. Use --role for an ARN")
    environment = load_environment(args.env, args.env_file)
    environment_token = environment.pop("HF_TOKEN", None)
    explicit_token = args.token or environment_token
    show_progress = args.output != "json" and not (args.dry_run or args.print_code)
    with display.activity(f"Resolving SageMaker deployment for {args.model_id}…", enabled=show_progress):
        plan = deployment_plan(
            args.model_id,
            name=args.name,
            region=session.region_name,
            profile=args.profile,
            role=args.role or args.role_name or os.environ.get("SAGEMAKER_ROLE_ARN"),
            instance_type=args.instance_type,
            count=args.instances,
            tensor_parallel_size=args.tensor_parallel_size,
            environment=environment,
            token=explicit_token,
            wait=not args.no_wait,
        )
    if args.dry_run:
        print(json.dumps(plan.public(), indent=2))
    elif args.print_code:
        print(deployment_code(plan), end="")
    else:
        with display.activity(
            f"Deploying {plan.model_id} to {plan.name} in {plan.region}…", enabled=args.output != "json"
        ):
            result = apply_plan({**plan.public(), "environment": plan.environment}, token=hub_token(explicit_token))
        emit_endpoint(args, "SageMaker deployment", result)


def run_deployments(args: argparse.Namespace, session: boto3.Session) -> None:
    endpoints = Deployments(session)
    if args.action == "list":
        emit_listing(
            args,
            f"SageMaker Deployments · {session.region_name}",
            {"name": "Endpoint", "status": "Status", "created_at": "Created", "updated_at": "Updated"},
            endpoints.list(args.status),
            empty=f"No SageMaker deployments found in {session.region_name}.",
        )
    elif args.action == "status":
        emit_endpoint(args, "Endpoint status", endpoints.status(args.name))
    elif args.action == "describe":
        emit_endpoint(args, "SageMaker endpoint", endpoints.describe(args.name))
    elif args.action == "wait":
        with display.activity(f"Waiting for {args.name}…", enabled=args.output != "json"):
            result = endpoints.wait(args.name, deleted=args.deleted)
        emit_endpoint(args, "Endpoint status", result)
    elif args.action == "scale":
        emit_endpoint(args, "Endpoint capacity", endpoints.scale(args.name, args.instances, variant=args.variant))
    elif args.action == "delete":
        if not args.yes:
            if not sys.stdin.isatty():
                raise AwsError("Pass --yes to confirm endpoint deletion in a non-interactive session")
            print(f"Delete endpoint {args.name} in {session.region_name}? [y/N] ", end="", file=sys.stderr, flush=True)
            if sys.stdin.readline().strip().lower() not in ("y", "yes"):
                raise AwsError("Deletion cancelled")
        with display.activity(f"Deleting {args.name}…", enabled=args.output != "json"):
            result = endpoints.delete(args.name, wait=args.wait)
        emit_endpoint(args, "Endpoint deletion", result)
    elif args.action == "logs":
        logs = endpoints.logs(args.name, args.tail)
        if not args.raw:
            logs = [ANSI.sub("", line) for line in logs]
        emit(args, logs if args.output == "json" else "\n".join(logs) or "No logs available.")
    elif args.action == "invoke":
        if args.payload is not None:
            data = args.payload
            if data.startswith("@"):
                data = Path(data[1:]).read_text(encoding="utf-8")
            elif data == "-":
                data = sys.stdin.read()
            payload = json.loads(data)
        else:
            payload = endpoints.input_payload(
                args.name, args.input, args.api, args.model, args.max_tokens, args.variant
            )
        with display.activity(f"Invoking {args.name}…", enabled=args.output != "json"):
            result = endpoints.invoke(args.name, payload, variant=args.variant)
        emit(args, result)


def run(args: argparse.Namespace) -> None:
    if args.command == "models":
        run_models(args)
    elif args.command in {"estimate", "instances"}:
        catalog_output(args)
    else:
        session = boto3.Session(profile_name=args.profile, region_name=args.region)
        if session.region_name is None:
            session = boto3.Session(profile_name=args.profile, region_name="us-east-1")
        if args.command == "deploy":
            run_deploy(args, session)
        elif args.command == "quota":
            emit_listing(
                args,
                f"SageMaker Endpoint Quotas · {session.region_name}",
                {
                    "instance_type": "Instance",
                    "quota": "Quota",
                    "adjustable": "Adjustable",
                    "quota_code": "Quota code",
                },
                quotas(session, args.instance_type),
                empty=f"No matching endpoint quota found in {session.region_name}.",
                numeric=("Quota",),
            )
        else:
            run_deployments(args, session)


def output_format(argv: list[str]) -> str:
    selected = "table"
    for index, token in enumerate(argv):
        if token == "--":
            break
        if token in {"-j", "--json"}:
            selected = "json"
        elif token in {"-o", "--output"} and index + 1 < len(argv):
            selected = argv[index + 1]
        elif token.startswith("--output="):
            selected = token.split("=", 1)[1]
        elif token.startswith("-o") and len(token) > 2:
            selected = token[2:].removeprefix("=")
    return selected


def print_error(message: str, output: str, *, code: int = 1) -> None:
    if output == "json":
        print(json.dumps({"error": message, "exit_code": code}), file=sys.stderr)
    else:
        display.terminal(stderr=True).print(Text.assemble(("Error: ", "bold red"), message))


def main() -> int:
    root = parser()
    if len(sys.argv) == 1:
        root.print_help()
        return 0
    args = root.parse_args(namespace=argparse.Namespace(output="table", profile=None, region=None))
    try:
        run(args)
    except (AwsError, BotoCoreError, ClientError, HfHubHTTPError, OSError, RuntimeError, ValueError) as error:
        print_error(redact(str(error), getattr(args, "token", None), os.environ.get("HF_TOKEN")), args.output)
        return 1
    except KeyboardInterrupt:
        print_error("Interrupted", args.output, code=130)
        return 130
    return 0
