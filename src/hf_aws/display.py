"""Rich terminal output for SageMaker planning and endpoint commands."""

import shlex
from collections.abc import Iterable
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

from huggingface_hub.utils import HFValidationError, validate_repo_id
from rich import box
from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

STATUS_STYLES = {
    "InService": "bold green",
    "Deleted": "bold green",
    "Failed": "bold red",
    "OutOfService": "bold red",
    "Deleting": "yellow",
    "Creating": "yellow",
    "Updating": "yellow",
    "RollingBack": "yellow",
    "SystemUpdating": "yellow",
    "Submitted": "yellow",
}


def terminal(*, stderr: bool = False) -> Console:
    return Console(stderr=stderr, markup=False, highlight=False)


def activity(message: str, *, enabled: bool = True):
    console = terminal(stderr=True)
    return (
        console.status(Text(message, style="cyan"), spinner="dots")
        if enabled and console.is_terminal
        else nullcontext()
    )


def value(item: Any) -> Text:
    if item is None:
        return Text("—", style="dim")
    if isinstance(item, bool):
        return Text("Yes" if item else "No")
    if isinstance(item, datetime):
        return Text(item.isoformat(sep=" ", timespec="seconds"))
    return Text(str(item))


def status_text(status: str) -> Text:
    return Text(status, style=STATUS_STYLES.get(status, ""))


def usd(amount: float | None) -> str:
    return f"${amount:,.2f}" if amount is not None else "—"


def table(
    console: Console,
    title: str,
    headers: tuple[str, ...],
    rows: Iterable[tuple[RenderableType, ...]],
    *,
    numeric: tuple[str, ...] = (),
    empty: str = "No results.",
) -> None:
    result = Table(
        title=Text(title, style="bold") if title else None,
        title_justify="left",
        box=box.SIMPLE_HEAD,
        header_style="bold",
    )
    for index, header in enumerate(headers):
        result.add_column(header, style="cyan" if index == 0 else "", justify="right" if header in numeric else "left")
    for row in rows:
        result.add_row(*row)
    if result.row_count:
        console.print(result)
    else:
        console.print(empty, style="dim")


def fields(rows: Iterable[tuple[str, RenderableType]]) -> Table:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold")
    grid.add_column()
    for label, item in rows:
        grid.add_row(label, item)
    return grid


def listing(
    title: str, headers: dict[str, str], rows: list[dict], *, empty: str, numeric: tuple[str, ...] = ()
) -> None:
    table(
        terminal(),
        title,
        tuple(headers.values()),
        (tuple(status_text(row[key]) if key == "status" else value(row.get(key)) for key in headers) for row in rows),
        numeric=numeric,
        empty=empty,
    )


def accelerator(row: dict) -> str:
    name = row["accelerator"].removeprefix("NVIDIA_").removeprefix("TESLA_")
    name = name.removesuffix(f"_{row['gib_per_gpu']:g}GB").replace("_", " ")
    return f"{row['gpus']} × {row['gib_per_gpu']:g} GiB {name}"


def _preview_command(result: dict, row: dict, region: str | None, profile: str | None) -> str | None:
    model = result["model_id"]
    workload = result["workload"]
    if workload["revision"] or workload["file"] or workload["format"] not in (None, "safetensors"):
        return None
    try:
        validate_repo_id(model)
    except HFValidationError:
        return None
    if Path(model).exists():
        return None

    args = ["hf", "aws", "deploy", model, "--instance-type", row["instance_type"], "--dry-run"]
    if result["instances"] != 1:
        args.extend(["--instances", str(result["instances"])])
    if row["tensor_parallel_size"] != row["gpus"]:
        args.extend(["--tensor-parallel-size", str(row["tensor_parallel_size"])])
    if region:
        args.extend(["--region", region])
    if profile:
        args.extend(["--profile", profile])
    return shlex.join(args)


def catalog(result: dict, *, region: str | None, profile: str | None) -> None:
    console = terminal()
    is_estimate = "workload" in result
    instance_count, hours = result["instances"], result["hours"]
    catalog_region = result["pricing_region"] or "region unspecified"
    replicas = f"{instance_count} {'replica' if instance_count == 1 else 'replicas'}"
    matches = result["matches"]
    headers = ["Instance", "GPU layout"]
    show_memory = is_estimate and console.width >= 76
    separate_tp = is_estimate and any(row["tensor_parallel_size"] != row["gpus"] for row in matches)
    if is_estimate:
        console.print(Text(f"SageMaker estimate for {result['model_id']}", style="bold"))
        workload = result["workload"]
        context = workload["max_model_len"]
        context_label = f"{context:,} tokens" if context else "model default"
        console.print(
            f"Context {context_label} · Batch {workload['batch_size']} · KV {workload['kv_cache_dtype']} "
            f"· Reserve {result['safety_margin'] * 100:g}%",
            style="dim",
        )
        if show_memory:
            headers.append("VRAM used / total")
        if separate_tp:
            headers.append("TP")
    else:
        headers.append("Startup timeout")
    rate_header = "Est. $/h" if hours == 1 else f"Est. {hours:g}h"
    headers.append(rate_header)
    price_key = "deployment_price_per_hour" if hours == 1 else "estimated_cost"

    rows = []
    for row in matches:
        cells = [row["instance_type"], accelerator(row)]
        if is_estimate:
            if show_memory:
                fit = row["fit"]
                required = fit["required_bytes"] / 1024**3
                total = fit["memory_per_gpu_bytes"] * fit["gpu_count"] / 1024**3
                cells.append(f"{required:,.2f} / {total:g}")
            if separate_tp:
                cells.append(str(row["tensor_parallel_size"]))
        else:
            timeout = row["startup_timeout_seconds"]
            cells.append(f"{timeout:g}s" if timeout is not None else "—")
        rows.append((*cells, usd(row[price_key])))
    table(
        console,
        "" if is_estimate else "SageMaker GPU Instances",
        tuple(headers),
        rows,
        numeric=("VRAM used / total", "TP", "Startup timeout", rate_header),
        empty="No compatible SageMaker GPU instances found.",
    )

    if is_estimate:
        if matches:
            memory_units = " · VRAM in GiB" if show_memory else ""
            console.print(
                f"{len(matches)} of {result['total_matches']} compatible instances · cheapest first{memory_units}",
                style="dim",
            )
            best = matches[0]
            if best["deployment_price_per_hour"] is not None:
                detail = Text(f"{usd(best['deployment_price_per_hour'])}/h · {best['instance_type']}")
                if hours != 1 or instance_count != 1:
                    detail.append(f"\n{usd(best['estimated_cost'])} for {hours:g}h × {replicas}")
                console.print(
                    Panel(detail, title="Cheapest compatible instance", title_align="left", border_style="green")
                )
                if command := _preview_command(result, best, region, profile):
                    console.print("Preview deployment", style="bold")
                    console.print(Text(command, style="cyan"), soft_wrap=True)
            else:
                console.print("Catalog prices are unavailable for the compatible instances.", style="yellow")
        console.print("Runtime support, AWS quota, and capacity are unverified.", style="dim")
        if result["warnings"]:
            stderr = terminal(stderr=True)
            stderr.print("Notes", style="bold yellow")
            for warning in result["warnings"]:
                stderr.print(Text(f"• {warning}", style="dim"))
    elif instance_count != 1 or hours != 1:
        console.print(f"Projected compute for {hours:g}h × {replicas}", style="dim")
    console.print(f"Price snapshot · {result['catalog_date']} · {catalog_region} · USD", style="dim")


def endpoint(title: str, info: dict) -> None:
    console = terminal()
    labels = {
        "name": "Endpoint",
        "status": "Status",
        "model_id": "Model",
        "region": "Region",
        "endpoint_config": "Configuration",
        "arn": "ARN",
        "created_at": "Created",
        "updated_at": "Updated",
        "failure_reason": "Failure reason",
        "variant": "Variant",
        "desired_instances": "Desired instances",
    }
    rows = (
        (label, status_text(info[key]) if key == "status" else value(info[key]))
        for key, label in labels.items()
        if info.get(key) is not None
    )
    border = STATUS_STYLES.get(info.get("status"), "cyan")
    console.print(Panel(fields(rows), title=title, title_align="left", border_style=border))
    variants = info.get("variants", [])
    if variants:
        listing(
            "Production variants",
            {
                "name": "Variant",
                "instance_type": "Instance",
                "model_name": "AWS model",
                "current_instances": "Current",
                "desired_instances": "Desired",
            },
            variants,
            empty="No production variants.",
            numeric=("Current", "Desired"),
        )
    containers = [
        {
            "variant": variant["name"],
            "model_id": container.get("model_id"),
            "runtime": container.get("model_server"),
            "served_model_name": container.get("served_model_name"),
            "image": container.get("image"),
        }
        for variant in variants
        for container in variant.get("containers", [])
    ]
    if containers:
        listing(
            "Model containers",
            {
                "variant": "Variant",
                "model_id": "Hub model",
                "runtime": "Runtime",
                "served_model_name": "Served as",
                "image": "Image",
            },
            containers,
            empty="No model containers.",
        )
