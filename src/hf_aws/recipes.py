"""Load and build SageMaker ModelPackage deployment recipes."""

import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import BucketFile, HfApi, RepoFile, hf_hub_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
    RemoteEntryNotFoundError,
)
from huggingface_hub.utils import HFValidationError, validate_repo_id

from hf_aws.environment import validate_environment
from hf_aws.errors import AwsError

INSTANCE_TYPE = re.compile(r"ml\.[a-z0-9]+(?:-[a-z0-9]+)*\.[a-z0-9]+")
RECIPE_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?")
PART_SUFFIX = re.compile(r"-part-([1-9][0-9]*)\.json\Z")
DEFAULT_RECIPE_SOURCE = "hf://datasets/hf-rev/sagemaker-recipes"
ECR_REGISTRY = re.compile(r"(?P<account>\d{12})\.dkr\.ecr(?:-fips)?\.(?P<region>[\w-]+)\.amazonaws\.com(?:\.cn)?/")
SECRET_NAME = re.compile(
    r"(?:^|_)TOKEN(?:_|$)|PASSWORD|SECRET|CREDENTIAL|API_?KEY|ACCESS_?KEY|PRIVATE_?KEY", re.IGNORECASE
)
IMAGE_FRAMEWORKS = {
    "huggingface-pytorch-inference": "huggingface",
    "huggingface-tensorflow-inference": "huggingface",
    "tei": "huggingface-tei",
    "tei-cpu": "huggingface-tei-cpu",
}


@dataclass(frozen=True)
class RecipeOption:
    source: str
    package: dict
    specification: dict


def _source_parts(source: str) -> tuple[str, str, str]:
    match = re.fullmatch(r"hf://(datasets|buckets)/([^/]+/[^/?#]+)(?:/([^?#]*))?", source)
    if not match:
        raise AwsError("Use hf://datasets/NAMESPACE/REPO or hf://buckets/NAMESPACE/BUCKET for --recipe-source")
    kind, repo_id, prefix = match.groups()
    validate_repo_id(repo_id)
    if prefix:
        prefix = prefix.rstrip("/")
        if any(part in {"", ".", ".."} for part in prefix.split("/")):
            raise AwsError("Use a relative directory prefix in --recipe-source")
    return kind, repo_id, f"{prefix}/" if prefix else ""


def _source_unavailable(error: HfHubHTTPError) -> bool:
    return error.response is not None and error.response.status_code in {401, 403, 404}


def list_recipes(source: str, token: str | None) -> list[dict]:
    kind, repo_id, prefix = _source_parts(source)
    api = HfApi()
    try:
        if kind == "datasets":
            files = api.list_repo_tree(
                repo_id, path_in_repo=prefix.rstrip("/") or None, repo_type="dataset", recursive=True, token=token
            )
            paths = [item.path for item in files if isinstance(item, RepoFile)]
        else:
            files = api.list_bucket_tree(repo_id, prefix=prefix or None, recursive=True, token=token)
            paths = [item.path for item in files if isinstance(item, BucketFile)]
    except HfHubHTTPError as error:
        if _source_unavailable(error):
            raise AwsError(f"Cannot read {source}. Check the source and sign in with hf auth login") from error
        raise

    models: dict[str, list[tuple[int, str]]] = {}
    for path in paths:
        if not path.startswith(prefix) or not path.endswith(".json"):
            continue
        part = PART_SUFFIX.search(path)
        model_id = path[len(prefix) : part.start() if part else -len(".json")]
        try:
            validate_repo_id(model_id)
        except HFValidationError:
            continue
        models.setdefault(model_id, []).append((int(part[1]) if part else 0, path))

    rows = []
    for model_id, entries in sorted(models.items()):
        entries.sort()
        number, path = entries[0]
        row: dict[str, str | list[str]] = {"model_id": model_id, "path": path}
        if number and len(entries) > 1:
            row["parts"] = [entry for _, entry in entries]
        rows.append(row)
    return rows


def _model_files(paths: Iterable[str], filename: str) -> list[str]:
    stem = filename.removesuffix(".json")
    parts = []
    for path in paths:
        if path == filename:
            return [path]
        match = PART_SUFFIX.search(path)
        if match and path[: match.start()] == stem:
            parts.append((int(match[1]), path))
    parts.sort()
    if any(number != expected for expected, (number, _) in enumerate(parts, 1)):
        raise AwsError(f"Incomplete recipe parts for {filename}. Found parts {', '.join(str(n) for n, _ in parts)}")
    return [path for _, path in parts]


def _dataset_recipes(repo_id: str, filename: str, token: str | None) -> list[tuple[dict, str]]:
    source = f"hf://datasets/{repo_id}"
    try:
        path = hf_hub_download(repo_id, filename, repo_type="dataset", token=token)
    except RemoteEntryNotFoundError:
        pass
    else:
        location = f"{source}/{filename}"
        return [(read_recipe(Path(path), location), location)]

    api = HfApi()
    revision = api.dataset_info(repo_id, expand=["sha"], token=token).sha
    if not revision:
        raise AwsError(f"Cannot resolve a dataset revision for {source}")
    try:
        entries = api.list_repo_tree(
            repo_id,
            path_in_repo=filename.rpartition("/")[0] or None,
            repo_type="dataset",
            revision=revision,
            token=token,
        )
        paths = _model_files((entry.path for entry in entries if isinstance(entry, RepoFile)), filename)
    except RemoteEntryNotFoundError:
        return []

    packages = []
    for filename in paths:
        location = f"{source}/{filename}"
        try:
            path = hf_hub_download(repo_id, filename, repo_type="dataset", revision=revision, token=token)
        except (HfHubHTTPError, OSError) as error:
            raise AwsError(f"Cannot download recipe file {location} at revision {revision}") from error
        packages.append((read_recipe(Path(path), location), location))
    return packages


def _bucket_recipes(repo_id: str, filename: str, token: str | None) -> list[tuple[dict, str]]:
    source = f"hf://buckets/{repo_id}"
    api = HfApi()
    with TemporaryDirectory() as directory:
        local = Path(directory, "recipe.json")
        try:
            api.download_bucket_files(repo_id, [(filename, local)], raise_on_missing_files=True, token=token)
        except EntryNotFoundError:
            pass
        else:
            location = f"{source}/{filename}"
            return [(read_recipe(local, location), location)]

        try:
            entries = api.list_bucket_tree(
                repo_id, prefix=filename.removesuffix(".json") + "-part-", recursive=True, token=token
            )
            matches = {item.path: item for item in entries if isinstance(item, BucketFile)}
        except RemoteEntryNotFoundError:
            return []
        files = [
            (matches[path], Path(directory, f"part-{index}.json"))
            for index, path in enumerate(_model_files(matches, filename), 1)
        ]
        if not files:
            return []
        api.download_bucket_files(repo_id, files, raise_on_missing_files=True, token=token)
        return [(read_recipe(path, f"{source}/{item.path}"), f"{source}/{item.path}") for item, path in files]


def load_recipes(
    model_id: str, source: str | None, token: str | None, *, use_default_source: bool = False
) -> dict[str, RecipeOption]:
    implicit = source is None and use_default_source
    if implicit:
        source = DEFAULT_RECIPE_SOURCE
    if not source:
        return {}
    kind, repo_id, prefix = _source_parts(source)
    filename = f"{prefix}{model_id}.json"
    try:
        packages = (_dataset_recipes if kind == "datasets" else _bucket_recipes)(repo_id, filename, token)
    except (LocalEntryNotFoundError, OfflineModeIsEnabled):
        if implicit:
            return {}
        raise
    except HfHubHTTPError as error:
        if implicit and _source_unavailable(error):
            return {}
        raise
    return _recipe_options(packages)


def read_recipe(path, location: str) -> dict:
    try:
        package = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise AwsError(f"Cannot read ModelPackage JSON at {location}") from error
    if not isinstance(package, dict):
        raise AwsError(f"Expected a ModelPackage JSON object at {location}")
    spec = package.get("InferenceSpecification")
    if package.get("ModelApprovalStatus", "Approved") != "Approved":
        raise AwsError(f"The ModelPackage recipe at {location} must be Approved for deployment")
    if "Task" in package and not isinstance(package["Task"], str):
        raise AwsError(f"Expected a ModelPackage Task string in {location}")
    additional = package.get("AdditionalInferenceSpecifications", [])
    if not isinstance(additional, list):
        raise AwsError(f"Invalid AdditionalInferenceSpecifications in {location}")
    names = set()
    for inference in [spec, *additional]:
        containers = inference.get("Containers") if isinstance(inference, dict) else None
        if not isinstance(containers, list) or not 1 <= len(containers) <= 15:
            raise AwsError(f"Expected 1–15 containers per inference specification in {location}")
        if inference is not spec:
            name = inference.get("Name")
            if not isinstance(name, str) or not RECIPE_NAME.fullmatch(name) or name in names:
                raise AwsError(f"Additional inference specifications in {location} need valid, unique names")
            names.add(name)
        for container in containers:
            image = container.get("Image") if isinstance(container, dict) else None
            if not isinstance(image, str) or "/" not in image or any(char.isspace() for char in image):
                raise AwsError(f"Every inference container in {location} requires an Image URI")
            validate_environment(container.get("Environment", {}))
            if (digest := container.get("ImageDigest")) and (
                not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", digest, re.IGNORECASE)
            ):
                raise AwsError(f"Invalid container ImageDigest in {location}")
        instances = inference.get("SupportedRealtimeInferenceInstanceTypes", [])
        if not isinstance(instances, list) or any(
            not isinstance(item, str) or not INSTANCE_TYPE.fullmatch(item) for item in instances
        ):
            raise AwsError(f"Invalid SupportedRealtimeInferenceInstanceTypes in {location}")
    return package


def _recipe_options(packages: list[tuple[dict, str]]) -> dict[str, RecipeOption]:
    names = Counter(
        spec["Name"] for package, _ in packages for spec in package.get("AdditionalInferenceSpecifications", [])
    )
    options = {}
    for part, (package, source) in enumerate(packages, 1):
        primary = "primary" if part == 1 else f"primary-part-{part}"
        if primary in names:
            primary = f"@{primary}"
        options[primary] = RecipeOption(source, package, package["InferenceSpecification"])
        for spec in package.get("AdditionalInferenceSpecifications", []):
            name = spec["Name"]
            if names[name] > 1:
                name = f"{name}@part-{part}"
            options[name] = RecipeOption(source, package, spec)
    return options


def select_recipe(options: dict[str, RecipeOption], name: str | None, instance_type: str | None, model_id: str) -> str:
    primary = next(iter(options))
    hint = f"Run hf aws deploy {model_id} --list-recipes"
    if name is not None:
        if name not in options:
            raise AwsError(f"Unknown recipe '{name}' for {model_id}. {hint}")
        return name
    if instance_type is None:
        return primary

    supported = "SupportedRealtimeInferenceInstanceTypes"
    primary_instances = options[primary].specification.get(supported, [])
    if not primary_instances or instance_type in primary_instances:
        return primary
    matches = [name for name, recipe in options.items() if instance_type in recipe.specification.get(supported, [])]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise AwsError(f"Multiple recipes match {instance_type}: {', '.join(matches)}. Choose one with --recipe")
    raise AwsError(f"No recipe for {model_id} lists {instance_type}. {hint}")


def image_for_region(image: str, region: str, instance_type: str, location: str) -> str:
    registry = ECR_REGISTRY.match(image)
    if registry is None or registry["region"] == region:
        return image

    from sagemaker.core import image_uris

    source_region = registry["region"]
    image_path, digest_separator, digest = image[registry.end() :].partition("@")
    repository, _, tag = image_path.partition(":")
    framework = IMAGE_FRAMEWORKS.get(repository, repository)
    version = tag.partition("-")[0]
    options = {"image_scope": "inference", "instance_type": instance_type}
    if py_version := re.search(r"(?:^|-)(py\d+)(?:-|$)", tag):
        options["py_version"] = py_version[1]
    if framework == "huggingface":
        base = "tensorflow" if "tensorflow" in repository else "pytorch"
        options["base_framework_version"] = f"{base}{version}"
        version = tag.partition("-transformers")[2].partition("-")[0]
    elif framework in {"huggingface-tei", "huggingface-tei-cpu"}:
        version = tag.partition("-tei")[2].partition("-")[0]

    if version and framework in image_uris.list_frameworks():
        try:
            # The Hugging Face vLLM DLC publishes new tags to its regional registries before SDK catalog updates.
            known_version = version
            if framework == "huggingface-vllm" and version not in image_uris.list_versions(framework, "inference"):
                known_version = None
            original = image_uris.retrieve(framework, region=source_region, version=known_version, **options)
            if known_version is None:
                regional = image_uris.retrieve(framework, region=region, **options)
                original_host, _, original_path = original.partition("/")
                regional_host, _, regional_path = regional.partition("/")
                if (
                    original_host.split(".", 1)[0] == regional_host.split(".", 1)[0] == registry["account"]
                    and original_path.partition(":")[0] == regional_path.partition(":")[0] == repository
                ):
                    return f"{regional_host}/{image_path}{digest_separator}{digest}"
            # Confirm catalog ownership and the exact repository and tag before switching registries.
            if original.split(".", 1)[0] == registry["account"] and original.partition("/")[2] == image_path:
                regional = image_uris.retrieve(framework, region=region, version=version, **options)
                if regional.partition("/")[2] == image_path:
                    return f"{regional}{digest_separator}{digest}"
        except ValueError:
            pass
    raise AwsError(
        f"Recipe {location} uses an ECR image in {source_region} with no matching catalog image in {region}. "
        f"Provide the image URI for {region} in the recipe or deploy with --region {source_region}"
    )


def default_package(name: str, model_id: str, image: str, instance_type: str, environment: dict[str, str]) -> dict:
    return {
        "ModelPackageName": name,
        "ModelPackageDescription": f"Hugging Face Hub model {model_id}",
        "InferenceSpecification": {
            "Containers": [{"Image": image, "Environment": environment}],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
            "SupportedRealtimeInferenceInstanceTypes": [instance_type],
        },
        "ModelApprovalStatus": "Approved",
    }
