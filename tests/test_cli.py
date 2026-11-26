import json
import sys
from unittest.mock import Mock

import pytest
from huggingface_hub import BucketFile, RepoFile
from huggingface_hub.errors import EntryNotFoundError, OfflineModeIsEnabled, RemoteEntryNotFoundError

from hf_aws import cli, planning, recipes
from hf_aws._serve import deployment_environment
from hf_aws.hf_mem import GpuFit, HfMemEstimate


def invoke(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["hf-aws", *args])
    return cli.main()


@pytest.fixture(autouse=True)
def offline_recipe_source(monkeypatch):
    monkeypatch.delenv("HF_AWS_RECIPE_SOURCE", raising=False)
    monkeypatch.setattr(recipes, "hf_hub_download", Mock(side_effect=OfflineModeIsEnabled("Offline test")))


def test_catalog_projected_cost(monkeypatch, capsys):
    assert invoke(monkeypatch, "-o", "json", "instances", "ml.g6.2xlarge", "--instances", "2", "--hours", "24") == 0
    data = json.loads(capsys.readouterr().out)
    assert data["currency"] == "USD"
    assert data["matches"][0]["deployment_price_per_hour"] == pytest.approx(2.444)
    assert data["matches"][0]["estimated_cost"] == pytest.approx(58.656)


def test_export_and_dry_run_match_and_exclude_credentials(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("SAGEMAKER_ROLE_ARN", "arn:aws:iam::123456789012:role/sagemaker")
    aws = Mock(side_effect=AssertionError("previews must not call AWS"))
    monkeypatch.setattr(cli, "apply_plan", aws)
    dotenv = tmp_path / "deploy.env"
    dotenv.write_text("HF_TOKEN=sample-file-token\nMODE=from-file\nAPI_KEY=sample-key\n")
    options = (
        "--region",
        "eu-west-1",
        "deploy",
        "openai-community/gpt2",
        "--name",
        "gpt2-preview",
        "--instance-type",
        "ml.g5.12xlarge",
        "--env",
        "HF_TASK=text-generation",
        "--env-file",
        str(dotenv),
        "--env",
        "MODE=override",
        "--token",
        "sample-explicit-token",
        "--no-wait",
    )
    assert invoke(monkeypatch, *options, "--dry-run") == 0
    preview = capsys.readouterr().out
    assert invoke(monkeypatch, *options, "--print-code") == 0
    script = capsys.readouterr().out
    for value in ("sample-file-token", "sample-key", "sample-explicit-token"):
        assert value not in preview
        assert value not in script
    scope = {"__name__": "generated"}
    exec(compile(script, "deploy.py", "exec"), scope)
    plan = json.loads(preview)
    assert scope["PLAN"] == plan
    assert plan["environment"]["SM_VLLM_TENSOR_PARALLEL_SIZE"] == "4"
    assert plan["environment"]["MODE"] == "override"
    assert plan["secret_environment"] == ["API_KEY"]
    assert plan["model_package"]["InferenceSpecification"]["Containers"][0]["Environment"] == plan["environment"]
    assert plan["recipe_source"] == "default"
    assert plan["use_hf_token"] is True
    assert plan["wait"] is False
    assert plan["region"] == "eu-west-1"
    assert plan["role"].endswith(":role/sagemaker")
    monkeypatch.setenv("API_KEY", "replace-at-execution")
    assert scope["deployment_environment"](plan, "new-token") == deployment_environment(plan, "new-token")
    aws.assert_not_called()


@pytest.mark.parametrize("kind", ["datasets", "buckets"])
def test_deploy_loads_hub_recipe_and_exports_resolved_settings(monkeypatch, capsys, tmp_path, kind):
    model_id = "example/model"
    source = f"hf://{kind}/example/recipes/configs"
    filename = f"configs/{model_id}.json"
    path = tmp_path / "recipe.json"
    package = recipes.default_package(
        "example",
        model_id,
        "123456789012.dkr.ecr.us-west-2.amazonaws.com/custom:1",
        "ml.g6.2xlarge",
        {"SM_VLLM_MODEL": model_id, "SM_VLLM_MAX_MODEL_LEN": "1024", "API_KEY": "recipe-secret"},
    )
    path.write_text(json.dumps(package), encoding="utf-8")
    if kind == "datasets":
        download = Mock(return_value=path)
        monkeypatch.setattr(recipes, "hf_hub_download", download)
    else:

        def download_bucket(bucket_id, files, *, raise_on_missing_files, token):
            assert bucket_id == "example/recipes"
            [(remote, target)] = files
            assert remote == filename
            assert raise_on_missing_files
            assert token is None
            target.write_bytes(path.read_bytes())

        download = Mock(side_effect=download_bucket)
        monkeypatch.setattr(recipes, "HfApi", lambda: Mock(download_bucket_files=download))
    monkeypatch.setenv("HF_AWS_RECIPE_SOURCE", source)
    estimate = Mock(side_effect=AssertionError("recipes select their own hardware"))
    monkeypatch.setattr(planning, "estimate", estimate)
    options = (
        "deploy",
        model_id,
        "--region",
        "us-west-2",
        "--name",
        "recipe-test",
        "--env",
        "SM_VLLM_MAX_MODEL_LEN=2048",
    )
    assert invoke(monkeypatch, *options, "--dry-run") == 0
    preview = capsys.readouterr().out
    assert "recipe-secret" not in preview
    plan = json.loads(preview)
    assert plan["recipe_source"] == f"hf://{kind}/example/recipes/{filename}"
    assert plan["instance_type"] == "ml.g6.2xlarge"
    assert plan["environment"]["SM_VLLM_MAX_MODEL_LEN"] == "2048"
    assert invoke(monkeypatch, *options, "--print-code") == 0
    script = capsys.readouterr().out
    assert "recipe-secret" not in script
    scope = {"__name__": "generated"}
    exec(compile(script, "deploy.py", "exec"), scope)
    assert scope["PLAN"] == plan
    if kind == "datasets":
        download.assert_called_with("example/recipes", filename, repo_type="dataset", token=None)
    estimate.assert_not_called()


def test_missing_hub_recipe_falls_back_but_invalid_recipe_fails(monkeypatch, capsys, tmp_path):
    download = Mock(side_effect=EntryNotFoundError("Recipe missing"))
    monkeypatch.setattr(
        recipes, "HfApi", lambda: Mock(download_bucket_files=download, list_bucket_tree=Mock(return_value=[]))
    )
    options = (
        "deploy",
        "deepseek-ai/DeepSeek-V3.2",
        "--name",
        "fallback-preview",
        "--region",
        "us-west-2",
        "--instance-type",
        "ml.p5e.48xlarge",
        "--env",
        "HF_TASK=text-generation",
        "--dry-run",
        "--json",
    )
    assert invoke(monkeypatch, *options) == 0
    default = json.loads(capsys.readouterr().out)
    assert default["recipe_source"] == "default"
    assert default["model_package"]["ModelPackageName"] == "fallback-preview"
    download.assert_not_called()

    assert invoke(monkeypatch, *options, "--recipe-source", "hf://buckets/example/recipes") == 0
    assert json.loads(capsys.readouterr().out) == default
    download.assert_called_once()
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(recipes, "hf_hub_download", Mock(return_value=invalid))
    assert invoke(monkeypatch, *options, "--recipe-source", "hf://datasets/example/recipes") == 1
    result = capsys.readouterr()
    assert result.out == ""
    assert "inference specification" in json.loads(result.err)["error"]


@pytest.mark.parametrize("kind", ["datasets", "buckets"])
def test_recipe_list_filters_models_in_source_prefix(monkeypatch, capsys, kind):
    entry = RepoFile if kind == "datasets" else BucketFile
    files = [
        Mock(spec=entry, path="configs/org/z-model.json"),
        Mock(spec=entry, path="configs/org/a-model.json"),
        Mock(spec=entry, path="configs/org/split-model-part-2.json"),
        Mock(spec=entry, path="configs/org/split-model-part-1.json"),
        Mock(spec=entry, path="configs/recipes.jsonl"),
        Mock(spec=entry, path="configs/org/model/metadata.json"),
        Mock(spec=entry, path="configs/README.md"),
        Mock(spec=entry, path="other/org/model.json"),
        Mock(path="configs/org/folder.json"),
    ]
    api = Mock(list_repo_tree=Mock(return_value=iter(files)), list_bucket_tree=Mock(return_value=iter(files)))
    monkeypatch.setattr(recipes, "HfApi", lambda: api)
    monkeypatch.setattr(cli.boto3, "Session", Mock(side_effect=AssertionError("Listing needs no AWS session")))
    source = f"hf://{kind}/example/recipes/configs"
    command = ("deploy", "--list-recipes") if kind == "datasets" else ("recipes", "list")
    assert invoke(monkeypatch, *command, "--recipe-source", source, "--json") == 0
    assert json.loads(capsys.readouterr().out) == [
        {"model_id": "org/a-model", "path": "configs/org/a-model.json"},
        {
            "model_id": "org/split-model",
            "path": "configs/org/split-model-part-1.json",
            "parts": ["configs/org/split-model-part-1.json", "configs/org/split-model-part-2.json"],
        },
        {"model_id": "org/z-model", "path": "configs/org/z-model.json"},
    ]
    if kind == "datasets":
        api.list_repo_tree.assert_called_once_with(
            "example/recipes", path_in_repo="configs", repo_type="dataset", recursive=True, token=None
        )
    else:
        api.list_bucket_tree.assert_called_once_with("example/recipes", prefix="configs/", recursive=True, token=None)


@pytest.mark.parametrize("kind", ["datasets", "buckets"])
def test_deploy_selects_from_all_recipe_parts(monkeypatch, capsys, tmp_path, kind):
    model = "example/model"
    repo = "example/recipes"
    source = f"hf://{kind}/{repo}/configs"
    stem = f"configs/{model}"
    contents = {}
    for part, instance in enumerate(("ml.g6.2xlarge", "ml.g5.12xlarge"), 1):
        package = recipes.default_package(
            f"recipe-part-{part}",
            model,
            "123456789012.dkr.ecr.us-west-2.amazonaws.com/custom:1",
            instance,
            {"SM_VLLM_MODEL": model, "SM_VLLM_MAX_MODEL_LEN": str(1024 * part)},
        )
        spec = package["InferenceSpecification"]
        package["AdditionalInferenceSpecifications"] = [{**spec, "Name": "long-context"}]
        path = tmp_path / f"part-{part}.json"
        path.write_text(json.dumps(package), encoding="utf-8")
        contents[f"{stem}-part-{part}.json"] = path

    if kind == "datasets":

        def download(repo_id, filename, *, repo_type, token, revision=None):
            assert repo_id == repo and repo_type == "dataset"
            if filename == f"{stem}.json":
                response = Mock(status_code=404, headers={}, text="", request=None)
                raise RemoteEntryNotFoundError("Recipe missing", response=response)
            assert revision == "pinned-revision"
            return contents[filename]

        api = Mock(dataset_info=Mock(return_value=Mock(sha="pinned-revision")))
        entries = [Mock(spec=RepoFile, path=path) for path in reversed(contents)]
        api.list_repo_tree.return_value = entries
        monkeypatch.setattr(recipes, "hf_hub_download", download)
    else:

        def download(repo_id, files, *, raise_on_missing_files, token):
            assert repo_id == repo and raise_on_missing_files
            for item, target in files:
                if isinstance(item, str):
                    raise EntryNotFoundError("Recipe missing")
                target.write_bytes(contents[item.path].read_bytes())

        api = Mock(download_bucket_files=download)
        entries = [Mock(spec=BucketFile, path=path) for path in reversed(contents)]
        api.list_bucket_tree.return_value = entries
    monkeypatch.setattr(recipes, "HfApi", lambda: api)
    args = ("deploy", model, "--recipe-source", source, "--region", "us-west-2", "--json")
    assert invoke(monkeypatch, *args, "--list-recipes") == 0
    rows = json.loads(capsys.readouterr().out)
    names = ["primary", "long-context@part-1", "primary-part-2", "long-context@part-2"]
    assert [row["name"] for row in rows] == names
    assert [row["name"] for row in rows if row["primary"]] == ["primary"]
    assert rows[2]["recipe_source"] == f"{source}/{model}-part-2.json"

    assert invoke(monkeypatch, *args, "--dry-run", "--recipe", "primary-part-2") == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["recipe_source"] == rows[2]["recipe_source"]
    assert plan["instance_type"] == "ml.g5.12xlarge"
    assert plan["environment"]["SM_VLLM_MAX_MODEL_LEN"] == "2048"
    assert plan["model_package"]["ModelPackageName"] == "recipe-part-2"
    assert plan["available_recipes"] == names

    for item in entries:
        if item.path.endswith("-part-2.json"):
            item.path = f"{stem}-part-3.json"
    assert invoke(monkeypatch, *args, "--dry-run") == 1
    assert "Incomplete recipe parts" in json.loads(capsys.readouterr().err)["error"]


def test_default_recipe_and_inspect_resolve_image_in_the_session_region(monkeypatch, capsys, tmp_path):
    config = tmp_path / "aws-config"
    config.write_text("[profile recipe-test]\nregion = eu-south-1\n", encoding="utf-8")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    model_id = "example/model"
    image_path = "huggingface-vllm:0.17.0-transformers4.57.5-gpu-py312-cu129-ubuntu22.04"
    package = recipes.default_package(
        "example",
        model_id,
        f"763104351884.dkr.ecr.us-west-2.amazonaws.com/{image_path}",
        "ml.p5e.48xlarge",
        {"SM_VLLM_MODEL": model_id, "SM_VLLM_TENSOR_PARALLEL_SIZE": "8", "API_KEY": "recipe-secret"},
    )
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(package), encoding="utf-8")
    download = Mock(return_value=path)
    monkeypatch.setattr(recipes, "hf_hub_download", download)
    settings = ("--profile", "recipe-test", "--role-name", "SageMakerExecutionRole", "--env", "MODE=preview")
    options = ("deploy", model_id, "--name", "recipe-preview", *settings)
    assert invoke(monkeypatch, *options, "--dry-run") == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["region"] == "eu-south-1"
    assert plan["recipe_source"] == f"hf://datasets/hf-rev/sagemaker-recipes/{model_id}.json"
    download.assert_called_once_with("hf-rev/sagemaker-recipes", f"{model_id}.json", repo_type="dataset", token=None)
    container = plan["model_package"]["InferenceSpecification"]["Containers"][0]
    assert container["Image"] == f"692866216735.dkr.ecr.eu-south-1.amazonaws.com/{image_path}"
    assert container["Environment"]["SM_VLLM_TENSOR_PARALLEL_SIZE"] == "8"

    assert invoke(monkeypatch, "recipes", "inspect", model_id, *settings, "--json") == 0
    inspect = capsys.readouterr().out
    assert "recipe-secret" not in inspect
    assert json.loads(inspect) == {
        "recipe_source": plan["recipe_source"],
        "recipe_name": "primary",
        "available_recipes": ["primary"],
        "role": "SageMakerExecutionRole",
        "instance_type": "ml.p5e.48xlarge",
        "containers": [
            {"image_uri": container["Image"], "environment": {**container["Environment"], "API_KEY": "[redacted]"}}
        ],
    }
    monkeypatch.setenv("COLUMNS", "72")
    assert invoke(monkeypatch, "recipes", "inspect", model_id, *settings) == 0
    output = capsys.readouterr().out
    assert container["Image"] in output
    assert "recipe-secret" not in output

    assert invoke(monkeypatch, *options, "--print-code") == 0
    scope = {"__name__": "generated"}
    exec(compile(capsys.readouterr().out, "deploy.py", "exec"), scope)
    assert scope["PLAN"] == plan

    assert invoke(monkeypatch, *options, "--region", "us-west-2", "--dry-run") == 0
    override = json.loads(capsys.readouterr().out)
    assert override["region"] == "us-west-2"
    assert override["model_package"]["InferenceSpecification"]["Containers"][0]["Image"] == (
        f"763104351884.dkr.ecr.us-west-2.amazonaws.com/{image_path}"
    )


def test_deploy_lists_and_selects_named_recipes(monkeypatch, capsys, tmp_path):
    model_id = "example/model"
    image = "123456789012.dkr.ecr.us-west-2.amazonaws.com/custom:1"
    package = recipes.default_package(
        "example", model_id, image, "ml.g6.2xlarge", {"SM_VLLM_MODEL": model_id, "SM_VLLM_MAX_MODEL_LEN": "1024"}
    )
    package["AdditionalInferenceSpecifications"] = [
        {
            "Name": name,
            "SupportedRealtimeInferenceInstanceTypes": [instance],
            "Containers": [
                {
                    "Image": image,
                    "Environment": {
                        "SM_VLLM_MODEL": model_id,
                        "SM_VLLM_MAX_MODEL_LEN": context,
                        "SM_VLLM_ENABLE_EXPERT_PARALLEL": "true",
                    },
                }
            ],
        }
        for name, instance, context in (
            ("expert", "ml.g6.2xlarge", "4096"),
            ("medium", "ml.g5.12xlarge", "8192"),
            ("long", "ml.g5.12xlarge", "16384"),
            ("largest", "ml.p5en.48xlarge", "65536"),
        )
    ]
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(package), encoding="utf-8")
    monkeypatch.setattr(recipes, "hf_hub_download", Mock(return_value=path))
    options = ("deploy", model_id, "--region", "us-west-2")
    with monkeypatch.context() as listing:
        listing.setattr(cli.boto3, "Session", Mock(side_effect=AssertionError("Listing needs no AWS session")))
        assert invoke(listing, *options, "--list-recipes", "--json") == 0
    rows = json.loads(capsys.readouterr().out)
    assert [row["name"] for row in rows] == ["primary", "expert", "medium", "long", "largest"]
    assert rows[0]["primary"] is True
    assert rows[1]["context_tokens"] == "4096"
    assert rows[1]["features"] == ["Expert parallel"]

    assert invoke(monkeypatch, *options, "--inspect", "--json", "--instance-type", "ml.g6.2xlarge") == 0
    assert json.loads(capsys.readouterr().out)["recipe_name"] == "primary"
    assert invoke(monkeypatch, *options, "--inspect", "--json", "--instance-type", "ml.p5en.48xlarge") == 0
    assert json.loads(capsys.readouterr().out)["recipe_name"] == "largest"
    assert invoke(monkeypatch, *options, "--inspect", "--json", "--instance-type", "ml.g5.12xlarge") == 1
    assert "Multiple recipes match" in json.loads(capsys.readouterr().err)["error"]

    assert invoke(monkeypatch, *options, "--recipe", "expert", "--dry-run") == 0
    selected = json.loads(capsys.readouterr().out)
    spec = selected["model_package"]["InferenceSpecification"]
    assert selected["recipe_name"] == "expert"
    assert spec["Containers"][0]["Environment"]["SM_VLLM_MAX_MODEL_LEN"] == "4096"
    assert "Name" not in spec
    assert "AdditionalInferenceSpecifications" not in selected["model_package"]


def test_json_errors_are_on_stderr(monkeypatch, capsys):
    assert invoke(monkeypatch, "instances", "ml.g6.2xlarge", "--hours", "nan", "-o", "json") == 1
    result = capsys.readouterr()
    assert result.out == ""
    assert "finite and positive" in json.loads(result.err)["error"]


def test_estimate_reuses_layouts_and_auto_deploy_selects_same_price_order(monkeypatch, capsys):
    fits = Mock(
        side_effect=lambda model, layouts, **kwargs: [
            HfMemEstimate(GpuFit(True, 1, count, int(gib * 1024**3), 1, 1024), ("estimate note",))
            for count, gib in layouts
        ]
    )
    monkeypatch.setattr(planning, "run_hf_mem_batch", fits)
    assert (
        invoke(
            monkeypatch,
            "estimate",
            "openai-community/gpt2",
            "--tensor-parallel-size",
            "1",
            "--limit",
            "1",
            "-o",
            "json",
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["total_matches"] > 1
    assert output["warnings"] == ["estimate note"]
    assert not output["runtime_verified"]
    assert len(output["matches"]) == 1
    _, layouts = fits.call_args.args
    assert len(layouts) == len(set(layouts))
    assert all(count == 1 for count, _ in layouts)
    assert (
        invoke(
            monkeypatch,
            "deploy",
            "openai-community/gpt2",
            "--tensor-parallel-size",
            "1",
            "--env",
            "HF_TASK=text-generation",
            "--dry-run",
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["instance_type"] == output["matches"][0]["instance_type"]


def test_deployment_rejects_invalid_gpu_topology(monkeypatch, capsys):
    assert (
        invoke(
            monkeypatch,
            "deploy",
            "openai-community/gpt2",
            "-i",
            "ml.g6.2xlarge",
            "--tensor-parallel-size",
            "2",
            "--env",
            "HF_TASK=text-generation",
            "--dry-run",
            "-o",
            "json",
        )
        == 1
    )
    assert "exceeds the 1 GPUs" in json.loads(capsys.readouterr().err)["error"]


def test_estimate_terminal_summary_and_json_alias(monkeypatch, capsys):
    fits = Mock(
        side_effect=lambda model, layouts, **kwargs: [
            HfMemEstimate(GpuFit(True, 2 * 1024**3, count, int(gib * 1024**3), 1, 1024), ()) for count, gib in layouts
        ]
    )
    monkeypatch.setattr(planning, "run_hf_mem_batch", fits)
    monkeypatch.setenv("COLUMNS", "100")
    options = ("estimate", "openai-community/gpt2", "--instances", "2", "--hours", "24", "--limit", "1")
    assert invoke(monkeypatch, *options) == 0
    output = capsys.readouterr().out
    heading = next(line for line in output.splitlines() if "Cheapest compatible instance" in line)
    assert heading.startswith("╭") and heading.endswith("╮")
    assert len(heading) < 100
    assert "VRAM used / total" in output
    assert "$58.66 for 24h × 2 replicas" in output
    assert "hf aws deploy openai-community/gpt2 --instance-type ml.g6.2xlarge --dry-run" in output
    assert invoke(monkeypatch, "--json", *options) == 0
    output = capsys.readouterr()
    assert json.loads(output.out)["matches"][0]["estimated_cost"] == pytest.approx(58.656)
    assert output.err == ""


def test_endpoint_failure_details_are_rendered_as_plain_text(monkeypatch, capsys):
    data = {"name": "test", "status": "Failed", "failure_reason": "Container [app] failed", "region": "us-east-1"}
    monkeypatch.setattr(cli, "Deployments", lambda session: Mock(status=Mock(return_value=data)))
    options = ("deployments", "status", "test", "--region", "us-east-1")
    assert invoke(monkeypatch, *options) == 0
    output = capsys.readouterr().out
    assert "Endpoint status" in output
    assert "Failure reason" in output
    assert "Container [app] failed" in output
    assert invoke(monkeypatch, *options, "-j") == 0
    assert json.loads(capsys.readouterr().out) == data


def test_json_alias_applies_to_nested_argument_errors(monkeypatch, capsys):
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, "deployments", "logs", "test", "--tail", "10001", "--json")
    assert error.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err)["error"].endswith("expected between 1 and 10000 lines")
