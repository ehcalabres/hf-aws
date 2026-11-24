import json
import sys
from unittest.mock import Mock

import pytest

from hf_aws import cli, planning
from hf_aws._serve import deployment_environment
from hf_aws.hf_mem import GpuFit, HfMemEstimate


def invoke(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["hf-aws", *args])
    return cli.main()


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
    assert plan["use_hf_token"] is True
    assert plan["wait"] is False
    assert plan["region"] == "eu-west-1"
    assert plan["role"].endswith(":role/sagemaker")
    monkeypatch.setenv("API_KEY", "replace-at-execution")
    assert scope["deployment_environment"](plan, "new-token") == deployment_environment(plan, "new-token")
    aws.assert_not_called()


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
    assert "Cheapest compatible instance" in output
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
