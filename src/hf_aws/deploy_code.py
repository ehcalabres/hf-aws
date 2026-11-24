"""Export the resolved settings and the same deploy implementation as a standalone script."""

from importlib.resources import files
from pprint import pformat

from hf_aws.planning import DeploymentPlan


def deployment_code(plan: DeploymentPlan) -> str:
    runtime = files("hf_aws").joinpath("_serve.py").read_text(encoding="utf-8")
    return (
        "# Install: pip install 'boto3>=1.42.36,<2' 'huggingface-hub>=1.3.4,<3' "
        "'sagemaker-core>=2.29.0,<3' 'sagemaker-serve>=1.29.0,<2'\n"
        f"{runtime}\n\n"
        f"PLAN = {pformat(plan.public(), sort_dicts=False)}\n\n"
        'if __name__ == "__main__":\n'
        "    print(json.dumps(apply_plan(PLAN), indent=2))\n"
    )
