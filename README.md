# hf-aws

`hf-aws` is a Hugging Face CLI extension for planning and deploying Hub
models on Amazon SageMaker AI real-time endpoints in your AWS account. You can
also run it directly as the `hf-aws` command.

## Installation

From the public repository:

```bash
hf extensions install ehcalabres/hf-aws
hf aws --help
```

From a local checkout, run `uv run hf-aws --help` or install it with
`uv tool install .` and run `hf-aws`. The package requires Python 3.10 or newer.
Memory estimates need [uv](https://docs.astral.sh/uv/getting-started/installation/)
to run the pinned hf-mem worker. Commands with an explicit deployment SKU
can run without the worker.

## Authentication

The CLI uses the standard Boto3 credential chain, including AWS CLI profiles,
SSO, and environment credentials. Pass `--profile NAME` and `--region REGION`
as needed. Region defaults to the Boto3 session region, then `us-east-1`.

For deployment, pass `--role ARN`, `--role-name NAME`, or set
`SAGEMAKER_ROLE_ARN`. The SDK can also discover a role, with
`sagemaker_execution_role` as a fallback. The AWS caller needs permission to
read the role with `iam:GetRole` and pass it with `iam:PassRole`. The role needs
trust for `sagemaker.amazonaws.com` and permissions for the image and model
artifacts. See [SageMaker execution roles](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html).

For gated or private models, accept the model's access terms on the Hub and
run `hf auth login`, set `HF_TOKEN`, or pass `--token` for the current command.
The token is passed to the serving container when needed. Limit access to
SageMaker model details and use a token scoped to read the required models.

## Plan a deployment

```bash
hf aws models list --search gpt2
hf aws models openai-community/gpt2
hf aws instances ml.g6.2xlarge --instances 2 --hours 24
hf aws estimate openai-community/gpt2 --max-model-len 1024 --instances 2 --hours 24
hf aws quota --region us-east-1 --instance-type ml.g6.2xlarge
```

`instances` inspects the bundled GPU catalog, including accelerator topology,
startup timeout, and compute cost. `estimate` runs hf-mem across distinct GPU
layouts and ranks compatible instances by catalog price. It accepts Hub IDs,
URLs, local model paths, and weight selectors supported by hf-mem. Deployment
takes a Hub model ID and lets the serving SDK resolve its weights.

The catalog contains snapshot prices in USD per instance-hour. Prices already
cover every GPU on an instance. The projected compute cost is the price per
instance-hour multiplied by instance count and hours. Check current
[Amazon SageMaker AI pricing](https://aws.amazon.com/sagemaker/ai/pricing/)
for your region. GPU fit, the AWS service quota, and available regional
capacity are separate checks. The quota command shows applied limits returned
by AWS Service Quotas, where available.

```bash
hf aws deploy openai-community/gpt2 --name gpt2-demo --region us-east-1 \
  --instance-type ml.g6.2xlarge --role-name SageMakerExecutionRole --dry-run

hf aws deploy openai-community/gpt2 --name gpt2-demo --region us-east-1 \
  --instance-type ml.g6.2xlarge --role-name SageMakerExecutionRole --print-code > deploy.py
```

Both previews resolve the same ModelBuilder plan as deployment and make no
AWS service calls. Omitting the instance type selects the cheapest priced fit
in the catalog. Text generation uses vLLM and defaults tensor parallelism to the
selected instance's GPU count. Use `--tensor-parallel-size` to override it.
Other tasks let ModelBuilder select the serving runtime. Pass repeatable
`--env KEY=VALUE` or `--env-file PATH` for runtime settings. Inline values
override files. For offline previews with explicit hardware, provide the task
as `--env HF_TASK=text-generation`.

The generated Python script uses the AWS SDK independently of this extension.
Hugging Face tokens and credential-like environment values are omitted from
previews. Supply them through your environment when running the script. Review
ordinary environment values before sharing a plan or script.

## Deploy and manage an endpoint

```bash
hf aws deploy openai-community/gpt2 --name gpt2-demo --region us-east-1 \
  --instance-type ml.g6.2xlarge --role-name SageMakerExecutionRole
hf aws deployments list --region us-east-1
hf aws deployments status gpt2-demo --region us-east-1
hf aws deployments describe gpt2-demo --region us-east-1
hf aws deployments invoke gpt2-demo --input "Hello world" --region us-east-1
hf aws deployments logs gpt2-demo --tail 20 --region us-east-1
hf aws deployments delete gpt2-demo --yes --wait --region us-east-1
```

Deployment waits for `InService` by default. Use `--no-wait` to submit and
`deployments wait ENDPOINT` to wait separately. For custom inference payloads,
use `deployments invoke ENDPOINT --payload @request.json`. Use
`deployments describe` to inspect the endpoint configuration and model names.
Change a production variant's desired instance count with
`deployments scale ENDPOINT --instances COUNT`. For a multi-variant endpoint,
also pass `--variant NAME`.
AWS deletes the endpoint's serving compute when deletion completes. AWS model
records, endpoint configurations, logs, and S3 artifacts have independent
lifecycles and can be managed separately.

The terminal view highlights endpoint status, compares GPU memory and prices,
and summarizes the cheapest compatible instance. Pass `--json` or `-o json`
at any command level for machine-readable results. Errors go to stderr in
the selected format. Dry runs always emit JSON and code export emits Python.

## Development

```bash
uv sync --locked
uv run ruff check
uv run ruff format --check
uv run pytest
```

The tests use stubs for AWS and Hub API calls. The
[agent skill](skills/hf-aws/SKILL.md) documents the planning and endpoint workflow.
