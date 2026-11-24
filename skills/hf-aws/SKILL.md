---
name: hf-aws
description: "Plan and manage Hugging Face Hub model deployments on Amazon SageMaker AI with the hf aws extension."
---

Use `hf aws` after `hf extensions install ehcalabres/hf-aws`, or use the
standalone `hf-aws` executable. Check `hf aws --help` for command options.

- Search Hub models with `hf aws models list --search QUERY`. Use
  `hf aws models MODEL_ID` to check the pipeline task.
- Run `hf aws estimate MODEL_ID --hours HOURS --instances COUNT -o json` to
  compare GPU memory fit and projected compute charges. Check regional
  endpoint quota with `hf aws quota --instance-type SKU --region REGION`.
- Review `hf aws deploy MODEL_ID --name NAME --region REGION --instance-type SKU --dry-run`
  before creating an endpoint. Export equivalent Python with `--print-code`.
- Deploy with the approved AWS profile, region, and execution role.
  Use `--no-wait` to submit without waiting and `hf aws deployments wait NAME`
  to wait later. Check status, logs, or invoke with `hf aws deployments`.
- Delete an endpoint with `hf aws deployments delete NAME --region REGION --wait`.
  Use `--yes` once deletion is confirmed.

Use `-o json` for machine-readable results and errors. Review the bundled
catalog's price date and the desired region's AWS prices before quoting cost.
