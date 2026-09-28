# sft-sagemaker

Supervised fine-tuning (SFT) of causal LMs with Hugging Face [TRL `SFTTrainer`](https://huggingface.co/docs/trl/en/sft_trainer),
executed as SageMaker training jobs through the [`@remote` decorator](https://docs.aws.amazon.com/sagemaker/latest/dg/train-remote-decorator.html).
AWS infrastructure is Terraform; credentials come from `.env`.

```
.env ──► scripts/with-env.sh ──► terraform (infra/) ──► S3 bucket, IAM roles, SSM HF token
                             │                     └──► .sagemaker/config.yaml (SDK defaults)
                             └──► python -m sft_sagemaker.launch
                                     remote(train)(cfg) ──► SageMaker training job
                                                            PyTorch 2.8 DLC + requirements-train.txt
                                                            TRL SFTTrainer ──► s3://<bucket>/models/<job>/
```

## Layout

| Path | Purpose |
| --- | --- |
| `infra/` | Terraform: artifact bucket, SageMaker execution role, launcher IAM policy, optional SSM HF token, generated SDK config |
| `sft_sagemaker/config.py` | `TrainConfig` dataclass (every field is also a CLI flag) |
| `sft_sagemaker/train.py` | `train(cfg)`: loads the dataset, runs `SFTTrainer` (full or LoRA), uploads the model to S3 |
| `sft_sagemaker/launch.py` | CLI that runs `train` locally or wraps it with `sagemaker.remote_function.remote` |
| `requirements.txt` / `requirements-train.txt` | Launcher deps / deps installed inside the job |

## Prerequisites

- Python **3.12** locally. The `@remote` decorator requires the job image's Python to match the local interpreter, and the default image (PyTorch 2.8 training DLC) is py312. Use `--image-uri` for anything else.
- Terraform >= 1.5 (tests need >= 1.7).
- AWS credentials allowed to create S3/IAM/SSM resources (for `make infra`), and a SageMaker quota for the chosen instance type (default `ml.g5.2xlarge`, 1x A10G 24 GB). Check *Service Quotas → Amazon SageMaker → "ml.g5.2xlarge for training job usage"*.

## Quick start

```bash
cp .env.example .env        # fill in AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (or AWS_PROFILE), AWS_REGION, optional HF_TOKEN
make setup                  # .venv with the launcher dependencies
make whoami                 # sanity-check the credentials from .env

make infra-plan
make infra                  # terraform apply; writes .sagemaker/config.yaml

make train ARGS="--max-steps 50"
```

`make train` blocks while the job runs, streams its CloudWatch logs, and prints the object returned by `train()` (metrics and the S3 URI of the saved model). The job name prefix is `sft`, so it is easy to find in the SageMaker console.

### Examples

```bash
# Full fine-tune of the default model (Qwen/Qwen3-0.6B) on trl-lib/Capybara
make train

# LoRA on a larger model, multi-GPU with torchrun
make train ARGS="--model-name-or-path Qwen/Qwen3-4B --use-peft --learning-rate 2e-4 \
  --instance-type ml.g5.12xlarge --torchrun --nproc-per-node 4"

# Your own dataset (any Hub dataset in "messages", "text" or prompt/completion format)
make train ARGS="--dataset-name my-org/my-sft-data --dataset-eval-split test --eval-steps 100 --assistant-only-loss"

# Keep the instance warm for 30 min so the next job skips provisioning (billed while idle)
make train ARGS="--keep-alive-seconds 1800"

# Push the result to the Hub (requires HF_TOKEN in .env before `make infra`)
make train ARGS="--push-to-hub --hub-model-id my-org/qwen3-sft"
```

Run `.venv/bin/python -m sft_sagemaker.launch --help` for every option.

### Local smoke test

`train()` is the exact function that runs on SageMaker, so it can be exercised locally on CPU with a tiny model:

```bash
make setup-train
make train-local            # tiny Qwen2 model, 4 steps
```

## Infrastructure

`make infra` creates, in `AWS_REGION`:

- **S3 bucket** `<project>-<account>-<region>`: private, SSE-S3, versioned, TLS-only. `remote-function/` holds the pickled function, arguments, dependencies and results the decorator uploads per job (expired after 30 days). `models/<job-name>/` holds trained models.
- **IAM role** `<project>-sagemaker-execution`, assumed by the training jobs: read/write on the bucket, pull DLC images from ECR, write CloudWatch logs and metrics, and read the HF token parameter.
- **IAM policy** `<project>-launcher`: the minimum needed to run `make train` (create/describe/stop training jobs, `iam:PassRole` on the execution role, bucket access, log streaming). Attach it to the `.env` identity if that identity should not be an administrator.
- **SSM SecureString** `/<project>/hf-token`, only if `HF_TOKEN` is set. Jobs read it at startup, so the token never shows up in the training job's environment variables in the console.
- **`.sagemaker/config.yaml`**: SageMaker Python SDK defaults (role, S3 root, environment variables, tags). The launcher points `SAGEMAKER_USER_CONFIG_OVERRIDE` at it.

Terraform state is local (`infra/terraform.tfstate`, git-ignored). It contains the HF token if one is set; add a remote backend if several people share the stack. `make infra-destroy` tears everything down. The bucket is only deleted when empty unless `TF_VAR_force_destroy_bucket=true`.

## How `.env` is used

`scripts/with-env.sh` exports `.env` and maps it onto Terraform inputs (`AWS_REGION` → `TF_VAR_region`, `HF_TOKEN` → `TF_VAR_hf_token`, `PROJECT_NAME` → `TF_VAR_project_name`). Empty keys are unset so they don't mask `~/.aws` profiles. Terraform, boto3 and the SageMaker SDK all read the standard `AWS_*` variables.

`.env` is never uploaded: `include_local_workdir` normally ships the whole working directory, and the launcher filters it down to the `.py` files of `sft_sagemaker/`.

## Checks

```bash
make check    # terraform fmt/validate, terraform test (mocked AWS provider), Python compile
```
