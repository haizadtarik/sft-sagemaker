PYTHON ?= .venv/bin/python
WITH_ENV := ./scripts/with-env.sh
TF := $(WITH_ENV) terraform -chdir=infra
ARGS ?=

.PHONY: help setup setup-train whoami infra-init infra-plan infra infra-output infra-destroy train train-local check

help:
	@echo "setup         Create .venv with launcher dependencies"
	@echo "setup-train   Also install training dependencies (for train-local)"
	@echo "whoami        Show the AWS identity loaded from .env"
	@echo "infra-plan    terraform plan"
	@echo "infra         terraform apply (S3 bucket, IAM roles, SSM token, .sagemaker/config.yaml)"
	@echo "infra-destroy terraform destroy"
	@echo "train         Run SFT as a SageMaker training job   (ARGS='--max-steps 100 ...')"
	@echo "train-local   Tiny local smoke test of the same training function"
	@echo "check         terraform fmt/validate and Python compile check"

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -r requirements.txt

setup-train: setup
	$(PYTHON) -m pip install -r requirements-train.txt torch

whoami:
	$(WITH_ENV) $(PYTHON) -c "import boto3; print(boto3.client('sts').get_caller_identity()['Arn'])"

infra-init:
	$(TF) init -input=false

infra-plan: infra-init
	$(TF) plan -input=false

infra: infra-init
	$(TF) apply -input=false $(TF_APPLY_FLAGS)

infra-output:
	$(TF) output

infra-destroy: infra-init
	$(TF) destroy -input=false

train:
	$(WITH_ENV) $(PYTHON) -m sft_sagemaker.launch $(ARGS)

train-local:
	$(WITH_ENV) $(PYTHON) -m sft_sagemaker.launch --local \
		--model-name-or-path trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 \
		--max-train-samples 32 --max-steps 4 --per-device-train-batch-size 2 \
		--gradient-accumulation-steps 1 --max-length 256 --logging-steps 1 $(ARGS)

check:
	terraform -chdir=infra fmt -check -recursive
	terraform -chdir=infra init -backend=false -input=false >/dev/null
	terraform -chdir=infra validate
	terraform -chdir=infra test
	$(PYTHON) -m compileall -q sft_sagemaker
