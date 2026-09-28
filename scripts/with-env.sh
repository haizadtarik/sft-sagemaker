#!/usr/bin/env bash
# Run a command with variables from .env exported, and map them onto the
# Terraform inputs so the same file drives both IaC and the launcher.
set -euo pipefail

ENV_FILE="${ENV_FILE:-$(dirname "$0")/../.env}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
else
  echo "warning: $ENV_FILE not found; using the current environment" >&2
fi

# Empty values in .env would otherwise shadow ~/.aws credentials or profiles.
for var in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_PROFILE HF_TOKEN; do
  if [[ -z "${!var:-}" ]]; then unset "$var"; fi
done

export AWS_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
export AWS_DEFAULT_REGION="$AWS_REGION"
export TF_VAR_region="$AWS_REGION"
export TF_VAR_hf_token="${HF_TOKEN:-}"
if [[ -n "${PROJECT_NAME:-}" ]]; then export TF_VAR_project_name="$PROJECT_NAME"; fi

exec "$@"
