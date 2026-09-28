variable "region" {
  description = "AWS region for the bucket, role and training jobs."
  type        = string
}

variable "project_name" {
  description = "Prefix used for resource names and tags."
  type        = string
  default     = "sft-sagemaker"

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]{1,30}[a-z0-9]$", var.project_name))
    error_message = "project_name must be 3-32 lowercase alphanumeric characters or hyphens."
  }
}

variable "hf_token" {
  description = "Optional Hugging Face token, stored as an SSM SecureString for gated models and Hub pushes. Leave empty to skip."
  type        = string
  default     = ""
  sensitive   = true
}

variable "force_destroy_bucket" {
  description = "Allow `terraform destroy` to delete the bucket even if it still contains model artifacts."
  type        = bool
  default     = false
}

variable "remote_function_artifact_retention_days" {
  description = "Days to keep the serialized function/dependency artifacts that the @remote decorator uploads for each job."
  type        = number
  default     = 30
}

variable "sagemaker_config_path" {
  description = "Where to write the generated SageMaker Python SDK defaults file consumed by the launcher."
  type        = string
  default     = "../.sagemaker/config.yaml"
}
