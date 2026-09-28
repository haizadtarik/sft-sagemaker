output "region" {
  value = var.region
}

output "bucket_name" {
  value = aws_s3_bucket.artifacts.bucket
}

output "execution_role_arn" {
  value = aws_iam_role.sagemaker_execution.arn
}

output "remote_function_s3_root_uri" {
  value = "s3://${aws_s3_bucket.artifacts.bucket}/${local.remote_function_prefix}"
}

output "model_output_s3_uri" {
  value = "s3://${aws_s3_bucket.artifacts.bucket}/${local.model_output_prefix}"
}

output "launcher_policy_arn" {
  description = "Attach to the IAM user/role in .env if it is not an administrator."
  value       = aws_iam_policy.launcher.arn
}

output "hf_token_parameter_name" {
  value = local.create_hf_token_resource ? local.hf_token_parameter_name : null
}

output "sagemaker_config_path" {
  value = local_file.sagemaker_config.filename
}
