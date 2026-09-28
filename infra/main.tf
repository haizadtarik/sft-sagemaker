data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  partition  = data.aws_partition.current.partition

  bucket_name              = "${var.project_name}-${local.account_id}-${var.region}"
  remote_function_prefix   = "remote-function"
  model_output_prefix      = "models"
  hf_token_parameter_name  = "/${var.project_name}/hf-token"
  create_hf_token_resource = nonsensitive(var.hf_token != "")

  tags = {
    Project   = var.project_name
    ManagedBy = "terraform"
  }
}

# ---------------------------------------------------------------------------
# S3: remote-function artifacts (pickled function, args, results) and models
# ---------------------------------------------------------------------------

resource "aws_s3_bucket" "artifacts" {
  bucket        = local.bucket_name
  force_destroy = var.force_destroy_bucket
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    id     = "expire-remote-function-artifacts"
    status = "Enabled"

    filter {
      prefix = "${local.remote_function_prefix}/"
    }

    expiration {
      days = var.remote_function_artifact_retention_days
    }

    noncurrent_version_expiration {
      noncurrent_days = 1
    }
  }

  rule {
    id     = "expire-noncurrent-versions"
    status = "Enabled"

    filter {}

    noncurrent_version_expiration {
      noncurrent_days = 30
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

resource "aws_s3_bucket_policy" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  policy = data.aws_iam_policy_document.bucket_tls_only.json

  depends_on = [aws_s3_bucket_public_access_block.artifacts]
}

data "aws_iam_policy_document" "bucket_tls_only" {
  statement {
    sid     = "DenyInsecureTransport"
    effect  = "Deny"
    actions = ["s3:*"]
    resources = [
      aws_s3_bucket.artifacts.arn,
      "${aws_s3_bucket.artifacts.arn}/*",
    ]

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

# ---------------------------------------------------------------------------
# Optional Hugging Face token (read by the training job at startup)
# ---------------------------------------------------------------------------

resource "aws_ssm_parameter" "hf_token" {
  count = local.create_hf_token_resource ? 1 : 0

  name        = local.hf_token_parameter_name
  description = "Hugging Face token used by SFT training jobs"
  type        = "SecureString"
  value       = var.hf_token
}

# ---------------------------------------------------------------------------
# IAM: execution role assumed by the SageMaker training jobs
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "sagemaker_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.account_id]
    }
  }
}

resource "aws_iam_role" "sagemaker_execution" {
  name               = "${var.project_name}-sagemaker-execution"
  description        = "Execution role for SFT training jobs launched with the SageMaker @remote decorator"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume_role.json
}

data "aws_iam_policy_document" "sagemaker_execution" {
  statement {
    sid       = "ListArtifactBucket"
    actions   = ["s3:ListBucket", "s3:GetBucketLocation"]
    resources = [aws_s3_bucket.artifacts.arn]
  }

  statement {
    sid = "ReadWriteArtifacts"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:AbortMultipartUpload",
    ]
    resources = ["${aws_s3_bucket.artifacts.arn}/*"]
  }

  # AWS Deep Learning Container images live in AWS-owned accounts, so the
  # repository ARNs cannot be scoped to this account.
  statement {
    sid = "PullTrainingImages"
    actions = [
      "ecr:GetAuthorizationToken",
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
    ]
    resources = ["*"]
  }

  statement {
    sid = "WriteTrainingLogs"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:DescribeLogStreams",
      "logs:PutLogEvents",
    ]
    resources = [
      "arn:${local.partition}:logs:${var.region}:${local.account_id}:log-group:/aws/sagemaker/*",
    ]
  }

  statement {
    sid       = "PublishMetrics"
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"]

    condition {
      test     = "StringLike"
      variable = "cloudwatch:namespace"
      values   = ["/aws/sagemaker/*", "aws/sagemaker/*"]
    }
  }

  dynamic "statement" {
    for_each = local.create_hf_token_resource ? [1] : []
    content {
      sid       = "ReadHfToken"
      actions   = ["ssm:GetParameter"]
      resources = [aws_ssm_parameter.hf_token[0].arn]
    }
  }
}

resource "aws_iam_role_policy" "sagemaker_execution" {
  name   = "sft-training"
  role   = aws_iam_role.sagemaker_execution.id
  policy = data.aws_iam_policy_document.sagemaker_execution.json
}

# ---------------------------------------------------------------------------
# IAM: least-privilege policy for whoever launches jobs (attach to a user/role
# if the .env credentials should not be administrator credentials)
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "launcher" {
  statement {
    sid = "ManageTrainingJobs"
    actions = [
      "sagemaker:CreateTrainingJob",
      "sagemaker:DescribeTrainingJob",
      "sagemaker:StopTrainingJob",
      "sagemaker:ListTrainingJobs",
      "sagemaker:AddTags",
      "sagemaker:ListTags",
      "sagemaker:Search",
    ]
    resources = ["*"]
  }

  statement {
    sid       = "PassExecutionRole"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.sagemaker_execution.arn]

    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["sagemaker.amazonaws.com"]
    }
  }

  # The SDK's default-bucket check calls ListBuckets before touching the bucket.
  statement {
    sid       = "FindArtifactBucket"
    actions   = ["s3:ListAllMyBuckets"]
    resources = ["*"]
  }

  statement {
    sid       = "ListArtifactBucket"
    actions   = ["s3:ListBucket", "s3:GetBucketLocation"]
    resources = [aws_s3_bucket.artifacts.arn]
  }

  statement {
    sid       = "ReadWriteArtifacts"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = ["${aws_s3_bucket.artifacts.arn}/*"]
  }

  statement {
    sid = "StreamJobLogs"
    actions = [
      "logs:DescribeLogStreams",
      "logs:GetLogEvents",
      "logs:FilterLogEvents",
    ]
    resources = [
      "arn:${local.partition}:logs:${var.region}:${local.account_id}:log-group:/aws/sagemaker/*",
    ]
  }
}

resource "aws_iam_policy" "launcher" {
  name        = "${var.project_name}-launcher"
  description = "Permissions needed to launch and monitor SFT training jobs via the SageMaker @remote decorator"
  policy      = data.aws_iam_policy_document.launcher.json
}

# ---------------------------------------------------------------------------
# SageMaker Python SDK defaults consumed by the @remote decorator
# ---------------------------------------------------------------------------

resource "local_file" "sagemaker_config" {
  filename        = abspath("${path.module}/${var.sagemaker_config_path}")
  file_permission = "0644"
  content = templatefile("${path.module}/templates/sagemaker-config.yaml.tftpl", {
    bucket                  = aws_s3_bucket.artifacts.bucket
    role_arn                = aws_iam_role.sagemaker_execution.arn
    remote_function_s3_root = "s3://${aws_s3_bucket.artifacts.bucket}/${local.remote_function_prefix}"
    model_output_s3_uri     = "s3://${aws_s3_bucket.artifacts.bucket}/${local.model_output_prefix}"
    hf_token_parameter      = local.create_hf_token_resource ? local.hf_token_parameter_name : ""
    tags                    = local.tags
  })
}
