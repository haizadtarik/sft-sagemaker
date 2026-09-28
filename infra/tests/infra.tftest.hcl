# Offline tests with a mocked AWS provider: `terraform -chdir=infra test`

mock_provider "aws" {
  mock_data "aws_caller_identity" {
    defaults = {
      account_id = "123456789012"
    }
  }

  mock_data "aws_partition" {
    defaults = {
      partition = "aws"
    }
  }

  mock_data "aws_iam_policy_document" {
    defaults = {
      json = "{}"
    }
  }

  mock_resource "aws_iam_role" {
    defaults = {
      arn = "arn:aws:iam::123456789012:role/sft-sagemaker-sagemaker-execution"
    }
  }

  mock_resource "aws_ssm_parameter" {
    defaults = {
      arn = "arn:aws:ssm:us-east-1:123456789012:parameter/sft-sagemaker/hf-token"
    }
  }
}

variables {
  region                = "us-east-1"
  sagemaker_config_path = "../.terraform/test-sagemaker-config.yaml"
}

run "without_hf_token" {
  command = apply

  assert {
    condition     = aws_s3_bucket.artifacts.bucket == "sft-sagemaker-123456789012-us-east-1"
    error_message = "unexpected bucket name"
  }

  assert {
    condition     = length(aws_ssm_parameter.hf_token) == 0
    error_message = "HF token parameter must not be created without a token"
  }

  assert {
    condition     = !strcontains(local_file.sagemaker_config.content, "SFT_HF_TOKEN_SSM_PARAMETER")
    error_message = "config must not reference a missing HF token parameter"
  }

  assert {
    condition = yamldecode(local_file.sagemaker_config.content).SageMaker.PythonSDK.Modules.RemoteFunction.S3RootUri == "s3://sft-sagemaker-123456789012-us-east-1/remote-function"
    error_message = "unexpected S3RootUri"
  }
}

run "with_hf_token" {
  command = apply

  variables {
    hf_token = "hf_dummy"
  }

  assert {
    condition     = length(aws_ssm_parameter.hf_token) == 1
    error_message = "HF token parameter should be created"
  }

  assert {
    condition     = yamldecode(local_file.sagemaker_config.content).SageMaker.PythonSDK.Modules.RemoteFunction.EnvironmentVariables.SFT_HF_TOKEN_SSM_PARAMETER == "/sft-sagemaker/hf-token"
    error_message = "config should point jobs at the HF token parameter"
  }

  assert {
    condition     = yamldecode(local_file.sagemaker_config.content).SageMaker.PythonSDK.Modules.RemoteFunction.RoleArn == aws_iam_role.sagemaker_execution.arn
    error_message = "config should use the execution role"
  }
}
