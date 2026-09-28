terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.5"
    }
  }
}

# Credentials and region come from the standard AWS environment variables
# (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN, AWS_PROFILE),
# which the Makefile exports from the repository's .env file.
provider "aws" {
  region = var.region

  default_tags {
    tags = local.tags
  }
}
