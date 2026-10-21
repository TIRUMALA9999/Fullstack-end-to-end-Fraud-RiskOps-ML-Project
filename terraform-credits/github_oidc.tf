
# GitHub Actions OIDC identity provider.
# Allows keyless authentication from GitHub to AWS.

resource "aws_iam_openid_connect_provider" "github" {
  url = "https://token.actions.githubusercontent.com"

  client_id_list = [
    "sts.amazonaws.com"
  ]
}

# Only this repository's main branch can assume the role.

data "aws_iam_policy_document" "github_ecr_assume" {
  statement {
    sid     = "AllowGitHubMainBranch"
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type = "Federated"
      identifiers = [
        aws_iam_openid_connect_provider.github.arn
      ]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"

      values = [
        "repo:TIRUMALA9999@102666177/Fullstack-end-to-end-Fraud-RiskOps-ML-Project@1427240250:ref:refs/heads/main"
      ]
    }
  }
}

resource "aws_iam_role" "github_ecr_push" {
  name = "fraudguard-riskops-dev-github-ecr-push"

  assume_role_policy = data.aws_iam_policy_document.github_ecr_assume.json
}

# Minimal permissions required to authenticate to ECR
# and push API/dashboard container images.

data "aws_iam_policy_document" "github_ecr_push" {
  statement {
    sid     = "ECRLogin"
    effect  = "Allow"
    actions = ["ecr:GetAuthorizationToken"]

    resources = ["*"]
  }

  statement {
    sid    = "PushApplicationImages"
    effect = "Allow"

    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:CompleteLayerUpload",
      "ecr:DescribeImages",
      "ecr:GetDownloadUrlForLayer",
      "ecr:InitiateLayerUpload",
      "ecr:PutImage",
      "ecr:UploadLayerPart"
    ]

    resources = [
      aws_ecr_repository.app["api"].arn,
      aws_ecr_repository.app["dashboard"].arn
    ]
  }
}

resource "aws_iam_role_policy" "github_ecr_push" {
  name   = "fraudguard-github-ecr-push"
  role   = aws_iam_role.github_ecr_push.id
  policy = data.aws_iam_policy_document.github_ecr_push.json
}

output "github_ecr_push_role_arn" {
  description = "IAM role used by GitHub Actions to publish Docker images"
  value       = aws_iam_role.github_ecr_push.arn
}
