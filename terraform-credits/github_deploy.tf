
# GitHub Actions deployment role.
# Reuses the existing restricted GitHub OIDC trust policy.

resource "aws_iam_role" "github_ssm_deploy" {
  name = "fraudguard-riskops-dev-github-ssm-deploy"

  assume_role_policy = data.aws_iam_policy_document.github_ecr_assume.json
}

data "aws_iam_policy_document" "github_ssm_deploy" {
  statement {
    sid    = "SendCommandToFraudGuardEC2"
    effect = "Allow"

    actions = [
      "ssm:SendCommand"
    ]

    resources = [
      "arn:aws:ssm:${var.aws_region}::document/AWS-RunShellScript",
      aws_instance.app[0].arn
    ]
  }

  statement {
    sid    = "ReadDeploymentCommandResults"
    effect = "Allow"

    actions = [
      "ssm:GetCommandInvocation",
      "ssm:ListCommandInvocations"
    ]

    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "github_ssm_deploy" {
  name   = "fraudguard-github-ssm-deploy"
  role   = aws_iam_role.github_ssm_deploy.id
  policy = data.aws_iam_policy_document.github_ssm_deploy.json
}

output "github_ssm_deploy_role_arn" {
  description = "GitHub Actions role for deploying to EC2"
  value       = aws_iam_role.github_ssm_deploy.arn
}
