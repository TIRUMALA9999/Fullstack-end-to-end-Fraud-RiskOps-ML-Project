data "aws_iam_policy_document" "ec2_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "app" {
  name               = "${var.project_name}-${var.environment}-ec2-role"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume_role.json
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.app.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

data "aws_iam_policy_document" "ec2_ecr_s3" {
  statement {
    sid       = "GetECRAuthenticationToken"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid = "PullFraudGuardImages"
    actions = [
      "ecr:BatchGetImage",
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
    ]
    resources = [for repo in aws_ecr_repository.app : repo.arn]
  }

  statement {
    sid       = "ListArtifactBucket"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.artifacts.arn]
  }

  statement {
    sid = "ReadWriteProjectArtifacts"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
    ]
    resources = [
      "${aws_s3_bucket.artifacts.arn}/models/*",
      "${aws_s3_bucket.artifacts.arn}/backups/*",
    ]
  }

  statement {
    sid     = "ReadApprovedModelRelease"
    actions = ["s3:GetObject"]

    resources = [
      "${aws_s3_bucket.artifacts.arn}/model-releases/lightgbm-shadow-v1/*",
    ]
  }

}

resource "aws_iam_role_policy" "ecr_s3" {
  name   = "fraudguard-ecr-s3-least-privilege"
  role   = aws_iam_role.app.id
  policy = data.aws_iam_policy_document.ec2_ecr_s3.json
}

resource "aws_iam_instance_profile" "app" {
  name = "${var.project_name}-${var.environment}-profile"
  role = aws_iam_role.app.name
}
