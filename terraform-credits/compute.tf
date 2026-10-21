data "aws_ssm_parameter" "al2023_ami" {
  count = var.create_ec2 ? 1 : 0
  name  = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
}

resource "aws_instance" "app" {
  count = var.create_ec2 ? 1 : 0

  ami                         = data.aws_ssm_parameter.al2023_ami[0].value
  instance_type               = var.instance_type
  subnet_id                   = aws_subnet.public.id
  vpc_security_group_ids      = [aws_security_group.app.id]
  associate_public_ip_address = true
  iam_instance_profile        = aws_iam_instance_profile.app.name
  monitoring                  = false
  user_data                   = file("${path.module}/user_data.sh")

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
  }

  credit_specification {
    cpu_credits = "standard"
  }

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.root_volume_gb
    encrypted             = true
    delete_on_termination = true
  }

  tags = { Name = "${var.project_name}-${var.environment}-app" }

  depends_on = [
    aws_iam_role_policy_attachment.ssm,
    aws_iam_role_policy.ecr_s3,
  ]

  lifecycle {
    # Publishing a newer AL2023 AMI does not silently replace the VM.
    ignore_changes = [ami]
  }
}
