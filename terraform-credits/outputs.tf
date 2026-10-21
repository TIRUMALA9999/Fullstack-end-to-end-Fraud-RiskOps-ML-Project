output "region" {
  value = var.aws_region
}

output "artifact_bucket_name" {
  value = aws_s3_bucket.artifacts.id
}

output "ecr_repository_urls" {
  value = { for component, repo in aws_ecr_repository.app : component => repo.repository_url }
}

output "vpc_id" {
  value = aws_vpc.fraudguard.id
}

output "subnet_id" {
  value = aws_subnet.public.id
}

output "app_security_group_id" {
  value = aws_security_group.app.id
}

output "instance_id" {
  value = try(aws_instance.app[0].id, null)
}

output "instance_public_ip" {
  value = try(aws_instance.app[0].public_ip, null)
}

output "monthly_budget_usd" {
  value = var.monthly_budget_usd
}
