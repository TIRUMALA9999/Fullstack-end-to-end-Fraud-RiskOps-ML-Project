variable "aws_region" {
  description = "AWS Region for FraudGuard infrastructure."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Lowercase, DNS-safe prefix for AWS resources."
  type        = string
  default     = "fraudguard-riskops"

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]*[a-z0-9]$", var.project_name))
    error_message = "project_name must use lowercase letters, numbers, and hyphens."
  }
}

variable "environment" {
  description = "Environment label."
  type        = string
  default     = "dev"
}

variable "budget_email" {
  description = "Email recipient for AWS budget alerts."
  type        = string
  nullable    = false

  validation {
    condition     = can(regex("^[^@ ]+@[^@ ]+\\.[^@ ]+$", var.budget_email))
    error_message = "budget_email must contain a valid email address."
  }
}

variable "monthly_budget_usd" {
  description = "Monthly AWS-account cost alert threshold. Not a hard spending cap."
  type        = number
  default     = 45

  validation {
    condition     = var.monthly_budget_usd >= 1 && var.monthly_budget_usd <= 100
    error_message = "monthly_budget_usd must be between 1 and 100."
  }
}

variable "create_ec2" {
  description = "Explicit opt-in to launch a BILLABLE EC2 instance and public IPv4 address."
  type        = bool
  default     = false
}

variable "instance_type" {
  description = "EC2 instance type once create_ec2 is enabled. Check Free-plan eligibility before use."
  type        = string
  default     = "t3.medium"

  validation {
    condition     = contains(["t3.small", "t3.medium", "t3.large"], var.instance_type)
    error_message = "Approved demo sizes: t3.small, t3.medium, t3.large."
  }
}

variable "root_volume_gb" {
  description = "Encrypted gp3 boot disk capacity. Destroying EC2 will delete this disk."
  type        = number
  default     = 30

  validation {
    condition     = var.root_volume_gb >= 20 && var.root_volume_gb <= 100
    error_message = "root_volume_gb must be between 20 and 100."
  }
}
