resource "aws_budgets_budget" "monthly" {
  name         = "${var.project_name}-${var.environment}-account-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  # Account-level budget (other resources in the same AWS account count).
  # These alerts DO NOT stop billing or shut down resources automatically.
  dynamic "notification" {
    for_each = toset(["50", "80", "100"])
    content {
      comparison_operator        = "GREATER_THAN"
      notification_type          = "ACTUAL"
      threshold                  = tonumber(notification.value)
      threshold_type             = "PERCENTAGE"
      subscriber_email_addresses = [var.budget_email]
    }
  }
}
