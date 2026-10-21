resource "aws_vpc" "fraudguard" {
  cidr_block           = "10.44.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = { Name = "${var.project_name}-${var.environment}-vpc" }
}

resource "aws_internet_gateway" "fraudguard" {
  vpc_id = aws_vpc.fraudguard.id
  tags   = { Name = "${var.project_name}-${var.environment}-igw" }
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.fraudguard.id
  cidr_block              = "10.44.10.0/24"
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = false

  tags = { Name = "${var.project_name}-${var.environment}-public" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.fraudguard.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.fraudguard.id
  }

  tags = { Name = "${var.project_name}-${var.environment}-public-rt" }
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

resource "aws_security_group" "app" {
  name_prefix = "${var.project_name}-${var.environment}-"
  description = "FraudGuard EC2 - no inbound traffic until secure HTTPS is configured"
  vpc_id      = aws_vpc.fraudguard.id

  # Deliberately NO ingress rules; SSH/HTTP/Streamlit are not public.

  # Public HTTP for HTTPS redirects and certificate validation.
  ingress {
    description = "Public HTTP - Caddy"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Public HTTPS for authenticated FraudGuard dashboard.
  ingress {
    description = "Public HTTPS - Caddy"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "Allow outbound access for SSM, updates, and ECR"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project_name}-${var.environment}-app-sg" }
}
