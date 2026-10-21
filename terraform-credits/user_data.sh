#!/bin/bash
set -euo pipefail

# The first EC2 stage only prepares the Docker host.
# Application deployment, TLS and persistent database storage come later.
dnf install -y docker
systemctl enable --now docker
usermod -aG docker ec2-user
