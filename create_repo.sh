#!/bin/bash

# --- CONFIGURATION ---
ACCOUNT_ID="955423509456"
REGION="us-east-1"
REPO_NAME="black-friday-sales-project"
ECR_URL="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "🚀 Starting Deployment for Plotly Dash & FastAPI: ${REPO_NAME}"

# Create the ECR repository if it doesn't exist
aws ecr describe-repositories --repository-names "${REPO_NAME}" --region ${REGION} >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "🆕 ECR repository not found. Creating repository: ${REPO_NAME}"
    aws ecr create-repository --repository-name "${REPO_NAME}" --region ${REGION}
else
    echo "✅ ECR repository found: ${REPO_NAME}"
fi

# 1. Authenticate Docker with ECR
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ECR_URL}

# 2. Build Single Image (Intel Mac native amd64) & Push
echo "📦 Building Base Image..."

# FIXED COMMAND: No spaces after backslashes
docker buildx build \
  --platform linux/amd64 \
  --provenance=false \
  --sbom=false \
  -t "${ECR_URL}/${REPO_NAME}:latest" \
  --push \
  .

echo "✅ Image pushed successfully!"