#!/bin/bash

# --- CONFIGURATION ---
ACCOUNT_ID="955423509456"
REGION="us-east-1"
REPO_NAME="black-friday-sales-project"
ECR_URL="955423509456.dkr.ecr.us-east-1.amazonaws.com"

echo "🚀 Starting ECR Image Push: ${REPO_NAME}"

# 1. Create ECR repository if it doesn't exist
aws ecr describe-repositories --repository-names "${REPO_NAME}" --region ${REGION} >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "🆕 Creating ECR repository: ${REPO_NAME}"
    aws ecr create-repository --repository-name "${REPO_NAME}" --region ${REGION}
fi

# 2. Authenticate Docker
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ECR_URL}

# 3. Build Single Image (AMD64 for Lambda)
echo "📦 Building Base Image..."
docker build --platform linux/amd64 -t ${REPO_NAME}:latest .

# 4. Push Tags
echo "📤 Pushing tags: api-v1 and eda-v1..."
docker tag ${REPO_NAME}:latest ${ECR_URL}/${REPO_NAME}:api-v1
docker tag ${REPO_NAME}:latest ${ECR_URL}/${REPO_NAME}:eda-v1

docker push ${ECR_URL}/${REPO_NAME}:api-v1
docker push ${ECR_URL}/${REPO_NAME}:eda-v1

echo "✅ Images pushed successfully!"