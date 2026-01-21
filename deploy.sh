#!/bin/bash

# --- CONFIGURATION ---
ACCOUNT_ID="955423509456"
REGION="us-east-1"
REPO_NAME="black-friday-sales-project"
ECR_URL="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
API_LAMBDA_NAME="black-friday-api"
EDA_LAMBDA_NAME="black-friday-eda"

echo "🚀 Starting Deployment for Plotly Dash & FastAPI: ${REPO_NAME}"
# create the ECR repository if it doesn't exist
aws ecr describe-repositories --repository-names "${REPO_NAME}" --region ${REGION} >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "🆕 ECR repository not found. Creating repository: ${REPO_NAME}"
    aws ecr create-repository --repository-name "${REPO_NAME}" --region ${REGION}
else
    echo "✅ ECR repository found: ${REPO_NAME}"
fi
# 1. Authenticate Docker with ECR
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ECR_URL}

# 2. Build Single Image (Intel Mac native amd64)
echo "📦 Building Base Image..."
docker build --platform linux/amd64 -t ${REPO_NAME}:latest .

# 3. Tag and Push API Version
echo "📤 Pushing API image..."
docker tag ${REPO_NAME}:latest ${ECR_URL}/${REPO_NAME}:api-v1
docker push ${ECR_URL}/${REPO_NAME}:api-v1

# 4. Tag and Push EDA (Dash) Version
echo "📤 Pushing EDA (Dash) image..."
docker tag ${REPO_NAME}:latest ${ECR_URL}/${REPO_NAME}:eda-v1
docker push ${ECR_URL}/${REPO_NAME}:eda-v1

# 5. Trigger Lambda Updates
echo "🔄 Updating Lambda Functions..."

aws lambda update-function-code \
    --function-name ${API_LAMBDA_NAME} \
    --image-uri ${ECR_URL}/${REPO_NAME}:api-v1

aws lambda update-function-code \
    --function-name ${EDA_LAMBDA_NAME} \
    --image-uri ${ECR_URL}/${REPO_NAME}:eda-v1

echo "✅ Deployment and Updates Complete!"