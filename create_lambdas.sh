#!/bin/bash

# --- CONFIGURATION ---
ACCOUNT_ID="955423509456"
REGION="us-east-1"
REPO_NAME="black-friday-sales-project"
ECR_URL="955423509456.dkr.ecr.us-east-1.amazonaws.com"
ROLE_ARN="arn:aws:iam::955423509456:role/black-friday-lambda-role"

echo "🛠️ Creating Lambda Functions..."

# 1. Create API Lambda
aws lambda create-function \
    --function-name salesAPI \
    --package-type Image \
    --code ImageUri=${ECR_URL}/${REPO_NAME}:api-v1 \
    --role ${ROLE_ARN} \
    --memory-size 3008 \
    --timeout 90 \
    --region ${REGION}

# 2. Create EDA (Dash) Lambda
# Note: This includes the Command Override for your Dash entry point
aws lambda create-function \
    --function-name salesEDA \
    --package-type Image \
    --code ImageUri=${ECR_URL}/${REPO_NAME}:eda-v1 \
    --role ${ROLE_ARN} \
    --memory-size 3008 \
    --timeout 90 \
    --region ${REGION} \
    --image-config '{"Command": ["python", "eda.py"]}'

echo "✅ Lambda creation complete!"