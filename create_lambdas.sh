#!/bin/bash

# --- CONFIGURATION (Literal Values) ---
ACCOUNT_ID="955423509456"
REGION="us-east-1"
REPO_NAME="black-friday-sales-project"
ECR_URL="955423509456.dkr.ecr.us-east-1.amazonaws.com"
API_ROLE_NAME="sales-api-role"
EDA_ROLE_NAME="sales-eda-role"
API_FUNC="salesAPI"
EDA_FUNC="salesEDA"

echo "🚀 Starting Full Stack Initialization..."

# ====================================================
# STEP 1 & 2: Create Trust Policy & IAM Role
# ====================================================
echo "🔐 Generating trust-policy.json..."
cat <<EOF > trust-policy.json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

echo "👤 Creating IAM Role: ${API_ROLE_NAME}..."
# Attempt to create role; suppress error if it already exists
aws iam create-role \
    --role-name ${ROLE_NAME} \
    --assume-role-policy-document file://trust-policy.json >/dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "   ✅ Role created."
else
    echo "   ⚠️ Role already exists (skipping creation)."
fi

echo "👤 Creating IAM Role: ${EDA_ROLE_NAME}..."
# Attempt to create role; suppress error if it already exists
aws iam create-role \
    --role-name ${EDA_ROLE_NAME} \
    --assume-role-policy-document file://trust-policy.json >/dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "   ✅ Role created."
else
    echo "   ⚠️ Role already exists (skipping creation)."
fi

# ====================================================
# STEP 3 & 4: Attach Permissions
# ====================================================
echo "🔗 Attaching Permissions..."

# 1. Basic Execution (CloudWatch Logs)
aws iam attach-role-policy \
    --role-name ${API_ROLE_NAME} \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam attach-role-policy \
    --role-name ${EDA_ROLE_NAME} \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
    echo "   ✅ Basic Execution attached."
 

# 2. ECR Read Access (Pull Container Images)
aws iam attach-role-policy \
    --role-name ${API_ROLE_NAME} \
    --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly

aws iam attach-role-policy \
    --role-name ${EDA_ROLE_NAME} \
    --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly

echo "   ✅ Permissions attached."

# Wait for IAM propagation (Crucial Step!)
echo "⏳ Waiting 30 seconds for permissions to propagate..."
sleep 30

# ====================================================
# STEP 5: Create Lambda Functions
# ====================================================
API_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${API_ROLE_NAME}"
EDA_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${EDA_ROLE_NAME}"

echo "🛠️ Creating API Lambda (${API_FUNC})..."
aws lambda create-function \
    --function-name ${API_FUNC} \
    --package-type Image \
    --code ImageUri=${ECR_URL}/${REPO_NAME}:api-v1 \
    --role ${API_ROLE_ARN} \
    --memory-size 3008 \
    --timeout 90 \
    --region ${REGION} >/dev/null 2>&1

if [ $? -eq 0 ]; then echo "   ✅ API created."; else echo "   ⚠️ API exists or failed."; fi

echo "🛠️ Creating EDA Lambda (${EDA_FUNC})..."
# Note: Includes Command Override for Dash
aws lambda create-function \
    --function-name ${EDA_FUNC} \
    --package-type Image \
    --code ImageUri=${ECR_URL}/${REPO_NAME}:eda-v1 \
    --role ${EDA_ROLE_ARN} \
    --memory-size 3008 \
    --timeout 90 \
    --region ${REGION} \
    --image-config '{"Command": ["python", "eda.py"]}' \
    --environment 'Variables={PORT=8050}' >/dev/null 2>&1

if [ $? -eq 0 ]; then echo "   ✅ EDA created."; else echo "   ⚠️ EDA exists or failed."; fi

# ====================================================
# STEP 6: Create Function URLs (Public Access)
# ====================================================
echo "🌐 Generating Public URLs..."

# --- API URL ---
aws lambda create-function-url-config \
    --function-name ${API_FUNC} \
    --auth-type NONE >/dev/null 2>&1

aws lambda add-permission \
    --function-name ${API_FUNC} \
    --action lambda:InvokeFunctionUrl \
    --statement-id FunctionURLAllowPublicAccess \
    --principal "*" \
    --function-url-auth-type NONE >/dev/null 2>&1

# --- EDA URL ---
aws lambda create-function-url-config \
    --function-name ${EDA_FUNC} \
    --auth-type NONE >/dev/null 2>&1

aws lambda add-permission \
    --function-name ${EDA_FUNC} \
    --action lambda:InvokeFunctionUrl \
    --statement-id FunctionURLAllowPublicAccess \
    --principal "*" \
    --function-url-auth-type NONE >/dev/null 2>&1

# ====================================================
# FINAL OUTPUT
# ====================================================
echo "🎉 Setup Complete! Access your services here:"
echo ""
echo "👉 API Endpoint:   $(aws lambda get-function-url-config --function-name ${API_FUNC} --query 'FunctionUrl' --output text)"
echo "👉 Dash Dashboard: $(aws lambda get-function-url-config --function-name ${EDA_FUNC} --query 'FunctionUrl' --output text)"
echo ""