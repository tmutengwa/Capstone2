#!/bin/bash

# --- CONFIGURATION (Literal Values) ---
ACCOUNT_ID="955423509456"
REGION="us-east-1"
REPO_NAME="black-friday-sales-project"
ECR_URL="955423509456.dkr.ecr.us-east-1.amazonaws.com"

# Exact names based on your AWS environment
API_ROLE_NAME="sales-api-role"
EDA_ROLE_NAME="sales-eda-role"
API_FUNC="salesAPI"
EDA_FUNC="salesEDA"

echo "🚀 Starting Deployment Fix..."

# ====================================================
# STEP 1: Create Trust Policy
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

# ====================================================
# STEP 2: Manage Roles (Robust Logic)
# ====================================================

# Function to handle Role Creation/Checking
ensure_role() {
    local ROLE_NAME=$1
    echo "👤 Configuring IAM Role: ${ROLE_NAME}..."

    # Check if role exists
    aws iam get-role --role-name "${ROLE_NAME}" >/dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "   🆕 Role not found. Creating..."
        aws iam create-role --role-name "${ROLE_NAME}" --assume-role-policy-document file://trust-policy.json
    else
        echo "   ✅ Role exists. Updating trust policy..."
        aws iam update-assume-role-policy --role-name "${ROLE_NAME}" --policy-document file://trust-policy.json
    fi
}

# Run for both roles
ensure_role "${API_ROLE_NAME}"
ensure_role "${EDA_ROLE_NAME}"

# ====================================================
# STEP 3: Attach Permissions
# ====================================================
echo "🔗 Attaching Permissions..."

# Loop through both roles to attach policies
for ROLE in "${API_ROLE_NAME}" "${EDA_ROLE_NAME}"; do
    echo "   ...processing ${ROLE}"
    # Basic Execution (Logs)
    aws iam attach-role-policy --role-name "${ROLE}" --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
    # ECR Read Access (Pull Images)
    aws iam attach-role-policy --role-name "${ROLE}" --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
done

echo "   ✅ Permissions attached."
echo "⏳ Waiting 10 seconds for permissions to propagate..."
sleep 10

# ====================================================
# STEP 4: Create/Update Lambda Functions
# ====================================================
API_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${API_ROLE_NAME}"
EDA_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${EDA_ROLE_NAME}"

# --- API FUNCTION ---
echo "🛠️ Configuring API Lambda (${API_FUNC})..."
aws lambda get-function --function-name "${API_FUNC}" >/dev/null 2>&1
if [ $? -ne 0 ]; then
    # Create if missing
    aws lambda create-function \
        --function-name "${API_FUNC}" \
        --package-type Image \
        --code ImageUri="${ECR_URL}/${REPO_NAME}:latest" \
        --role "${API_ROLE_ARN}" \
        --memory-size 3008 \
        --timeout 90 \
        --region "${REGION}"
    echo "   ✅ API Function Created."
else
    # Update if exists
    echo "   ⚠️ Function exists. Updating Config..."
    aws lambda update-function-configuration --function-name "${API_FUNC}" --role "${API_ROLE_ARN}" --memory-size 3008 --timeout 60 >/dev/null
fi

# --- EDA FUNCTION ---
echo "🛠️ Configuring EDA Lambda (${EDA_FUNC})..."
aws lambda get-function --function-name "${EDA_FUNC}" >/dev/null 2>&1
if [ $? -ne 0 ]; then
    # Create if missing
    aws lambda create-function \
        --function-name "${EDA_FUNC}" \
        --package-type Image \
        --code ImageUri="${ECR_URL}/${REPO_NAME}:latest" \
        --role "${EDA_ROLE_ARN}" \
        --memory-size 3008 \
        --timeout 90 \
        --region "${REGION}" \
        --image-config '{"Command": ["python", "eda.py"]}' \
        --environment 'Variables={PORT=8050}'
    echo "   ✅ EDA Function Created."
else
    # Update if exists
    echo "   ⚠️ Function exists. Updating Config..."
    aws lambda update-function-configuration --function-name "${EDA_FUNC}" --role "${EDA_ROLE_ARN}" --memory-size 3008 --timeout 90 --image-config '{"Command": ["python", "eda.py"]}' --environment 'Variables={PORT=8050}' >/dev/null
fi

# ====================================================
# STEP 5: Public Access (URLs)
# ====================================================
echo "🌐 Configuring Public URLs..."

for FUNC in "${API_FUNC}" "${EDA_FUNC}"; do
    # Create URL config (ignore error if exists)
    aws lambda create-function-url-config --function-name "${FUNC}" --auth-type NONE >/dev/null 2>&1
    # Add public permission (ignore error if exists)
    aws lambda add-permission --function-name "${FUNC}" --action lambda:InvokeFunctionUrl --statement-id PublicAccess --principal "*" --function-url-auth-type NONE >/dev/null 2>&1
done

echo "🎉 Setup Complete!"
echo "👉 API Endpoint:   $(aws lambda get-function-url-config --function-name "${API_FUNC}" --query 'FunctionUrl' --output text)"
echo "👉 Dash Dashboard: $(aws lambda get-function-url-config --function-name "${EDA_FUNC}" --query 'FunctionUrl' --output text)"s