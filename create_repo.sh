
echo "🚀 Starting ECR Image Push:"

# # 1. Create ECR repository if it doesn't exist
# aws ecr describe-repositories --repository-names "${REPO_NAME}" --region ${REGION} >/dev/null 2>&1
# if [ $? -ne 0 ]; then
#     echo "🆕 Creating ECR repository: ${REPO_NAME}"
#     aws ecr create-repository --repository-name "${REPO_NAME}" --region ${REGION}
# fi
# 2. Authenticate Docker
#Retrieve an authentication token and authenticate your Docker client to your registry. Use the AWS CLI:
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 955423509456.dkr.ecr.us-east-1.amazonaws.com
#Note: If you receive an error using the AWS CLI, make sure that you have the latest version of the AWS CLI and Docker installed.

# 3. Build your Docker image
#Build your Docker image using the following command. For information on building a Docker file from scratch see the instructions here . You can skip this step if your image is already built:
docker build --platform linux/amd64 --provenance=false --output type=docker -t 955423509456.dkr.ecr.us-east-1.amazonaws.com/black-friday-sales-project:latest .
# 4. Tag your image
#After the build completes, tag your image so you can push the image to this repository:
docker tag black-friday-sales-project:latest 955423509456.dkr.ecr.us-east-1.amazonaws.com/black-friday-sales-project:latest
# 5. Push your image to AWS ECR
#Run the following command to push this image to your newly created AWS repository:
docker push 955423509456.dkr.ecr.us-east-1.amazonaws.com/black-friday-sales-project:latest