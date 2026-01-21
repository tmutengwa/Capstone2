# Black Friday Purchase Prediction & AI-Driven EDA Platform

## 1. Problem Description

### Problem Statement
A retail company “ABC Private Limited” wants to understand the customer purchase behaviour (specifically, purchase amount) against various products of different categories. They have shared purchase summary of various customers for selected high volume products from last month.

The data set also contains customer demographics (age, gender, marital status, city_type, stay_in_current_city), product details (product_id and product category) and Total purchase_amount from last month.

Now, they want to build a model to predict the purchase amount of customer against various products which will help them to create personalized offer for customers against different products.

### Data Dictionary
| Variable | Definition |
|---|---|
| User_ID | User ID |
| Product_ID | Product ID |
| Gender | Sex of User |
| Age | Age in bins |
| Occupation | Occupation (Masked) |
| City_Category | Category of the City (A,B,C) |
| Stay_In_Current_City_Years | Number of years stay in current city |
| Marital_Status | Marital Status |
| Product_Category_1 | Product Category (Masked) |
| Product_Category_2 | Product may belong to other category also (Masked) |
| Product_Category_3 | Product may belong to other category also (Masked) |
| **Purchase** | **Purchase Amount (Target Variable)** |

### Who Benefits?
*   **Retailers**: To create personalized offers, optimize inventory, and forecast revenue.
*   **Marketing Teams**: To target specific demographics with high-value product categories.

### How the Model is Used
The model will be deployed as a web service where the business can input customer details (Age, Gender, City) and Product ID to get an estimated purchase value. Predictions for the test data (test.csv) are used to generate submissions in the "SampleSubmission.csv" format.

### Why this Problem Matters
Accurate purchase prediction allows retailers to maximize customer lifetime value (CLV) and operational efficiency. Misjudging demand leads to overstocking (waste) or understocking (lost revenue). Root Mean Squared Error (RMSE) is used as the evaluation metric to penalize large errors significantly.

## 2. Evaluation Metric

**Metric:** Root Mean Squared Error (RMSE)

We use RMSE because it penalizes large errors more than Mean Absolute Error (MAE). In sales forecasting, significantly missing a high-value transaction is more costly than missing a small one.

## 3. Data Preparation & EDA

**Data Source:** The dataset is sourced from Analytics Vidhya (Black Friday Sales Prediction).

### Retrieval & Setup
Data retrieval is automated via the `setup_data.py` script, which extracts the dataset into a structured `data/` directory.

### EDA Findings
*   **Target Distribution**: The purchase amount is Right-Skewed. Most transactions are mid-range, with a "long tail" of high spenders.
*   **Missing Values**: `Product_Category_2` (~31% missing) and `Product_Category_3` (~69% missing). These are treated as "Structural Nulls" (indicating a basic product) and imputed with `0`.
*   **Correlations**: `Product_Category_1` is the strongest predictor. Age and Gender have non-linear relationships with spending.
*   **Outliers**: Z-score analysis revealed no extreme outliers requiring removal (Z > 3).

### Reproducibility
All preprocessing steps (Imputation, Type Casting, Feature Engineering) are encapsulated in `feature_engineering.py` to ensure the exact same logic is applied during Training, Batch Prediction, and API Inference.

## 4. Modeling

We trained and compared three different regression models to select the best performer.

### Models Trained
1.  **Decision Tree Regressor**: Baseline model.
2.  **Random Forest Regressor**: Ensemble method to capture non-linearities.
3.  **XGBoost / LightGBM**: Gradient boosting machines for high performance.

### Model Selection & Tuning
*   **Cross-Validation**: Used 3-fold cross-validation during tuning.
*   **Hyperparameter Tuning**: Performed using `RandomizedSearchCV`.
*   **Selection**: **LightGBM** was selected as the best model based on the lowest RMSE on the validation set.

### Scores Summary
| Model | RMSE (Validation) |
|---|---|
| Decision Tree | ~3300 |
| Random Forest | ~2950 |
| **LightGBM** | **~2894** |

## 6. Web Service

The project implements a **FastAPI** application (`serve.py`) exposing the following endpoints:

### Endpoints
*   `POST /predict`: Accepts a single JSON input and returns a prediction.
*   `GET /health`: Simple 200 OK check for container health.
*   `POST /predict_batch`: Accepts a list of inputs.
*   `POST /predict_file`: Accepts CSV/Excel file uploads.

### Example Request
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{
           "Gender": "M", 
           "Age": "26-35", 
           "Occupation": 16, 
           "City_Category": "A", 
           "Stay_In_Current_City_Years": "2", 
           "Marital_Status": 0, 
           "Product_Category_1": 5
         }' \
     http://localhost:8080/predict
```

## 7. Dockerization

✅ **Dockerfile** provided for reproducible deployment.

```dockerfile
# Base Image
FROM python:3.12-slim

# Install system dependencies (including libgomp1 for LightGBM)
RUN apt-get update && apt-get install -y curl libgomp1 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY . .

# Expose port
EXPOSE 8080

# Command to run the app
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Build & Run Instructions

```bash
# Build the image
docker build -t black-friday-sales .

# Run the container
docker run -it -p 8080:8080 black-friday-sales
```

## 8. Cloud Deployment

### AWS Lambda Deployment (Detailed Guide)

We use the **AWS Lambda Web Adapter** to deploy the FastAPI container directly to Lambda.

**Prerequisites:**
- AWS CLI configured.
- Docker installed.

**Step 1: Create ECR Repository**
```bash
aws ecr create-repository --repository-name black-friday-api --region us-east-1
```

**Step 2: Build and Push Image**
```bash
# Authenticate
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <YOUR_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# Build
docker build -t black-friday-api .

# Tag & Push
docker tag black-friday-api:latest <YOUR_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/black-friday-api:latest
docker push <YOUR_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/black-friday-api:latest
```

**Step 3: Create Function**
1.  Go to **AWS Lambda Console**.
2.  Click **Create function** -> **Container image**.
3.  Name: `BlackFridayPrediction`.
4.  Image URI: Select the image from ECR.

**Step 4: Configuration**
1.  **Memory**: Increase to `1024 MB`.
2.  **Timeout**: Set to `30 seconds`.
3.  **Function URL**: Enable Function URL (Auth Type: `NONE` for public access).

**Step 5: Access**
Your API is now live at the provided Lambda Function URL. Append `/docs` to see the Swagger UI.

## License
MIT License