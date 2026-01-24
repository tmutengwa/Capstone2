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
The model will be deployed as a web service where the business can input customer details (Age, Gender, City) and Product ID to get an estimated purchase value. Predictions for the test data (test.csv) are used to generate submissions in "SampleSubmission.csv" format.

### Why this Problem Matters
Accurate purchase prediction allows retailers to maximize customer lifetime value (CLV) and operational efficiency. Misjudging demand leads to overstocking (waste) or understocking (lost revenue). Root Mean Squared Error (RMSE) is used as the evaluation metric to penalize large errors significantly.

## 2. AutoEDA Platform

The project transforms `eda.py` into a powerful, intelligent Exploratory Data Analysis (EDA) platform with advanced features including automated statistical test selection, non-linear relationship detection, and AI-powered insights.

### AI Consultant (LLM-Powered)
- **Natural Language Insights**: Explains statistical findings in plain English using **Gemini 2.5 Flash**.
- **Business Context**: Connects statistics to actionable business decisions.
- **Interactive Q&A**: Ask "Godobori" questions about your data.
- **Transformation Strategy**: AI-generated feature engineering plans grounded in mathematical discovery.

### Interactive Dash UI
- **Real-Time Analysis**: No massive HTML files - renders on-demand.
- **Multi-Tab Interface**: Overview, Visual Analysis, AI Insights.
- **Smart Sampling**: Automatic sampling for large datasets (>1M rows) to ensure responsiveness.
- **Beautiful Visualizations**: Plotly-powered interactive charts (Sequential 1-7 analysis).

### Relationships & Statistical Testing
- **Bivariate Analysis**: Automatic analysis between any two variables.
- **Automatic Test Selection**:
  - **Categorical × Categorical**: Chi-squared test.
  - **Categorical × Numerical**: ANOVA or Kruskal-Wallis.
  - **Numerical × Numerical**: Pearson + Spearman + Mutual Information.
- **The Spotlight**: Flags non-linear relationships that traditional stats might miss.

### Self-Correcting ANOVA
Before running ANOVA, the platform checks:
1.  **Normality** (Shapiro-Wilk test).
2.  **Homogeneity of variance** (Levene's test).

If assumptions fail → Automatically switches to **Kruskal-Wallis** (non-parametric).

## 3. Evaluation Metric

**Metric:** Root Mean Squared Error (RMSE)

We use RMSE because it penalizes large errors more than Mean Absolute Error (MAE). In sales forecasting, significantly missing a high-value transaction is more costly than missing a small one.

## 4. Data Preparation & EDA

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

## 5. Modeling

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

## 6. Refactoring & Logic Analysis

We have significantly upgraded the core logic to improve accuracy, cost-efficiency, and grounding.

| Feature | Original Logic | Refactored Logic | Reason for Change |
| :--- | :--- | :--- | :--- |
| **LLM Provider** | OpenAI / GPT-4o | **Gemini 2.5 Flash** | Massive cost reduction (~25x) and native multimodal support. |
| **Instruction Method** | Static Markdown prompt strings | **Dynamic System Instruction Factory** | System instructions have higher "cognitive weight," forcing the model to stick to the Lattice Check. |
| **Image Handling** | Manual Base64 string encoding | **Direct PIL.Image Interleaving** | Gemini processes raw pixels more accurately than text-described images. |
| **Data Discovery** | Generic prompt | **Deterministic Function Injection** | Passing Lattice Check results as facts prevents hallucination; Python functions provide undeniable math facts. |
| **Sequencing** | Random/Unordered | **Sequential Numbering (1-7)** | Forces 1:1 analysis of every visual evidence piece. |

### Logic State & The "Ground Truth"
The **System Instruction Factory** is the most critical addition. It allows the model to "learn" the dataset's structural hierarchy at the firmware level of the conversation.

**Key Grounding Prompts:**
- **Data Health Assessment**: Checks "Price Signature" to decide between Labelling vs. Dropping.
- **Visual Analysis**: Sequentially analyzes plots anchored by the Lattice Root.
- **Ask Godobori**: Answers strict Q&A based on the statistical context provided.

## 7. Web Service (FastAPI)

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

## 8. Dockerization

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

## 9. Deployment Strategy

Deploying both the **API** (FastAPI) and **Dashboard** (Dash) to AWS Lambda within a single repo requires a specific architectural shift using the **Lambda Web Adapter**.

### The Challenge
A Lambda function only has one URL/Port. You cannot run both the API and Dashboard on different ports inside the same function.

### The Solution: "One Repo, Two Tags"
We create two separate Lambda functions that point to the same ECR repository but use different image tags.

1.  **Lambda 1 (API)**: Points to `black-friday-sales-project:api` (or `latest`).
2.  **Lambda 2 (EDA)**: Points to `black-friday-sales-project:eda` (or `latest` with CMD override).

### Architecture Overview
- **Source**: Single GitHub Repository.
- **Artifact**: Single ECR Repository (`black-friday-sales-project`).
- **Runtime**: AWS Lambda + Lambda Web Adapter.

### Deployment Guide

#### 1. Local Build & Push
Run the deployment script from your project root:
```bash
./push_image.sh
```
*   **Result**: This builds the image locally (amd64), pushes tagged versions to ECR.

#### 2. Lambda Configuration (Manual Check)
**API Lambda (`black-friday-api`)**
- **Handler/CMD**: Default (`serve:app`).
- **Web Adapter**: Translates HTTP to JSON events.

**EDA Lambda (`black-friday-eda`)**
- **Image Configuration**: Set **CMD override** to `python`, `eda.py`.
- **Note**: `eda.py` is configured to run on `0.0.0.0` with `debug=False` for Lambda compatibility.

**Common Settings**
- **Memory**: Set to **3008 MB** for optimal ML performance.
- **Timeout**: ~30 seconds.

## License
MIT License
