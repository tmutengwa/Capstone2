# Black Friday Purchase Prediction & AI-Driven EDA Platform

A comprehensive machine learning project combining a robust prediction API with an intelligent, AI-powered Exploratory Data Analysis (EDA) dashboard.

## Features

### 🚀 Prediction Service (API)
- **FastAPI Backend**: High-performance REST API.
- **Production ML Pipeline**: Scikit-learn & XGBoost with `DictVectorizer`.
- **Artifact Management**: Versioned model storage.
- **Serverless Ready**: AWS Lambda compatible.

### 📊 AI-Driven EDA Platform (Dashboard)
- **Interactive UI**: Plotly Dash-based interface for real-time analysis.
- **Smart Profiling**: Automated statistical tests (Chi-squared, ANOVA, Pearson/Spearman).
- **The Spotlight™**: Detection of non-linear relationships using Mutual Information.
- **AI Consultant**: LLM-powered insights and Q&A (via OpenAI).
- **DuckDB Engine**: Fast in-memory SQL processing for data transformations.
- **Pydantic Configuration**: Robust settings management using `pydantic-settings`.

## Project Structure

```
Capstone2/
├── data/                          # Data directory
├── models/                        # Trained artifacts
├── logs/                          # Application logs
├── eda.py                         # EDA Dashboard Application (Entry point)
├── orchestrator.py                # Data Management (DuckDB)
├── profiler.py                    # Statistical Profiling Engine
├── llm_consultant.py              # AI Integration (OpenAI)
├── config.py                      # Pydantic Settings Configuration
├── logger.py                      # Logging Configuration
├── serve.py                       # Prediction API
├── train.py                       # Training Script
├── Dockerfile                     # Unified Container Image
├── docker-compose.yaml            # Multi-service Orchestration
├── deployment.yaml                # Kubernetes Manifests
└── requirements.txt               # Dependencies
```

## Installation

1. **Clone & Setup Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configuration**
   Create a `.env` file and configure your settings:
   ```bash
   # LLM Settings
   ENABLE_LLM=True
   OPENAI_API_KEY=your_key_here
   LLM_MODEL=gpt-4o
   SIGNIFICANCE_LEVEL=0.05
   CATEGORICAL_THRESHOLD=15
   DUCKDB_MEMORY_LIMIT=4GB
   ```

3. **Data Setup**
   ```bash
   python3 setup_data.py
   ```

## Configuration Reference

The application uses `pydantic-settings` to manage configuration via environment variables or a `.env` file.

| Variable | Description | Default |
|----------|-------------|---------|
| `MAX_FILE_SIZE_MB` | Max upload size | 500 |
| `LARGE_DATASET_THRESHOLD` | Row count for sampling | 1,000,000 |
| `SIGNIFICANCE_LEVEL` | Statistical p-value threshold | 0.05 |
| `DUCKDB_MEMORY_LIMIT` | Memory limit for DuckDB | 4GB |
| `ENABLE_LLM` | Enable AI Consultant features | False |
| `OPENAI_API_KEY` | OpenAI API Key | None |
| `LLM_MODEL` | OpenAI Model to use | gpt-4 |

## Usage

### 1. AI-Driven EDA Dashboard
Launch the interactive platform:
```bash
python3 eda.py
```
Access at: `http://localhost:8050`

### 2. Model Training
Train the prediction model:
```bash
python3 train.py
```

### 3. Prediction API
Start the REST API:
```bash
python3 serve.py
```
Access at: `http://localhost:8080`

## Docker Deployment

### Using Docker Compose (Recommended)
Run both the API and Dashboard simultaneously:
```bash
docker-compose up --build
```
- **API**: `http://localhost:8080`
- **Dashboard**: `http://localhost:8050`

### Using Docker Manually
Build the unified image:
```bash
docker build -t purchase-prediction .
```

**Run API:**
```bash
docker run -p 8080:8080 purchase-prediction
```

**Run Dashboard:**
```bash
docker run -p 8050:8050 --env-file .env purchase-prediction python eda.py
```

## Cloud Deployment

### AWS Lambda (API Only)
The Docker image includes the AWS Lambda Adapter.
1. Push image to ECR.
2. Create Lambda function from the image.
3. Set `CMD` to `serve:app` (default) or configure via environment variables.

### Kubernetes
Deploy both services to a cluster:
```bash
kubectl apply -f deployment.yaml
```
*Note: Ensure you update the image URI and create the necessary secrets for OpenAI keys.*

## License
MIT License