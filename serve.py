"""
Black Friday Purchase Prediction - Prediction Service

FastAPI web service for serving purchase amount predictions.
Uses a trained model artifact and DictVectorizer for feature transformation.
Includes Async Job processing for large files.

Usage:
    uvicorn serve:app --host 0.0.0.0 --port 8080 --reload

Endpoints:
    - GET  /: Health check
    - POST /predict: Single prediction
    - POST /predict_batch: Batch predictions (JSON)
    - POST /predict_file: Start batch prediction job (Returns job_id)
    - GET  /jobs/{job_id}: Check job status and get results
"""

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
import logging
import os
import io
import uuid
import warnings
import json
from datetime import datetime
from feature_engineering import preprocess_single_record

# Suppress specific sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")

# Detect if running in AWS Lambda
# AWS_LAMBDA_FUNCTION_NAME is a standard variable set by the Lambda runtime
IS_LAMBDA = os.environ.get("AWS_LAMBDA_FUNCTION_NAME") is not None

# Set the jobs directory accordingly
if IS_LAMBDA:
    # Use the writable /tmp directory in Lambda
    JOBS_DIR = "/tmp/jobs"
    LOGS_DIR = "/tmp/logs"
else:
    # Use the local directory for your MacBook/Docker
    JOBS_DIR = "jobs"
    LOGS_DIR = "logs"

# Create the directories safely
os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    filename=f'{LOGS_DIR}/serve.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# Initialize FastAPI app
app = FastAPI(
    title="Black Friday Purchase Prediction API",
    description="Predict purchase amount for retail customers using DictVectorizer and tuned model",
    version="1.1.0"
)

# Load model artifacts on startup
model = None
dv = None

# In-memory Job Store (For demo/simplicity. Production should use Redis/DB)
# Structure: { job_id: { status: 'pending'|'processing'|'completed'|'failed', submitted_at: datetime, result_file: str, error: str } }
jobs = {}

@app.on_event("startup")
async def load_model():
    """
    Load trained artifacts when API starts
    """
    global model, dv
    logging.info("Loading model artifacts...")
    try:
        model = joblib.load('models/best_model.pkl')
        dv = joblib.load('models/preprocessor.b')
        logging.info("✓ Model and Vectorizer loaded successfully")
    except Exception as e:
        logging.error(f"Error loading artifacts: {str(e)}")
        pass

# --- Pydantic Models ---

class PredictionRequest(BaseModel):
    Gender: str
    Age: str
    Occupation: int
    City_Category: str
    Stay_In_Current_City_Years: str
    Marital_Status: int
    Product_Category_1: int
    Product_Category_2: Optional[float] = None
    Product_Category_3: Optional[float] = None

class BatchPredictionRequest(BaseModel):
    predictions: List[PredictionRequest]

class PredictionResponse(BaseModel):
    purchase_prediction: float
    Gender: str
    Age: str
    Product_Category_1: int

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int

class JobResponse(BaseModel):
    job_id: str
    status: str
    submitted_at: datetime
    message: str = "Use /jobs/{job_id} to check status"

# --- Background Processing Logic ---

def process_file_background(job_id: str, file_content: bytes, filename: str):
    """
    Background task to process large files.
    """
    logging.info(f"Starting background processing for job {job_id}")
    jobs[job_id]['status'] = 'processing'
    
    try:
        file_ext = filename.split('.')[-1].lower()
        if file_ext == 'csv':
            df = pd.read_csv(io.BytesIO(file_content))
        elif file_ext in ['xls', 'xlsx']:
            df = pd.read_excel(io.BytesIO(file_content))
        else:
            raise ValueError("Invalid file type")

        data = df.to_dict(orient='records')
        processed_list = [preprocess_single_record(item) for item in data]
        
        # Transform & Predict
        X = dv.transform(processed_list)
        predictions = model.predict(X)
        predictions = np.maximum(predictions, 0)
        
        # Create Result DataFrame
        result_df = df.copy()
        result_df['Purchase_Prediction'] = np.round(predictions, 2)
        
        # Save result
        result_filename = f"{JOBS_DIR}/{job_id}_result.csv"
        result_df.to_csv(result_filename, index=False)
        
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['result_file'] = result_filename
        logging.info(f"Job {job_id} completed successfully.")
        
    except Exception as e:
        logging.error(f"Job {job_id} failed: {e}")
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)

# --- API Endpoints ---

@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "Black Friday Purchase Prediction API",
        "version": "1.1.0"
    }

@app.get("/health")
async def health_check():
    if model is None or dv is None:
        raise HTTPException(status_code=503, detail="Model or Vectorizer not loaded")
    return {"status": "healthy", "model": "loaded"}

@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: PredictionRequest):
    if not model or not dv:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        data_dict = input_data.dict()
        processed_data = preprocess_single_record(data_dict)
        X = dv.transform([processed_data])
        prediction = model.predict(X)[0]
        result = float(prediction)
        return PredictionResponse(
            purchase_prediction=round(result, 2),
            Gender=input_data.Gender,
            Age=input_data.Age,
            Product_Category_1=input_data.Product_Category_1
        )
    except Exception as e:
        logging.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    if not model or not dv:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        results = []
        processed_list = [preprocess_single_record(item.dict()) for item in request.predictions]
        X = dv.transform(processed_list)
        predictions = model.predict(X)
        predictions = np.maximum(predictions, 0)
        
        for i, pred in enumerate(predictions):
            item = request.predictions[i]
            results.append(PredictionResponse(
                purchase_prediction=round(float(pred), 2),
                Gender=item.Gender,
                Age=item.Age,
                Product_Category_1=item.Product_Category_1
            ))
        return BatchPredictionResponse(predictions=results, count=len(results))
    except Exception as e:
        logging.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_file", response_model=JobResponse)
async def predict_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Start a background job to process a CSV/Excel file.
    Returns a Job ID immediately.
    """
    logging.info(f"Received file upload: {file.filename}")
    if not model or not dv:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Generate Job ID
    job_id = str(uuid.uuid4())
    
    # Read content into memory (for this scale, 230k rows ~20-50MB is fine in memory)
    # If files were HUGE (GBs), we would stream to disk first.
    try:
        file_content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    # Register Job
    jobs[job_id] = {
        "status": "pending",
        "submitted_at": datetime.now(),
        "filename": file.filename
    }

    # Start Background Task
    background_tasks.add_task(process_file_background, job_id, file_content, file.filename)

    return JobResponse(
        job_id=job_id,
        status="pending",
        submitted_at=jobs[job_id]["submitted_at"]
    )

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    Check status of a job. If completed, returns download link or result preview.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    response = {
        "job_id": job_id,
        "status": job["status"],
        "submitted_at": job["submitted_at"],
    }

    if job["status"] == "completed":
        response["download_url"] = f"/jobs/{job_id}/download"
    elif job["status"] == "failed":
        response["error"] = job.get("error")

    return response

@app.get("/jobs/{job_id}/download")
async def download_job_result(job_id: str):
    """
    Download the result file for a completed job.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    file_path = job.get("result_file")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=500, detail="Result file missing")

    return FileResponse(file_path, filename=f"predictions_{job_id}.csv", media_type='text/csv')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)