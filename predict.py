"""
Black Friday Purchase Prediction - Prediction Service

FastAPI web service for serving purchase amount predictions.
Uses a trained model artifact and DictVectorizer for feature transformation.

Usage:
    uvicorn predict:app --host 0.0.0.0 --port 8080 --reload

Endpoints:
    - GET  /: Health check
    - POST /predict: Single prediction
    - POST /predict_batch: Batch predictions (JSON)
    - POST /predict_csv: Predictions from raw data (list of dicts)
"""

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
from feature_engineering import preprocess_single_record

# Initialize FastAPI app
app = FastAPI(
    title="Black Friday Purchase Prediction API",
    description="Predict purchase amount for retail customers using DictVectorizer and tuned model",
    version="1.0.0"
)

# Load model artifacts on startup
model = None
dv = None

@app.on_event("startup")
async def load_model():
    """
    Load trained artifacts when API starts
    """
    global model, dv

    try:
        model = joblib.load('models/best_model.pkl')
        dv = joblib.load('models/preprocessor.b')
        print("✓ Model and Vectorizer loaded successfully")
    except Exception as e:
        print(f"Error loading artifacts: {str(e)}")
        # Note: Startup might fail if files don't exist, which is expected before training
        pass

# Request models
class PredictionRequest(BaseModel):
    """
    Single prediction request schema
    """
    Gender: str = Field(..., description="Sex of User (F or M)")
    Age: str = Field(..., description="Age in bins (e.g., 0-17, 26-35, 55+)")
    Occupation: int = Field(..., description="Occupation (Masked ID)")
    City_Category: str = Field(..., description="Category of the City (A, B, C)")
    Stay_In_Current_City_Years: str = Field(..., description="Number of years stay in current city ('4+' allowed)")
    Marital_Status: int = Field(..., description="Marital Status (0 or 1)")
    Product_Category_1: int = Field(..., description="Product Category (Masked ID)")
    Product_Category_2: Optional[float] = Field(None, description="Secondary Product Category (Optional)")
    Product_Category_3: Optional[float] = Field(None, description="Tertiary Product Category (Optional)")

    class Config:
        schema_extra = {
            "example": {
                "Gender": "M",
                "Age": "26-35",
                "Occupation": 16,
                "City_Category": "A",
                "Stay_In_Current_City_Years": "2",
                "Marital_Status": 0,
                "Product_Category_1": 5,
                "Product_Category_2": 8.0,
                "Product_Category_3": None
            }
        }

class BatchPredictionRequest(BaseModel):
    """
    Batch prediction request schema
    """
    predictions: List[PredictionRequest]

class PredictionResponse(BaseModel):
    """
    Prediction response schema
    """
    purchase_prediction: float = Field(..., description="Predicted purchase amount")
    Gender: str
    Age: str
    Product_Category_1: int

class BatchPredictionResponse(BaseModel):
    """
    Batch prediction response schema
    """
    predictions: List[PredictionResponse]
    count: int

@app.get("/")
async def root():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "service": "Black Friday Purchase Prediction API",
        "version": "1.0.0",
        "model_loaded": model is not None and dv is not None
    }

@app.get("/health")
async def health_check():
    """
    Detailed health check
    """
    if model is None or dv is None:
        raise HTTPException(status_code=503, detail="Model or Vectorizer not loaded")

    return {
        "status": "healthy",
        "model": "loaded",
        "vectorizer": "loaded",
        "feature_count": len(dv.feature_names_) if dv else 0
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a single prediction
    """
    if model is None or dv is None:
        raise HTTPException(status_code=503, detail="Artifacts not loaded")

    try:
        # Convert request to dictionary
        data = request.dict()

        # Apply feature engineering
        processed_data = preprocess_single_record(data)

        # Transform and Predict
        X = dv.transform([processed_data])
        prediction = model.predict(X)[0]
        prediction = max(0, float(prediction))

        return PredictionResponse(
            purchase_prediction=round(prediction, 2),
            Gender=request.Gender,
            Age=request.Age,
            Product_Category_1=request.Product_Category_1
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch predictions (JSON)
    """
    if model is None or dv is None:
        raise HTTPException(status_code=503, detail="Artifacts not loaded")

    try:
        results = []
        for item in request.predictions:
            data = item.dict()
            processed_data = preprocess_single_record(data)
            
            X = dv.transform([processed_data])
            prediction = model.predict(X)[0]
            prediction = max(0, float(prediction))

            results.append(
                PredictionResponse(
                    purchase_prediction=round(prediction, 2),
                    Gender=item.Gender,
                    Age=item.Age,
                    Product_Category_1=item.Product_Category_1
                )
            )

        return BatchPredictionResponse(
            predictions=results,
            count=len(results)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.post("/predict_csv")
async def predict_from_data(data: List[dict]):
    """
    Make predictions from raw data (useful for CSV/JSON uploads as list of dicts)
    """
    if model is None or dv is None:
        raise HTTPException(status_code=503, detail="Artifacts not loaded")

    try:
        # Apply preprocessing to all items
        processed_list = [preprocess_single_record(item) for item in data]

        # Transform all at once
        X = dv.transform(processed_list)
        
        # Predict
        predictions = model.predict(X)
        predictions = np.maximum(predictions, 0)

        # Create response
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "index": i,
                "purchase_prediction": round(float(pred), 2),
                "Gender": data[i].get('Gender'),
                "Age": data[i].get('Age'),
                "Product_Category_1": data[i].get('Product_Category_1')
            })

        return {
            "predictions": results,
            "count": len(results)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
