import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from feature_engineering import preprocess_features
from llm_consultant import LLMConsultant
from profiler import SmartProfiler
import logging
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")

# Setup specific logger for diagnostics
if not os.path.exists('logs'):
    os.makedirs('logs')

# Create a custom logger
diag_logger = logging.getLogger("ModelDiagnostics")
diag_logger.setLevel(logging.INFO)
# Avoid adding multiple handlers if re-imported
if not diag_logger.handlers:
    file_handler = logging.FileHandler('logs/model_diagnostics.log', mode='w')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    diag_logger.addHandler(file_handler)
    # Also print to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    diag_logger.addHandler(console_handler)

def evaluate_model_and_llm():
    diag_logger.info("=========================================")
    diag_logger.info("   DATA SCIENCE PERFORMANCE REPORT")
    diag_logger.info("=========================================\n")

    # 1. Model Performance Metrics
    diag_logger.info("--- [1/2] Model Performance ---")
    try:
        model = joblib.load('models/best_model.pkl')
        dv = joblib.load('models/preprocessor.b')
        df = pd.read_csv('data/train.csv')
        
        # Split a small validation set for quick evaluation (same random_state as train.py to match validation set)
        _, df_val = train_test_split(df, test_size=0.2, random_state=42)
        
        y_true = df_val['Purchase'].values
        X_val_processed = preprocess_features(df_val)
        
        # Define features (Must match train.py exactly)
        categorical_features = ['Gender', 'Age', 'City_Category', 
                                'Gender_Age', 'City_Product', 'Job_Product', 'Life_Stage']
        numerical_features = ['Occupation', 'Stay_In_Current_City_Years', 'Marital_Status', 
                              'Product_Category_1', 'Product_Category_2', 'Product_Category_3']
        feature_cols = categorical_features + numerical_features
        
        val_dicts = X_val_processed[feature_cols].to_dict(orient='records')
        X_val = dv.transform(val_dicts)
        
        y_pred = model.predict(X_val)
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        diag_logger.info(f"RMSE: {rmse:.2f}")
        diag_logger.info(f"MAE:  {mae:.2f}")
        diag_logger.info(f"R2 Score: {r2:.4f}")
        
    except Exception as e:
        diag_logger.error(f"Model evaluation failed: {e}")

    # 2. LLM Reasoning Evaluation
    diag_logger.info("\n--- [2/2] LLM Analyst Quality ---")
    consultant = LLMConsultant()
    if not consultant.is_available():
        diag_logger.warning("LLM not enabled or API key missing. Skipping.")
    else:
        try:
            profiler = SmartProfiler(df)
            context = profiler.generate_advanced_stats()
            
            test_questions = [
                "Who are the top 1% spenders?",
                "Which product is the 'Loss Leader'?"
            ]
            
            for q in test_questions:
                diag_logger.info(f"\nQ: {q}")
                answer = consultant.answer_question(context, q)
                diag_logger.info(f"A: {answer}")
                
        except Exception as e:
            diag_logger.error(f"LLM evaluation failed: {e}")

    diag_logger.info("\n=========================================")

if __name__ == "__main__":
    evaluate_model_and_llm()
