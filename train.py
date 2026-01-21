import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import DictVectorizer
import joblib
import time
import os
import logging
import warnings
from feature_engineering import preprocess_features
# Import the evaluation function (will create evaluate.py next)
from evaluate import evaluate_model_and_llm

# Suppress specific sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")

# Configure logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    filename='logs/train.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

def load_data(path):
    return pd.read_csv(path)

def train_multiple_models():
    logging.info("Starting model training pipeline.")
    
    # Create models directory if not exists
    if not os.path.exists('models'):
        os.makedirs('models')
        logging.info("Created models directory.")

    logging.info("Loading and Preprocessing data...")
    try:
        df = load_data('data/train.csv')
    except FileNotFoundError:
        logging.error("data/train.csv not found.")
        return

    df = preprocess_features(df)
    
    target = 'Purchase'
    
    # Define features
    categorical_features = ['Gender', 'Age', 'City_Category', 
                            'Gender_Age', 'City_Product', 'Job_Product', 'Life_Stage']
    numerical_features = ['Occupation', 'Stay_In_Current_City_Years', 'Marital_Status', 
                          'Product_Category_1', 'Product_Category_2', 'Product_Category_3']
    
    feature_cols = categorical_features + numerical_features
    
    # Split Data
    logging.info("Splitting data into train and validation sets.")
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
    
    y_train = df_train[target].values
    y_val = df_val[target].values
    
    # Convert to Dicts
    logging.info("Converting data to dictionaries and vectorizing...")
    train_dicts = df_train[feature_cols].to_dict(orient='records')
    val_dicts = df_val[feature_cols].to_dict(orient='records')
    
    # Vectorize
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)
    
    models = {
        'DecisionTree': DecisionTreeRegressor(random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1),
        'LightGBM': lgb.LGBMRegressor(objective='regression', n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1, verbosity=-1)
    }
    
    best_model_name = None
    best_rmse = float('inf')
    best_model_obj = None
    
    logging.info("--- Training Multiple Models ---")
    for name, model in models.items():
        start_time = time.time()
        logging.info(f"Training {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        
        duration = time.time() - start_time
        logging.info(f"{name} RMSE: {rmse:.4f} (Time: {duration:.2f}s)")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_model_obj = model

    logging.info(f"Best Model found: {best_model_name} with RMSE: {best_rmse:.4f}")
    
    # Hyperparameter Tuning for the Best Model
    logging.info(f"--- Hyperparameter Tuning for {best_model_name} ---")
    
    param_grids = {
        'DecisionTree': {
            'max_depth': [5, 10, 15, 20],
            'min_samples_split': [2, 10, 20]
        },
        'RandomForest': {
            'n_estimators': [50, 100],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 10]
        },
        'XGBoost': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.05, 0.1, 0.2]
        },
        'LightGBM': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.05, 0.1, 0.2],
            'num_leaves': [31, 50, 70]
        }
    }
    
    final_model = best_model_obj
    
    if best_model_name in param_grids:
        logging.info("Starting RandomizedSearchCV...")
        search = RandomizedSearchCV(
            models[best_model_name],
            param_distributions=param_grids[best_model_name], 
            n_iter=10,
            scoring='neg_root_mean_squared_error', 
            cv=3, 
            n_jobs=-1, 
            random_state=42,
            verbose=0 # Verbose output handles by us or sklearn, keeping logs clean
        )
        
        search.fit(X_train, y_train)
        
        logging.info(f"Best Parameters: {search.best_params_}")
        best_tuned_rmse = -search.best_score_
        logging.info(f"Best CV RMSE: {best_tuned_rmse:.4f}")
        
        final_model = search.best_estimator_
    else:
        logging.info("No hyperparameter grid defined. Using initial trained model.")

    # Final Evaluation
    y_pred_final = final_model.predict(X_val)
    final_rmse = np.sqrt(mean_squared_error(y_val, y_pred_final))
    logging.info(f"Final Validation RMSE after tuning: {final_rmse:.4f}")

    # Feature Importance
    try:
        feature_names = dv.get_feature_names_out()
        
        # Handle different models' feature importance attributes
        if hasattr(final_model, 'feature_importances_'):
            importances = final_model.feature_importances_
        elif hasattr(final_model, 'coef_'): # Linear models
            importances = np.abs(final_model.coef_)
        else:
            importances = None
            logging.warning("Model does not support feature importance extraction.")

        if importances is not None:
            # Create a sorted dataframe
            fi_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            logging.info("\n--- Feature Importance (Top 20) ---")
            logging.info(fi_df.head(20).to_string(index=False))
            
            # Save detailed feature importance
            fi_df.to_csv('models/feature_importance.csv', index=False)
            logging.info("Full feature importance saved to 'models/feature_importance.csv'")
            
    except Exception as e:
        logging.error(f"Error calculating feature importance: {e}")

    # Save
    logging.info("Saving best model and vectorizer...")
    try:
        with open('models/preprocessor.b', 'wb') as f_out:
            joblib.dump(dv, f_out)
        with open('models/best_model.pkl', 'wb') as f_out:
            joblib.dump(final_model, f_out)
        logging.info("Saved to models/preprocessor.b and models/best_model.pkl")
        
        # Run Evaluation
        logging.info("Starting model and LLM evaluation...")
        evaluate_model_and_llm()
        
    except Exception as e:
        logging.error(f"Failed to save artifacts: {e}")

if __name__ == "__main__":
    train_multiple_models()
