import pandas as pd

def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all preprocessing and feature engineering steps to the dataframe.
    """
    df = df.copy()
    
    # Handle 'Stay_In_Current_City_Years'
    if 'Stay_In_Current_City_Years' in df.columns:
        # Check if type is string before replace to avoid warnings/errors
        if df['Stay_In_Current_City_Years'].dtype == 'object':
            df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].replace('4+', '4')
        df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].astype(int)
    
    # Handle Missing Values in Product Categories
    if 'Product_Category_2' in df.columns:
        df['Product_Category_2'] = df['Product_Category_2'].fillna(0).astype(int)
    if 'Product_Category_3' in df.columns:
        df['Product_Category_3'] = df['Product_Category_3'].fillna(0).astype(int)
    
    # --- Feature Engineering ---
    
    # 1. Gender + Age (Gender_Age)
    if 'Gender' in df.columns and 'Age' in df.columns:
        df['Gender_Age'] = df['Gender'] + "_" + df['Age']
    
    # 2. City_Category + Product_Category_1 (City_Product)
    if 'City_Category' in df.columns and 'Product_Category_1' in df.columns:
        df['City_Product'] = df['City_Category'] + "_" + df['Product_Category_1'].astype(str)
    
    # 3. Occupation + Product_Category_1 (Job_Product)
    if 'Occupation' in df.columns and 'Product_Category_1' in df.columns:
        df['Job_Product'] = df['Occupation'].astype(str) + "_" + df['Product_Category_1'].astype(str)
    
    # 4. Age + Marital_Status (Life_Stage)
    if 'Age' in df.columns and 'Marital_Status' in df.columns:
        df['Life_Stage'] = df['Age'] + "_" + df['Marital_Status'].astype(str)
    
    return df

def preprocess_single_record(data: dict) -> dict:
    """
    Apply preprocessing to a single dictionary record (for API serving).
    """
    # Create a clean dictionary for the model
    processed = {}
    
    # Helper to safe get
    def get_val(key, default=""):
        return str(data.get(key, default))

    # --- Basic Cleaning ---
    
    # Stay_In_Current_City_Years
    stay_val = get_val('Stay_In_Current_City_Years', "0")
    if stay_val == '4+':
        processed['Stay_In_Current_City_Years'] = 4
    else:
        try:
            processed['Stay_In_Current_City_Years'] = int(float(stay_val)) # Handle "2.0" string
        except ValueError:
            processed['Stay_In_Current_City_Years'] = 0

    # Helper to safe get int (handles None, NaN, strings)
    def get_int(key, default=0):
        val = data.get(key)
        if pd.isna(val) or val is None or val == "":
            return default
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return default

    # Numerical Inputs (with defaults)
    processed['Occupation'] = get_int('Occupation')
    processed['Marital_Status'] = get_int('Marital_Status')
    processed['Product_Category_1'] = get_int('Product_Category_1')
    processed['Product_Category_2'] = get_int('Product_Category_2')
    processed['Product_Category_3'] = get_int('Product_Category_3')

    # Categorical Inputs
    processed['Gender'] = get_val('Gender')
    processed['Age'] = get_val('Age')
    processed['City_Category'] = get_val('City_Category')

    # --- Feature Engineering ---
    
    # 1. Gender + Age
    processed['Gender_Age'] = f"{processed['Gender']}_{processed['Age']}"
    
    # 2. City_Product
    processed['City_Product'] = f"{processed['City_Category']}_{processed['Product_Category_1']}"
    
    # 3. Job_Product
    processed['Job_Product'] = f"{processed['Occupation']}_{processed['Product_Category_1']}"
    
    # 4. Life_Stage
    processed['Life_Stage'] = f"{processed['Age']}_{processed['Marital_Status']}"
        
    return processed
