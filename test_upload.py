import requests
import pandas as pd
import json

def test_file_upload():
    url = "http://localhost:8080/predict_file"
    file_path = "data/test_Vges7qu.csv"
    
    # Create a small sample CSV for testing
    sample_df = pd.read_csv(file_path).head(10)
    sample_csv = "sample_test.csv"
    sample_df.to_csv(sample_csv, index=False)
    
    print(f"Testing upload with {sample_csv}...")
    
    try:
        with open(sample_csv, 'rb') as f:
            files = {'file': (sample_csv, f, 'text/csv')}
            response = requests.post(url, files=files)
            
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Response Preview:")
            print(json.dumps(response.json(), indent=2)[:500])
        else:
            print("Error Response:")
            print(response.text)
            
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_file_upload()
