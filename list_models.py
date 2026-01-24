
import os
from config import settings
from google import genai

def list_models():
    api_key = settings.GOOGLE_API_KEY
    if not api_key:
        print("GOOGLE_API_KEY not found in settings.")
        return

    try:
        client = genai.Client(api_key=api_key)
        # The SDK documentation suggests using models.list() or similar. 
        # Since I am not 100% sure of the exact method on the 'genai.Client' object for this specific version 
        # (it changed recently from google.generativeai), I will try to inspect or find the method.
        # Actually, standard pattern for this new SDK is client.models.list()
        
        print("Attempting to list models...")
        for model in client.models.list():
            print(f"Name: {model.name}, Display Name: {model.display_name}")
            
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    list_models()
