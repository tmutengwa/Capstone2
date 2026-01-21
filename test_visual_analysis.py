import pandas as pd
from profiler import SmartProfiler
from llm_consultant import LLMConsultant
from logger import logger
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def test_visual_analysis():
    print("--- Starting Visual Analysis Test ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv('data/train.csv')
    except FileNotFoundError:
        print("Error: data/train.csv not found.")
        return

    profiler = SmartProfiler(df)
    consultant = LLMConsultant()
    
    if not consultant.is_available():
        print("LLM Consultant not configured.")
        return

    # 2. Generate Plots
    print("Generating plots...")
    plots = profiler.generate_all_plots()
    
    if "Purchase Distribution" in plots:
        print("Converting 'Purchase Distribution' to image bytes...")
        fig = plots["Purchase Distribution"]
        img_bytes = profiler.get_plot_bytes(fig)
        
        print("Sending image to Gemini for analysis...")
        analysis = consultant.analyze_plot_image("Purchase Distribution", img_bytes)
        
        print("\n" + "="*50)
        print("📸 VISUAL ANALYSIS RESULT")
        print("="*50 + "\n" + analysis)
    else:
        print("Purchase Distribution plot not found.")

if __name__ == "__main__":
    test_visual_analysis()
