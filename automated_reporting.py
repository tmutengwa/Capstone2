import pandas as pd
import json
import plotly.io as pio
from profiler import SmartProfiler
from llm_consultant import LLMConsultant
from logger import logger

def run_automated_ai_assessment(file_path):
    """
    Orchestrates the EDA process: Extracts structural diagnostics, 
    injects mandatory Retail Logic, and generates Multimodal AI Reports.
    """
    try:
        # 1. Load Data
        df = pd.read_csv(file_path)
        profiler = SmartProfiler(df)
        consultant = LLMConsultant()
        
        if not consultant.is_available():
            logger.error("Gemini API not configured.")
            return "Gemini API not configured.", ""

        # 2. Extract Numerical Proof of Hierarchy
        hierarchy_stats = profiler.get_hierarchical_analysis()
        correlations = profiler.get_correlations()
        
        # --- SECTION 1: DATA HEALTH (Persona: Senior Retail Data Strategist) ---
        health_system_prompt = (
            "### PERSONA: Senior Retail Data Strategist & SRE\n"
            "### MANDATORY STRUCTURAL LOGIC (OVERRIDE STANDARD RULES):\n"
            "1. HIERARCHY VERIFIED: Nesting diagnostics confirm Product_Category_2/3 are hierarchical children. "
            "A NULL value here is a 'Structural Null' indicating a 'Basic/Single-Category' product.\n"
            "2. NO DELETION: You are strictly forbidden from suggesting row deletion for Category nulls. "
            "This would destroy 70% of the valid transaction record.\n"
            "3. IMPUTATION: Recommend constant imputation with '0'. Explain that '0' acts as a "
            "discrete mathematical flag for 'No Sub-category' for tree-based models like XGBoost.\n"
            "4. CARDINALITY: Flag User_ID and Product_ID for Target Encoding to prevent dimensionality explosion."
        )

        health_context = {
            "hierarchy_evidence": hierarchy_stats,
            "missing_values": df.isnull().sum().to_dict(),
            "distributions": profiler.get_distribution_analysis(),
            "sample_data": df.head(3).to_dict(orient='records')
        }

        logger.info("Generating Data Health Assessment with Structural Logic...")
        health_report = consultant._query_llm(health_system_prompt, json.dumps(health_context))

        # --- SECTION 2: VISUAL ANALYSIS (Persona: Principal Data Visualization Architect) ---
        # We use a multimodal approach if possible, or high-fidelity metadata
        visual_system_prompt = (
            "### PERSONA: Principal Data Visualization Architect\n"
            "### MANDATORY VISUAL INTERPRETATION:\n"
            "1. THE PRICE TIER INDEX: Interpret the negative correlation (-0.34) in the 'Purchase Distribution' "
            "as a Price Tier Index. Lower IDs (1, 2, 3) are High-Value/Luxury; higher IDs are budget items.\n"
            "2. BUNDLING EVIDENCE: Interpret the high correlation (0.54) between Categories as 'Bundle Evidence' "
            "and structural validation, NOT as multicollinearity to be removed.\n"
            "3. WHALE ANALYSIS: Focus on the KDE and Box plots. Identify the 'Whale' shoppers in the right-hand tail "
            "and explain what the visual skewness (0.60) means for high-margin targeting.\n"
            "4. NON-LINEAR SEGMENTATION: Analyze the Marital/Gender Heatmap. Identify clusters where specific "
            "demographics dominate high-value product tiers."
        )

        # Generating the list of plots and their "Visual Facts" (metadata approach)
        plots = profiler.generate_all_plots()
        visual_facts = {
            "Plot_List": [
                "Purchase Distribution with KDE and Box",
                "Average Purchase by Gender",
                "Average Purchase by Age Group",
                "Transaction Volume by City Category",
                "Purchase by Marital Status and Gender",
                "Pairwise Scatter Matrix (Sample n=10000)"
            ],
            "Correlations": correlations,
            "Categorical_Trends": profiler.get_categorical_analysis(),
            "Hierarchy_Context": hierarchy_stats
        }

        logger.info("Generating Strategic Visual Analysis Report...")
        visual_report = consultant._query_llm(visual_system_prompt, json.dumps(visual_facts))

        return health_report, visual_report

    except Exception as e:
        logger.error(f"Failed automated assessment: {e}")
        return f"Assessment Error: {e}", ""

# Run the execution
if __name__ == "__main__":
    health, visual = run_automated_ai_assessment('data/train.csv')
    
    if health and visual:
        logger.info("\n" + "="*50)
        logger.info("🛡️ DATA HEALTH ASSESSMENT")
        logger.info("="*50 + "\n" + health)
        
        logger.info("\n" + "="*50)
        logger.info("📊 VISUAL ANALYTICS STRATEGY")
        logger.info("="*50 + "\n" + visual)
    else:
        logger.warning("Assessment incomplete.")