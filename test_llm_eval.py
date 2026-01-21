import pandas as pd
from profiler import SmartProfiler
from llm_consultant import LLMConsultant
from orchestrator import DuckOrchestrator
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("LLMEval")

def evaluate_llm():
    logger.info("--- Starting LLM Evaluation ---")
    
    # 1. Load Data
    orch = DuckOrchestrator()
    try:
        df = orch.load_data('data/train.csv')
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    profiler = SmartProfiler(df)
    consultant = LLMConsultant()
    
    if not consultant.is_available():
        logger.error("LLM Consultant is not enabled or API key is missing.")
        return

    # 2. Generate Context (simulating eda.py logic)
    advanced_stats = profiler.generate_advanced_stats()
    
    target = 'Purchase' if 'Purchase' in df.columns else None
    corr_info = ""
    if target:
        num_cols = df.select_dtypes(include=['number']).columns
        if len(num_cols) > 1:
            corrs = df[num_cols].corr()[target].sort_values(ascending=False).to_string()
            corr_info = f"\nCorrelations with {target}:\n{corrs}"

    context = (
        f"Columns: {list(df.columns)}\n"
        f"{advanced_stats}\n"
        f"{corr_info}\n"
        f"Data Sample (first 10 rows):\n{df.head(10).to_string()}"
    )

    # 3. Define Test Questions
    questions = [
        "What is the average purchase amount?",
        "Who are the top 1% of spenders?",
        "Which product category has the highest variance in price?",
        "Does staying longer in a city affect purchase volume?",
        # Complex/Tricky question to test "I don't know" behavior vs inference
        "If a user buys Product_Category_5, what else do they buy?" 
    ]

    # 4. Run Evaluation
    results = []
    for q in questions:
        logger.info(f"\nQuestion: {q}")
        try:
            response = consultant.answer_question(context, q)
            logger.info(f"Answer: {response[:200]}...") # Log first 200 chars
            results.append({"Question": q, "Response_Length": len(response), "Response_Preview": response[:100]})
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            results.append({"Question": q, "Error": str(e)})

    # 5. Summary
    logger.info("\n--- Evaluation Complete ---")
    logger.info(pd.DataFrame(results))

if __name__ == "__main__":
    evaluate_llm()
