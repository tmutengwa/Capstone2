import openai
import base64
from config import settings
from logger import logger

class LLMConsultant:
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.model_name = settings.LLM_MODEL
        self.temperature = settings.LLM_TEMPERATURE
        self.enabled = settings.ENABLE_LLM
        
        logger.info(f"AI Consultant Init: Enabled={self.enabled}, OPENAI_API_KEY_Set={self.api_key is not None}")
        
        if self.enabled and self.api_key:
            try:
                self.client = openai.OpenAI(api_key=self.api_key)
                logger.info(f"OpenAI Client configured successfully with model '{self.model_name}'.")
            except Exception as e:
                logger.error(f"Failed to configure OpenAI Client: {e}")
                self.enabled = False

    def is_available(self):
        return self.enabled and self.api_key is not None

    def _query_llm(self, system_prompt, user_prompt):
        if not self.is_available():
            return "AI Consultant is not available. Please configure OPENAI_API_KEY in .env file."
            
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI Query failed: {e}")
            return f"Error communicating with AI: {e}"

    def analyze_plot_image(self, plot_name: str, image_bytes: bytes):
        """
        Multimodal analysis using OpenAI Vision.
        """
        if not self.is_available():
            return "AI Consultant is not available."

        try:
            # Encode bytes to base64 string
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            prompt = (
                f"You are a Principal Data Visualization Architect. Analyze this '{plot_name}' plot from the Black Friday dataset. "
                "Produce a beautifully formatted Markdown report.\n\n"
                "### 🎨 Visual Momentum Analysis\n"
                "- **Shape & Distribution**: Explain what the skewness/clusters tell us about 'Whale' vs 'Average' shoppers.\n"
                "- **Patterns & Anomalies**: Identify striping, outliers, or categorical bottlenecks.\n"
                "- **Strategic Insight**: Shift your analysis to be **business-strategic**.\n\n"
                "> *Key Takeaway*: Summarize the most important finding in one sentence."
            )

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI Vision Query failed for {plot_name}: {e}")
            return f"Error analyzing plot: {e}"

    def generate_summary_insight(self, data_summary: str):
        """
        Generate an executive summary based on dataset statistics.
        """
        system_prompt = (
            "You are a Senior Data Scientist. Provide a concise, business-oriented executive summary "
            "of the dataset based on the provided statistics. Highlight key data quality issues "
            "and potential analytical directions."
        )
        return self._query_llm(system_prompt, data_summary)

    def analyze_diagnostics(self, context: str):
        """
        Explain diagnostic test results and skewness/kurtosis with detailed interpretation.
        """
        system_prompt = (
            """You are a Senior Data Reliability Engineer. Perform a 'Data Health Assessment'.
Analyze the provided stats and metrics to generate a concise report.

**STRICT RULES**:
- **NO TITLES OR INTROS**: Start directly with the first section header.
- **NO TABLES**: Use bullet points or short paragraphs only.
- **BE CONCISE**: Limit to key risks and actions.

Include these sections:
### 🟢 Statistical Health Check
Analyze sparsity, Skewness/Kurtosis, and Normality. State if transformation is required.

### 🔵 Relationship & Cardinality
Identify dependency risks and high-cardinality IDs.

### 🔴 Business Action Plan
Provide specific 'Possible Actions' and 2-3 high-impact interaction features.

### 🚀 Model Strategy
Briefly suggest the best model type and why."""
        )
        return self._query_llm(system_prompt, context)

    def analyze_visuals(self, correlation_matrix: str):
        """
        Generate detailed insights based on the correlation matrix (Heatmap).
        """
        system_prompt = (
            """You are a Senior Data Scientist analyzing a Correlation Heatmap. 
Generate a comprehensive visual analysis report.

**STRICT RULES**:
- **NO TITLES OR INTROS**: Start directly with the first insight header.
- **NO FLUFF**: Go straight to the interpretation.

Follow this structure:

#### 1. Dominance of [Strongest Feature]
   - **Interpretation**: Explain relationship.
   - **Business Impact**: How this influences strategy.

#### 2. Demographic Insights (Weak Correlations)
   - **Interpretation**: Explain non-linearity.
   - **The Nuance**: Value for complex models.

#### 3. Secondary Influencers
   - **Interpretation**: Describe trends.
   - **Business Impact**: Why capture these.

#### 4. Independence of Predictors
   - **Interpretation**: Explain independence (good for ML).

---
#### Summary Recommendation
   - Concluding strategy (e.g., Feature Engineering)."""
        )
        return self._query_llm(system_prompt, f"Correlation Matrix:\n{correlation_matrix}")

    def interpret_test_result(self, test_result: dict, col1: str, col2: str):
        """
        Explain a statistical test result in plain English.
        """
        system_prompt = (
            "You are a helpful statistical consultant. Explain the results of this statistical test "
            "to a non-technical business stakeholder. Focus on whether there is a significant relationship "
            "and the strength of that relationship. Avoid jargon where possible."
        )
        user_prompt = f"Variables: '{col1}' and '{col2}'.\nTest Results: {test_result}"
        return self._query_llm(system_prompt, user_prompt)

    def suggest_feature_engineering(self, columns_info: str):
        """
        Suggest feature engineering steps.
        """
        system_prompt = (
            "You are a Feature Engineering expert. Suggest 3-5 high-impact feature transformations "
            "for this dataset. Explain WHY each transformation would help a machine learning model."
        )
        return self._query_llm(system_prompt, columns_info)

    def answer_question(self, context: str, question: str):
        """
        Answer a user's specific question about the data.
        """
        system_prompt = (
            "You are an expert Data Analyst. Answer the user's question directly using the provided statistical context (Correlations, Group Averages, Data Sample). "
            "Do not be vague or generic. Do not say 'as an AI I cannot...'. "
            "Instead, interpret the provided numbers to give a concrete answer. "
            "If the data shows a trend (e.g., Age Group X spends more), state it clearly."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
        return self._query_llm(system_prompt, user_prompt)