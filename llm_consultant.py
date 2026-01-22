from google import genai
from google.genai import types
import PIL.Image
import os
import logging
from config import settings
from logger import logger

class PromptFactory:
    """
    Dynamic System Instruction Factory.
    Anchors Gemini 2.5 Flash-Lite in mathematical 'Ground Truths' from the Lattice discovery.
    """
    @staticmethod
    def create_grounded_instruction(eda_metadata):
        instruction = (
            "You are the Godobori Senior Data Architect. You MUST build your analysis from the bottom-up using these truths:\n"
            f"1. LATTICE ROOT: {eda_metadata.get('Lattice_Root', 'N/A')}. This is the primary anchor of the dataset.\n"
            f"2. PRICE SIGNATURE: {eda_metadata.get('Price_Signature', 'N/A')}. Missingness in Category 3 indicates a pricing floor.\n"
            f"3. BIAS SIMULATION: {eda_metadata.get('Simulation_Bias', 'N/A')}.\n"
            "4. NO GENERALIZATION: Use only provided stats and images. Do not use generic internet knowledge.\n"
            "5. SEQUENTIAL ANALYSIS: Analyze all 7 evidence plots in strict numerical order (1-7).\n"
            "6. FORMATTING: Heading (Bold 14), Subheading (Bold 13), Body (Regular 12)."
        )
        return instruction

    @staticmethod
    def get_report_instruction(eda_metadata):
        return (
            "You are the Godobori Senior Data Architect. Perform a 'Technical Analysis & Strategic Discovery'.\n"
            f"Lattice Root: {eda_metadata.get('Lattice_Root', 'N/A')}.\n"
            "Analyze the provided visual evidence sequentially."
        )

    @staticmethod
    def get_health_instruction(eda_metadata):
        return (
            "You are a Senior Data Reliability Engineer. Perform a 'Data Health Assessment'.\n"
            f"Grounded Fact: Price Signature is {eda_metadata.get('Price_Signature', 'N/A')}.\n"
            "Focus on explaining why this signature requires 'Labelling' vs 'Dropping' for budget segment preservation."
        )

    @staticmethod
    def get_summary_instruction(eda_metadata):
        return (
            "You are the Godobori Executive Consultant. Provide a high-level summary of the dataset.\n"
            f"Lattice Root: {eda_metadata.get('Lattice_Root', 'N/A')}.\n"
            f"Price Signature: {eda_metadata.get('Price_Signature', 'N/A')}."
        )

    @staticmethod
    def get_chat_instruction(eda_metadata):
        return (
            "You are the Godobori AI Consultant. Ground every answer in these structural truths:\n"
            f"{eda_metadata}\n"
            "Cite the Lattice Check or Price Signature as mathematical proof for structural claims."
        )

class LLMConsultant:
    def __init__(self, cache_file="insights_cache.txt"):
        self.api_key = settings.GOOGLE_API_KEY
        self.model_name = settings.LLM_MODEL
        self.enabled = settings.ENABLE_LLM
        self.cache_file = cache_file
        self.temperature = settings.LLM_TEMPERATURE
        self.client = None
        
        logger.info(f"Gemini Consultant Init: Enabled={self.enabled}")
        
        if self.enabled and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info(f"Gemini Client configured successfully with model '{self.model_name}'.")
            except Exception as e:
                logger.error(f"Failed to configure Gemini Client: {e}")
                self.enabled = False

    def is_available(self):
        return self.enabled and self.client is not None

    def get_persisted_insights(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                return f.read()
        return None

    def generate_grounded_report(self, plot_paths: dict, eda_metadata: dict):
        """Sequential Multimodal Analysis anchored by the PromptFactory."""
        if not self.is_available():
            return "AI Consultant is not available."

        system_instruction = PromptFactory.get_report_instruction(eda_metadata)

        try:
            prompt_parts = ["## **Technical Analysis & Strategic Discovery**\nPerform sequential analysis:"]
            
            for key in sorted(plot_paths.keys()):
                path = plot_paths[key]
                if os.path.exists(path):
                    img = PIL.Image.open(path)
                    prompt_parts.append(f"\n### Analysis of Evidence Plot: {key}")
                    prompt_parts.append(img)

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt_parts,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=self.temperature
                )
            )
            report = response.text
            
            with open(self.cache_file, "w") as f:
                f.write(report)
            return report

        except Exception as e:
            logger.error(f"Gemini Generation failed: {e}")
            return f"Error communicating with Gemini: {e}"

    def analyze_diagnostics(self, context: str, eda_metadata: dict):
        """Grounded Data Health Assessment."""
        system_prompt = PromptFactory.get_health_instruction(eda_metadata)
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=context,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=self.temperature
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini Diagnostics failed: {e}")
            return f"Error: {e}"

    def generate_summary_insight(self, context: str, eda_metadata: dict):
        """Generates a high-level summary insight."""
        system_prompt = PromptFactory.get_summary_instruction(eda_metadata)
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=context,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=self.temperature
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini Summary failed: {e}")
            return f"Error: {e}"

    def answer_question(self, context: str, question: str, eda_metadata: dict):
        """Grounded Q&A via Godobori chat agent."""
        system_instruction = PromptFactory.get_chat_instruction(eda_metadata)
        user_prompt = f"Statistical Context:\n{context}\n\nUser Question: {question}"
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=self.temperature
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini Q&A failed: {e}")
            return f"Error: {e}"