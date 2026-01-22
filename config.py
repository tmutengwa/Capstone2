from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os

class Settings(BaseSettings):
    # Data Settings
    MAX_FILE_SIZE_MB: int = 500
    LARGE_DATASET_THRESHOLD: int = 1000000
    SAMPLE_SIZE_LARGE: int = 50000

    # Statistical Thresholds
    SIGNIFICANCE_LEVEL: float = 0.05
    CATEGORICAL_THRESHOLD: int = 15
    HIGH_CARDINALITY_THRESHOLD: int = 50
    SKEWNESS_THRESHOLD: float = 1.0

    # DuckDB Settings
    DUCKDB_MEMORY_LIMIT: str = "4GB"
    DUCKDB_THREADS: int = 4

    # LLM Settings
    ENABLE_LLM: bool = True
    OPENAI_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    LLM_MODEL: str = "gemini-1.5-pro"
    LLM_TEMPERATURE: float = 0.1

    # Cloud Settings (Optional)
    GOOGLE_PROJECT_ID: Optional[str] = None
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_DEFAULT_REGION: Optional[str] = "us-east-1"
    AZURE_STORAGE_CONNECTION_STRING: Optional[str] = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

# Create a singleton instance
settings = Settings()