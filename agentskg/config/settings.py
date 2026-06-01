from functools import lru_cache
from typing import Dict, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration class"""

    GENERATE_AGENT_MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"
    # Base project configuration
    PROJECT_NAME: str = "AgentsKG"
    DEBUG: bool = False

    # API key configuration
    OPENAI_API_KEY: str
    OPENAI_API_BASE: Optional[str] = None

    # LLM model configuration
    DEFAULT_MODEL_NAME: str = "gpt-4o"
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_MAX_TOKENS: int = 1000

    # Database configuration
    DATABASE_URL: Optional[str] = None

    # Log configuration
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = "agentskg.log"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True
    )


@lru_cache()
def get_settings() -> Settings:
    """Get the configuration singleton"""
    return Settings()


# Export the configuration instance
settings = get_settings()
