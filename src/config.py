"""Application configurations loaded from environment variables."""

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API keys
    OPENAI_API_KEY: Optional[str] = Field(
        default=None, description="OpenAI API key"
    )
    LANGFUSE_PUBLIC_KEY: Optional[str] = Field(
        None, description="Langfuse public key for tracing"
    )
    LANGFUSE_SECRET_KEY: Optional[str] = Field(
        None, description="Langfuse secret key for tracing"
    )
    LANGFUSE_HOST: str = Field(
        default="https://cloud.langfuse.com", description="Langfuse host URL"
    )

    # application configurations
    ENVIRONMENT: str = Field(default="development", description="Environment name")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    API_PORT: int = Field(default=8000, description="FastAPI server port")

    # model configurations
    LLM_MODEL: str = Field(
        default="gpt-4o-mini", description="OpenAI model to use"
    )
    LLM_TEMPERATURE: float = Field(
        default=0.7, description="Temperature for LLM responses"
    )
    LLM_MAX_TOKENS: int = Field(default=1024, description="Max tokens per response")

    # paths
    DATA_DIR: Path = Field(
        default=Path("data"), description="Directory containing data files"
    )
    PROMPTS_DIR: Path = Field(
        default=Path("prompts"), description="Directory containing agent prompts"
    )
    CHROMA_PERSIST_DIR: Path = Field(
        default=Path("data/chroma_db"), description="ChromaDB persistence directory"
    )

    # vector store configurations
    EMBEDDING_MODEL: str = Field(
        default="text-embedding-3-small", description="OpenAI embedding model"
    )
    CHUNK_SIZE: int = Field(default=500, description="Document chunk size for RAG")
    CHUNK_OVERLAP: int = Field(default=50, description="Overlap between chunks")
    TOP_K_RESULTS: int = Field(
        default=3, description="Number of RAG results to retrieve"
    )

    @field_validator("OPENAI_API_KEY")
    @classmethod
    def validate_openai_key(cls, v: Optional[str]) -> Optional[str]:
        """Validate OpenAI API key is not placeholder."""
        if v and v.startswith("your_"):
            raise ValueError(
                "OPENAI_API_KEY must be set. Get one from https://platform.openai.com/account/api-keys"
            )
        return v

    @property
    def langfuse_enabled(self) -> bool:
        """Check if Langfuse tracing is configured."""
        return bool(self.LANGFUSE_PUBLIC_KEY and self.LANGFUSE_SECRET_KEY)


# global instance
settings = Settings()
