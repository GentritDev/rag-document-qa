"""
Application configuration using Pydantic Settings.

Loads values from a `.env` file (if present) and from environment
variables, so the same code runs locally, in Docker, and on
Hugging Face Spaces / Render without code changes.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the RAG application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Gemini LLM (free tier via Google AI Studio) ---
    groq_api_key: str
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0

    # --- Embeddings (local, free, runs on CPU) ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Storage paths ---
    vectorstore_dir: Path = Path("./vectorstore")
    data_dir: Path = Path("./data")

    # --- Text splitting ---
    chunk_size: int = 500
    chunk_overlap: int = 50

    # --- Retrieval ---
    retrieval_k: int = 3

    # --- Upload limits ---
    max_file_size_mb: int = 10

    # --- Embedding batching (for progress reporting) ---
    embedding_batch_size: int = 16

    # --- Logging ---
    log_level: str = "INFO"

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist yet."""
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


# Singleton instance, imported across the app.
settings = Settings()
settings.ensure_directories()
