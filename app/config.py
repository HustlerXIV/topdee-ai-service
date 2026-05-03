"""Settings loaded from env. Validated at import time."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ai_service_port: int = 8000

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "topdee_chunks"

    # LLM (Google Gemini via LangChain)
    google_api_key: str | None = None
    llm_model: str = "gemini-2.0-flash"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Ingestion
    chunk_size: int = 800
    chunk_overlap: int = 100


settings = Settings()
