"""
Configuration management for KnowledgeForge AI.
"""

from pathlib import Path
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    # Paths
    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    DATA_RAW_DIR: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "data" / "raw"
    )
    DATA_PROCESSED_DIR: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "data" / "processed"
    )
    CHROMA_DB_DIR: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "database" / "chroma"
    )
    MODELS_DIR: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "models_cache"
    )

    # Embedding Settings
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # Chunking Settings
    DEFAULT_CHUNK_SIZE: int = 500
    DEFAULT_CHUNK_OVERLAP: int = 50

    # Retrieval Settings
    DENSE_TOP_K: int = 15
    SPARSE_TOP_K: int = 15
    FINAL_RERANK_TOP_K: int = 5
    HYBRID_RRF_K: int = 60

    # Local LLM Settings
    OLLAMA_BASE_URL: str = Field(
        default_factory=lambda: __import__("os").getenv(
            "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
        )
    )
    DEFAULT_LLM_MODEL: str = "gemma:2b"
    TEMPERATURE: float = 0.2
    MAX_TOKENS: int = 1024

    def ensure_directories(self):
        """Ensure that all necessary directories exist."""
        self.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self.CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)


# Global configuration instance
CONFIG = AppConfig()
CONFIG.ensure_directories()
