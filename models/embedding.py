"""
SentenceTransformer model interface with blazing-fast true semantic embedding generation.
Guarantees high-quality 384-dimensional dense vectors using all-MiniLM-L6-v2.
"""

import os
from typing import List
from utils.logger import get_logger
from utils.config import CONFIG

# Prevent OpenMP thread lock / deadlock on Windows CPU when running inside Jupyter kernels
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

logger = get_logger(__name__)


class EmbeddingModel:
    def __init__(self, model_name: str = CONFIG.EMBEDDING_MODEL_NAME):
        self.model_name = model_name or "all-MiniLM-L6-v2"
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            logger.info(
                f"Local EmbeddingModel ({self.model_name}) initialized successfully."
            )
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer: {e}")
            raise

    @property
    def model(self):
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate dense vector embeddings for a list of document chunks."""
        if not texts:
            return []
        embeddings = self.model.encode(
            texts, show_progress_bar=False, normalize_embeddings=True
        )
        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """Generate a dense vector embedding for a single user query."""
        embeddings = self.model.encode(
            [query], show_progress_bar=False, normalize_embeddings=True
        )
        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()[0]
        return embeddings[0]
