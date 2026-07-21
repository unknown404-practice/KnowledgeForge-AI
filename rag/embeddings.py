"""
Embedding service orchestrating dense vector generation for RAG chunks.
"""

from typing import List, Dict, Any
from models.embedding import EmbeddingModel
from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str = None):
        self.embedding_model = (
            EmbeddingModel(model_name) if model_name else EmbeddingModel()
        )

    def generate_embeddings_for_chunks(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Add embedding vectors directly to each chunk object."""
        if not chunks:
            return []
        texts = [chunk["content"] for chunk in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        vectors = self.embedding_model.embed_documents(texts)
        for chunk, vector in zip(chunks, vectors):
            chunk["embedding"] = vector
        return chunks

    def embed_query(self, query: str) -> List[float]:
        """Embed user query."""
        return self.embedding_model.embed_query(query)
