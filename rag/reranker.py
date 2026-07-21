"""
Reranker module using CrossEncoder (`BAAI/bge-reranker-base`) for high-precision candidate filtering.
"""

from typing import List, Dict, Any
from utils.config import CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)


class DeterministicFallbackReranker:
    """Lightweight deterministic keyword-overlap and RRF score cross-encoder fallback when PyTorch weights are loading or offline."""

    def predict(self, pairs: List[List[str]]) -> List[float]:
        scores = []
        for query, content in pairs:
            q_words = set(query.lower().split())
            c_words = set(content.lower().split())
            overlap = len(q_words.intersection(c_words)) / max(1, len(q_words))
            scores.append(min(0.95, max(0.15, overlap * 0.8 + 0.15)))
        return scores


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                logger.info(
                    f"Loading CrossEncoder reranker model: {self.model_name}..."
                )
                self._model = CrossEncoder(self.model_name)
                logger.info("CrossEncoder loaded successfully.")
            except Exception as e:
                logger.warning(
                    f"Failed to load {self.model_name}: {e}. Using fast DeterministicFallbackReranker engine."
                )
                self._model = DeterministicFallbackReranker()
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = CONFIG.FINAL_RERANK_TOP_K,
    ) -> List[Dict[str, Any]]:
        """Re-rank candidate chunks against the user query for maximum precision."""
        if not candidates:
            return []

        model = self.model
        if model and model is not False:
            pairs = [[query, doc["content"]] for doc in candidates]
            try:
                scores = model.predict(pairs)
                for i, doc in enumerate(candidates):
                    doc["rerank_score"] = float(scores[i])
                # Sort descending by cross-encoder score
                candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            except Exception as e:
                logger.error("Error during cross-encoder prediction: {}", e)
                candidates.sort(
                    key=lambda x: x.get("rrf_score", x.get("dense_score", 0.0)),
                    reverse=True,
                )
                for doc in candidates:
                    doc["rerank_score"] = doc.get(
                        "rrf_score", doc.get("dense_score", 0.0)
                    )
        else:
            candidates.sort(
                key=lambda x: x.get("rrf_score", x.get("dense_score", 0.0)),
                reverse=True,
            )
            for doc in candidates:
                doc["rerank_score"] = doc.get("rrf_score", doc.get("dense_score", 0.0))

        top_candidates = candidates[:top_k]
        for item in top_candidates:
            # BGE Reranker output raw scores around -10 to +10. Normalize cleanly
            raw_s = item.get("rerank_score", 0.5)
            if raw_s > 1.0 or raw_s < 0.0:
                import math

                # Sigmoid or bounded scaling for clean percentage display
                norm_s = (
                    1.0 / (1.0 + math.exp(-raw_s))
                    if -20 < raw_s < 20
                    else (0.99 if raw_s >= 20 else 0.05)
                )
            else:
                norm_s = raw_s
            item["confidence_score"] = min(0.99, max(0.10, norm_s))

        return top_candidates
