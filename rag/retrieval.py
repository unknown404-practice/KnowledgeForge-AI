"""
Hybrid Retrieval engine combining ChromaDB Dense Search and BM25 Sparse Search via Reciprocal Rank Fusion (RRF).
"""

from typing import List, Dict, Any
from rag.vector_store import ChromaVectorStore
from rag.embeddings import EmbeddingService
from utils.config import CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    def __init__(
        self, vector_store: ChromaVectorStore, embedding_service: EmbeddingService
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.bm25_model = None
        self.indexed_chunks = []
        self.refresh_bm25_index()

    def refresh_bm25_index(self):
        """Index all current chunks in ChromaDB into a fast BM25 keyword index."""
        try:
            from rank_bm25 import BM25Okapi

            self.indexed_chunks = self.vector_store.get_all_chunks()
            if self.indexed_chunks:
                tokenized_corpus = [
                    doc["content"].lower().split() for doc in self.indexed_chunks
                ]
                self.bm25_model = BM25Okapi(tokenized_corpus)
                logger.info(
                    f"BM25 index built across {len(self.indexed_chunks)} chunks."
                )
            else:
                self.bm25_model = None
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            self.bm25_model = None

    def sparse_search(
        self, query: str, top_k: int = CONFIG.SPARSE_TOP_K
    ) -> List[Dict[str, Any]]:
        """Perform sparse keyword retrieval using BM25Okapi."""
        if not self.bm25_model or not self.indexed_chunks:
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25_model.get_scores(tokenized_query)

        # Sort top indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]
        sparse_candidates = []
        for idx in top_indices:
            if scores[idx] > 0.0:
                doc = self.indexed_chunks[idx].copy()
                doc["sparse_score"] = float(scores[idx])
                sparse_candidates.append(doc)
        return sparse_candidates

    def retrieve(
        self, query: str, top_k: int = CONFIG.DENSE_TOP_K
    ) -> List[Dict[str, Any]]:
        """Execute Hybrid Retrieval using Reciprocal Rank Fusion (RRF)."""
        # 1. Dense search
        query_vec = self.embedding_service.embed_query(query)
        dense_results = self.vector_store.dense_search(query_vec, top_k=top_k)

        # 2. Sparse search
        sparse_results = self.sparse_search(query, top_k=top_k)

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        doc_map = {}

        # Process dense rankings
        for rank, item in enumerate(dense_results):
            doc_id = item["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (
                1.0 / (CONFIG.HYBRID_RRF_K + rank + 1)
            )
            doc_map[doc_id] = item

        # Process sparse rankings
        for rank, item in enumerate(sparse_results):
            doc_id = item["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (
                1.0 / (CONFIG.HYBRID_RRF_K + rank + 1)
            )
            if doc_id not in doc_map:
                doc_map[doc_id] = item

        # Sort combined results
        sorted_ids = sorted(
            rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True
        )

        hybrid_results = []
        for doc_id in sorted_ids[:top_k]:
            candidate = doc_map[doc_id]
            candidate["rrf_score"] = rrf_scores[doc_id]
            hybrid_results.append(candidate)

        return hybrid_results
