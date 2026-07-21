"""
Final RAG Generator orchestrating retrieval, context formatting, and LLM generation with exact roadmap response metadata.
"""

from typing import List, Dict, Any, Generator, Tuple, Optional
from rag.prompts import build_context_block, get_rag_system_prompt, format_final_prompt
from utils.config import CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)


class RAGGenerator:
    def __init__(
        self,
        retriever: Optional[Any] = None,
        reranker: Optional[Any] = None,
        llm: Optional[Any] = None,
    ):
        # Support flexible argument ordering or single llm initialization
        if retriever and not reranker and not llm:
            self.llm = retriever
            self.retriever = None
            self.reranker = None
        else:
            self.retriever = retriever
            self.reranker = reranker
            self.llm = llm

    def calculate_confidence_estimate(
        self, reranked_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Derive an estimated confidence score (0-100%) cleanly labeled per ultimateroadmap.md."""
        if not reranked_chunks:
            return {
                "score_percentage": 0.0,
                "label": "Low Confidence (Estimated)",
                "details": "No relevant document chunks retrieved.",
            }

        top_chunk = (
            reranked_chunks[0]
            if isinstance(reranked_chunks, list) and len(reranked_chunks) > 0
            else {}
        )
        top_score = top_chunk.get(
            "rerank_score",
            top_chunk.get(
                "confidence_score",
                top_chunk.get(
                    "rrf_score",
                    top_chunk.get("dense_score", top_chunk.get("score", 0.0)),
                ),
            ),
        )
        # Normalize cross-encoder or RRF score to estimated percentage (0.0 to 100.0)
        # BGE CrossEncoder typically scores around -10 to +10, or RRF scores ~0.01 to 0.05
        if top_score > 1.0:
            pct = min(100.0, max(0.0, (top_score + 5.0) / 15.0 * 100.0))
        elif top_score > 0.0:
            pct = min(
                100.0, top_score * 100.0 if top_score > 0.5 else top_score * 2000.0
            )
        else:
            pct = max(5.0, min(100.0, (top_score + 10.0) / 10.0 * 100.0))

        if pct >= 75.0:
            label = "High Confidence (Estimated)"
        elif pct >= 40.0:
            label = "Moderate Confidence (Estimated)"
        else:
            label = "Low Confidence (Estimated)"

        return {
            "score_percentage": round(pct, 1),
            "label": label,
            "details": "Estimated precision derived from Cross-Encoder semantic reranking scores.",
        }

    def generate_response(
        self,
        query: str,
        top_k: int = CONFIG.FINAL_RERANK_TOP_K,
        use_reranker: bool = True,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate a complete synchronous response with exact answer and sources with confidence estimates."""
        # 1. Hybrid Retrieval
        candidates = self.retriever.retrieve(query, top_k=CONFIG.DENSE_TOP_K)
        logger.info(f"Retrieved {len(candidates)} hybrid candidates for query.")

        # 2. Rerank
        if use_reranker and self.reranker:
            reranked_chunks = self.reranker.rerank(query, candidates, top_k=top_k)
            logger.info(f"Selected top {len(reranked_chunks)} reranked chunks.")
        else:
            reranked_chunks = candidates[:top_k]

        # 3. Calculate Confidence Indicator (Roadmap Step 10)
        confidence = self.calculate_confidence_estimate(reranked_chunks)
        self.last_confidence = confidence
        for chunk in reranked_chunks:
            chunk["confidence_estimate"] = confidence

        # 4. Build Context
        context_str = build_context_block(reranked_chunks)
        system_prompt = get_rag_system_prompt()
        user_prompt = format_final_prompt(query, context_str)

        # 5. Generate Response
        answer = self.llm.generate(user_prompt, system_prompt=system_prompt)
        return answer, reranked_chunks

    def stream_response(
        self, query: str, top_k: int = CONFIG.FINAL_RERANK_TOP_K
    ) -> Tuple[Generator[str, None, None], List[Dict[str, Any]]]:
        """Generate a streaming response along with retrieved sources and confidence estimates."""
        candidates = self.retriever.retrieve(query, top_k=CONFIG.DENSE_TOP_K)
        reranked_chunks = self.reranker.rerank(query, candidates, top_k=top_k)
        confidence = self.calculate_confidence_estimate(reranked_chunks)
        self.last_confidence = confidence
        for chunk in reranked_chunks:
            chunk["confidence_estimate"] = confidence

        context_str = build_context_block(reranked_chunks)
        system_prompt = get_rag_system_prompt()
        user_prompt = format_final_prompt(query, context_str)

        stream = self.llm.stream_generate(user_prompt, system_prompt=system_prompt)
        return stream, reranked_chunks
