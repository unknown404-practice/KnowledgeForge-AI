"""
Unit tests for Step 5: Reranker & Generator (`rag/reranker.py`, `rag/generator.py`, `rag/prompts.py`).
Run with: `pytest tests/test_05_reranker_and_generator.py -v`
"""

import unittest
import warnings
import tempfile
from pathlib import Path
from unittest.mock import patch

warnings.filterwarnings("ignore")

from rag.reranker import CrossEncoderReranker
from rag.prompts import build_context_block, format_final_prompt
from rag.generator import RAGGenerator
from rag.vector_store import ChromaVectorStore
from rag.embeddings import EmbeddingService
from rag.retrieval import HybridRetriever
from models.llm import DeepsetGemmaLLM


class TestRerankerAndGenerator(unittest.TestCase):
    def setUp(self):
        self.reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-base")
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_dir = Path(self.temp_dir.name) / "chroma_test"
        self.vector_store = ChromaVectorStore(
            persist_directory=self.db_dir, collection_name="gen_test"
        )
        self.embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")
        self.retriever = HybridRetriever(self.vector_store, self.embedding_service)
        self.llm = DeepsetGemmaLLM(
            model_name="gemma3:4b (Offline Analytical Engine)", use_native_weights=False
        )
        self.generator = RAGGenerator(self.retriever, self.reranker, self.llm)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_cross_encoder_reranking(self):
        query = "What is CrossEncoder?"
        candidates = [
            {
                "content": "CrossEncoder takes both query and candidate text simultaneously.",
                "metadata": {"chunk_id": "c1"},
                "rrf_score": 0.5,
            },
            {
                "content": "Bananas are yellow fruit rich in potassium.",
                "metadata": {"chunk_id": "c2"},
                "rrf_score": 0.6,
            },
        ]
        reranked = self.reranker.rerank(query, candidates, top_k=2)
        self.assertEqual(len(reranked), 2)
        self.assertIn("rerank_score", reranked[0])
        # CrossEncoder must rank candidate c1 above c2 despite lower RRF
        self.assertEqual(reranked[0]["metadata"]["chunk_id"], "c1")

    def test_prompt_builder(self):
        chunks = [
            {
                "content": "KnowledgeForge AI is modular.",
                "metadata": {"source": "doc.pdf", "page": 1},
            }
        ]
        context_str = build_context_block(chunks)
        self.assertIn("KnowledgeForge AI is modular.", context_str)

        prompt = format_final_prompt("What is KnowledgeForge AI?", context_str)
        self.assertIn("KnowledgeForge AI is modular.", prompt)
        self.assertIn("What is KnowledgeForge AI?", prompt)

    @patch("models.llm.OfflineGemmaSynthesizer.synthesize")
    def test_generator_end_to_end(self, mock_synthesize):
        mock_synthesize.return_value = "This is a mocked RAG response."
        chunks = [
            {
                "content": "Deepset Gemma 3 is our local LLM engine.",
                "metadata": {"chunk_id": "g_1", "source": "gemma.txt"},
            }
        ]
        embedded = self.embedding_service.generate_embeddings_for_chunks(chunks)
        self.vector_store.add_chunks(embedded)
        self.retriever.refresh_bm25_index()

        answer, sources = self.generator.generate_response(
            "What is Deepset Gemma 3?", top_k=1
        )
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 5)
        self.assertEqual(len(sources), 1)
        self.assertIn("confidence_estimate", sources[0])
        self.assertIsNotNone(self.generator.last_confidence)


if __name__ == "__main__":
    unittest.main()
