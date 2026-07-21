"""
Unit tests for Step 4: Vector Store & Hybrid Retrieval (`rag/vector_store.py`, `rag/retrieval.py`).
Run with: `pytest tests/test_04_vectorstore_and_retrieval.py -v`
"""

import unittest
import warnings
import tempfile
from pathlib import Path

warnings.filterwarnings("ignore")

from rag.vector_store import ChromaVectorStore
from rag.embeddings import EmbeddingService
from rag.retrieval import HybridRetriever


class TestVectorStoreAndRetrieval(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_dir = Path(self.temp_dir.name) / "chroma_test"
        self.vector_store = ChromaVectorStore(
            persist_directory=self.db_dir, collection_name="test_col"
        )
        self.embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")
        self.retriever = HybridRetriever(self.vector_store, self.embedding_service)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_vectorstore_insert_and_query(self):
        chunks = [
            {
                "content": "Python is a versatile programming language for AI.",
                "metadata": {"chunk_id": "doc_py", "source": "py.txt"},
            },
            {
                "content": "ChromaDB is an open-source vector database for embeddings.",
                "metadata": {"chunk_id": "doc_db", "source": "db.txt"},
            },
        ]
        embedded = self.embedding_service.generate_embeddings_for_chunks(chunks)
        count = self.vector_store.add_chunks(embedded)
        self.assertEqual(count, 2)
        self.assertEqual(self.vector_store.index.ntotal, 2)

        # Query dense
        q_emb = self.embedding_service.embed_query("vector database")
        results = self.vector_store.dense_search(query_embedding=q_emb, top_k=1)
        self.assertIn("ChromaDB", results[0]["content"])

    def test_hybrid_retriever_rrf(self):
        chunks = [
            {
                "content": "Machine learning enables computers to learn from data.",
                "metadata": {"chunk_id": "ml_1", "source": "ml.txt"},
            },
            {
                "content": "BM25 is a sparse bag-of-words ranking algorithm.",
                "metadata": {"chunk_id": "bm_1", "source": "bm.txt"},
            },
        ]
        embedded = self.embedding_service.generate_embeddings_for_chunks(chunks)
        self.vector_store.add_chunks(embedded)

        # Test BM25 refresh and hybrid search
        self.retriever.refresh_bm25_index()
        retrieved = self.retriever.retrieve("sparse ranking algorithm BM25", top_k=2)
        self.assertTrue(len(retrieved) > 0)
        self.assertIn("BM25", retrieved[0]["content"])
        self.assertIn("rrf_score", retrieved[0])


if __name__ == "__main__":
    unittest.main()
