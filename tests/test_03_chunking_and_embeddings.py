"""
Unit tests for Step 3: Chunking & Embeddings (`rag/chunking.py`, `rag/embeddings.py`).
Run with: `pytest tests/test_03_chunking_and_embeddings.py -v`
"""

import unittest
import warnings

warnings.filterwarnings("ignore")

from rag.chunking import SmartChunker
from rag.embeddings import EmbeddingService


class TestChunkingAndEmbeddings(unittest.TestCase):
    def setUp(self):
        self.chunker = SmartChunker(chunk_size=150, chunk_overlap=30)
        self.embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")

    def test_smart_chunker_structure(self):
        raw_doc = {
            "content": "# Intro\nKnowledgeForge AI is modular.\n\n## Deep Dive\n"
            + ("We use recursive character splitting. " * 15),
            "metadata": {"source": "test.md", "page": 1, "document_type": "Markdown"},
        }
        chunks = self.chunker.chunk_documents([raw_doc])
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertIn("content", c)
            self.assertIn("metadata", c)
            self.assertIn("chunk_id", c["metadata"])
            self.assertIn("heading", c["metadata"])
            self.assertLessEqual(
                len(c["content"]), 220
            )  # Allow small buffer over chunk_size

    def test_embedding_service_generation(self):
        test_chunks = [
            {"content": "Chunk one about embeddings.", "metadata": {"chunk_id": "c1"}},
            {
                "content": "Chunk two about dense vector representations.",
                "metadata": {"chunk_id": "c2"},
            },
        ]
        embedded = self.embedding_service.generate_embeddings_for_chunks(test_chunks)
        self.assertEqual(len(embedded), 2)
        self.assertIn("embedding", embedded[0])
        self.assertIsInstance(embedded[0]["embedding"], list)
        self.assertEqual(
            len(embedded[0]["embedding"]), 384
        )  # all-MiniLM-L6-v2 dim is 384


if __name__ == "__main__":
    unittest.main()
