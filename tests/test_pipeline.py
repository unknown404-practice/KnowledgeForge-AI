"""
Unit tests for the KnowledgeForge AI RAG Pipeline.
"""

import unittest
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*NumPy version.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*sklearn.*")

from utils.config import AppConfig
from rag.chunking import SmartChunker


class TestKnowledgeForgePipeline(unittest.TestCase):
    def setUp(self):
        self.config = AppConfig()
        self.chunker = SmartChunker(chunk_size=200, chunk_overlap=20)

    def test_smart_chunking(self):
        sample_text = "KnowledgeForge AI is an advanced RAG platform. " * 20
        chunks = self.chunker.split_text(sample_text)
        self.assertTrue(len(chunks) > 0)
        self.assertTrue(all(len(c) <= 250 for c in chunks))


if __name__ == "__main__":
    unittest.main()
