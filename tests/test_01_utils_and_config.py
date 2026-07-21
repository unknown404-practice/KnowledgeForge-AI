"""
Unit tests for Step 1: Utils & Config layer (`utils/config.py`, `utils/logger.py`, `utils/helpers.py`).
Run with: `pytest tests/test_01_utils_and_config.py -v`
"""

import unittest
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
from utils.config import AppConfig
from utils.logger import get_logger
from utils.helpers import clean_text_string, format_confidence_score, truncate_snippet


class TestUtilsAndConfig(unittest.TestCase):
    def test_app_config_initialization(self):
        config = AppConfig()
        self.assertIsInstance(config.CHROMA_DB_DIR, Path)
        self.assertIsInstance(config.DATA_RAW_DIR, Path)
        self.assertIsInstance(config.DATA_PROCESSED_DIR, Path)
        self.assertEqual(config.EMBEDDING_MODEL_NAME, "all-MiniLM-L6-v2")
        self.assertGreater(config.DEFAULT_CHUNK_SIZE, 0)
        self.assertGreater(config.DEFAULT_CHUNK_OVERLAP, 0)

    def test_logger_creation(self):
        logger = get_logger("TestLogger")
        self.assertIsNotNone(logger)

    def test_clean_text_string(self):
        raw = "  Hello   World!\t\nThis is a   test.  "
        cleaned = clean_text_string(raw)
        self.assertEqual(cleaned, "Hello World!\nThis is a test.")

    def test_format_confidence_score(self):
        self.assertEqual(format_confidence_score(0.87654), "87.7%")
        self.assertEqual(format_confidence_score(1.0), "100.0%")
        self.assertEqual(format_confidence_score(-0.5), "0.0%")
        self.assertEqual(format_confidence_score(1.5), "100.0%")

    def test_truncate_snippet(self):
        text = "KnowledgeForge AI is an advanced RAG platform for local document chat."
        truncated = truncate_snippet(text, max_length=25)
        self.assertTrue(truncated.endswith("..."))
        self.assertLessEqual(len(truncated), 28)
        self.assertEqual(truncate_snippet("Short", max_length=50), "Short")


if __name__ == "__main__":
    unittest.main()
