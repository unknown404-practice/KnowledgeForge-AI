"""
Unit tests for Step 6: FastAPI REST API Endpoints (`app/api.py`).
Run with: `pytest tests/test_06_api_endpoints.py -v`
"""

import unittest
import warnings
from fastapi.testclient import TestClient
from unittest.mock import patch

warnings.filterwarnings("ignore")

from app.api import app


class TestFastAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("chromadb_chunks", data)
        self.assertTrue(data["gemma_deepset_available"])

    def test_query_validation(self):
        # Empty query should return 400
        response = self.client.post("/query", json={"query": ""})
        self.assertEqual(response.status_code, 400)

    @patch("app.api._generator.generate_response")
    def test_query_execution(self, mock_generate):
        mock_generate.return_value = (
            "Mocked Answer",
            [{"content": "source snippet", "metadata": {"source": "fake"}}],
        )
        response = self.client.post(
            "/query", json={"query": "Hello KnowledgeForge AI", "top_k": 2}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("answer", data)
        self.assertIn("sources", data)
        self.assertIsInstance(data["sources"], list)


if __name__ == "__main__":
    unittest.main()
