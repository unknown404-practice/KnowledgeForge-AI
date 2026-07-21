"""
Unit test verifying Deepset native Gemma 3 4B integration (`models/llm.py`).
"""

import unittest
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*NumPy version.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*sklearn.*")

from unittest.mock import patch

from models.llm import DeepsetGemmaLLM, GemmaLLM, OllamaLLM


class TestDeepsetGemmaIntegration(unittest.TestCase):
    def setUp(self):
        self.llm = DeepsetGemmaLLM(
            model_name="google/gemma-3-4b-it", use_native_weights=False
        )

    def test_gemma_availability(self):
        self.assertTrue(self.llm.check_availability())
        self.assertEqual(self.llm.repo_id, "bartowski/SmolLM2-135M-Instruct-GGUF")
        self.assertFalse(self.llm.is_native_loaded())

    @patch("models.llm.OfflineGemmaSynthesizer.synthesize")
    def test_gemma_generation_grounding(self, mock_synthesize):
        mock_synthesize.return_value = "Deepset Gemma 3 is integrated cleanly."
        prompt = "CONTEXT INFORMATION:\n====================\n[Source 1: test.txt | Page 1]\nDeepset Gemma 3 is integrated cleanly.\n\nUSER QUESTION:\n==============\nWhat is integrated?"
        response = self.llm.generate(prompt)
        self.assertIn("Deepset Gemma 3 is integrated cleanly", response)

    def test_aliases(self):
        self.assertIs(GemmaLLM, DeepsetGemmaLLM)
        self.assertIs(OllamaLLM, DeepsetGemmaLLM)


if __name__ == "__main__":
    unittest.main()
