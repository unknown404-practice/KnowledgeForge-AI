"""
Unit tests for Step 2: Document Loaders (`loaders/txt_loader.py`, `loaders/markdown_loader.py`, `loaders/docx_loader.py`, `loaders/pdf_loader.py`).
Run with: `pytest tests/test_02_loaders.py -v`
"""

import unittest
import warnings
import tempfile
from pathlib import Path

warnings.filterwarnings("ignore")

from loaders.txt_loader import TXTLoader
from loaders.markdown_loader import MarkdownLoader
from loaders.docx_loader import DOCXLoader
import docx


class TestLoaders(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_txt_loader(self):
        txt_file = self.temp_path / "sample.txt"
        txt_file.write_text(
            "Hello KnowledgeForge AI!\nThis is a plain text verification test.",
            encoding="utf-8",
        )

        loader = TXTLoader(str(txt_file))
        docs = loader.load()
        self.assertEqual(len(docs), 1)
        self.assertIn("Hello KnowledgeForge AI!", docs[0]["content"])
        self.assertEqual(docs[0]["metadata"]["file_type"], "txt")

    def test_markdown_loader(self):
        md_file = self.temp_path / "sample.md"
        md_content = (
            "# Section 1\nHello from Markdown.\n## Section 2\nMore details here."
        )
        md_file.write_text(md_content, encoding="utf-8")

        loader = MarkdownLoader(str(md_file))
        docs = loader.load()
        self.assertTrue(len(docs) >= 1)
        self.assertEqual(docs[0]["metadata"]["file_type"].lower(), "markdown")

    def test_docx_loader(self):
        docx_file = self.temp_path / "sample.docx"
        doc = docx.Document()
        doc.add_heading("Docx Title", level=1)
        doc.add_paragraph("This is a test paragraph inside docx.")
        doc.save(str(docx_file))

        loader = DOCXLoader(str(docx_file))
        docs = loader.load()
        self.assertEqual(len(docs), 1)
        self.assertIn("Docx Title", docs[0]["content"])
        self.assertIn("test paragraph inside docx", docs[0]["content"])
        self.assertEqual(docs[0]["metadata"]["file_type"].lower(), "docx")


if __name__ == "__main__":
    unittest.main()
