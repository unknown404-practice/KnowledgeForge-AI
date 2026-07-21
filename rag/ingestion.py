"""
Document Ingestion pipeline routing file formats to specialized loaders (supporting all 7 roadmap formats).
"""

import datetime
from typing import List, Dict, Any
from pathlib import Path
from loaders.pdf_loader import PDFLoader
from loaders.docx_loader import DOCXLoader
from loaders.txt_loader import TXTLoader
from loaders.markdown_loader import MarkdownLoader
from loaders.csv_loader import CSVLoader
from loaders.html_loader import HTMLLoader
from loaders.python_loader import PythonLoader
from utils.logger import get_logger

logger = get_logger(__name__)


class DocumentIngestionPipeline:
    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)

    def ingest_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Ingest a single document and return clean document objects with complete creation metadata."""
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        logger.info(f"Ingesting file: {file_path.name} (Type: {ext})")

        if ext == ".pdf":
            loader = PDFLoader(file_path)
        elif ext == ".docx":
            loader = DOCXLoader(file_path)
        elif ext in [".md", ".markdown"]:
            loader = MarkdownLoader(file_path)
        elif ext == ".csv":
            loader = CSVLoader(file_path)
        elif ext in [".html", ".htm"]:
            loader = HTMLLoader(file_path)
        elif ext == ".py":
            loader = PythonLoader(file_path)
        elif ext in [".txt", ".log", ".json"]:
            loader = TXTLoader(file_path)
        else:
            logger.warning(
                f"Unsupported file format '{ext}' for {file_path.name}. Defaulting to TXTLoader."
            )
            loader = TXTLoader(file_path)

        documents = loader.load()

        # Ensure creation date and page metadata are attached to every document block
        try:
            creation_date = datetime.datetime.fromtimestamp(
                file_path.stat().st_ctime
            ).strftime("%Y-%m-%d")
        except Exception:
            creation_date = datetime.datetime.now().strftime("%Y-%m-%d")

        for d in documents:
            if "metadata" not in d:
                d["metadata"] = {}
            if "creation_date" not in d["metadata"]:
                d["metadata"]["creation_date"] = creation_date
            if "page" not in d["metadata"]:
                d["metadata"]["page"] = 1

        logger.info(f"Extracted {len(documents)} document blocks from {file_path.name}")
        return documents

    def ingest_all(self) -> List[Dict[str, Any]]:
        """Ingest all files present in the raw data directory."""
        all_docs = []
        if not self.raw_dir.exists():
            return all_docs
        for file_path in self.raw_dir.iterdir():
            if file_path.is_file() and not file_path.name.startswith("."):
                all_docs.extend(self.ingest_file(file_path))
        return all_docs
