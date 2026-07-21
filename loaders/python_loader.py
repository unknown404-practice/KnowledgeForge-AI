"""
Python File Loader extracting source code blocks cleanly with creation date and doc type metadata.
"""

import datetime
from pathlib import Path
from typing import List, Dict, Any
from utils.logger import get_logger

logger = get_logger(__name__)


class PythonLoader:
    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)

    def load(self) -> List[Dict[str, Any]]:
        """Load Python source file into structured RAG documents."""
        documents = []
        try:
            creation_date = datetime.datetime.fromtimestamp(
                self.file_path.stat().st_ctime
            ).strftime("%Y-%m-%d")
        except Exception:
            creation_date = datetime.datetime.now().strftime("%Y-%m-%d")

        try:
            with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
                code_text = f.read()
                if code_text.strip():
                    documents.append(
                        {
                            "content": code_text.strip(),
                            "metadata": {
                                "source": self.file_path.name,
                                "page": 1,
                                "creation_date": creation_date,
                                "file_type": "python",
                            },
                        }
                    )
            logger.info(f"Loaded Python source code document: {self.file_path.name}")
        except Exception as e:
            logger.error(f"Failed to read Python file {self.file_path.name}: {e}")
        return documents
