"""
Markdown Loader preserving header hierarchy and structure.
"""

from typing import List, Dict, Any
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)


class MarkdownLoader:
    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)

    def load(self) -> List[Dict[str, Any]]:
        """Load text from Markdown file and structure metadata."""
        documents = []
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                content = f.read()
            if content.strip():
                documents.append(
                    {
                        "content": content,
                        "metadata": {
                            "source": self.file_path.name,
                            "page": 1,
                            "file_type": "markdown",
                        },
                    }
                )
        except Exception as e:
            logger.error(f"Failed to load Markdown file {self.file_path.name}: {e}")
        return documents
