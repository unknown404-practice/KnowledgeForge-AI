"""
HTML Loader extracting visible text and section headings using BeautifulSoup with complete metadata.
"""

import datetime
from pathlib import Path
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from utils.logger import get_logger

logger = get_logger(__name__)


class HTMLLoader:
    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)

    def load(self) -> List[Dict[str, Any]]:
        """Extract clean text from HTML file, stripping script/style tags."""
        documents = []
        try:
            creation_date = datetime.datetime.fromtimestamp(
                self.file_path.stat().st_ctime
            ).strftime("%Y-%m-%d")
        except Exception:
            creation_date = datetime.datetime.now().strftime("%Y-%m-%d")

        try:
            with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
                soup = BeautifulSoup(f.read(), "html.parser")

                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer"]):
                    script.decompose()

                # Convert HTML headings to Markdown headings for the SmartChunker
                for i in range(1, 7):
                    for tag in soup.find_all(f"h{i}"):
                        tag.insert_before(f"{'#' * i} ")
                        tag.insert_after("\n\n")

                text = soup.get_text(separator="\n", strip=True)
                if text:
                    documents.append(
                        {
                            "content": text,
                            "metadata": {
                                "source": self.file_path.name,
                                "page": 1,
                                "creation_date": creation_date,
                                "file_type": "html",
                            },
                        }
                    )
            logger.info(f"Loaded clean HTML document: {self.file_path.name}")
        except Exception as e:
            logger.error(f"Failed to read HTML {self.file_path.name}: {e}")
        return documents
