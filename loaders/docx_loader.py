"""
DOCX Loader extracting structured paragraphs from Word documents.
"""

from typing import List, Dict, Any
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)


class DOCXLoader:
    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)

    def load(self) -> List[Dict[str, Any]]:
        """Load text from DOCX and return structured blocks."""
        documents = []
        try:
            import docx

            doc = docx.Document(str(self.file_path))
            full_text = []
            for para in doc.paragraphs:
                if para.text.strip():
                    text = para.text.strip()
                    style_name = para.style.name if para.style else ""
                    if style_name.startswith("Heading"):
                        try:
                            level = int(style_name.split()[-1])
                            text = f"{'#' * min(level, 6)} {text}"
                        except ValueError:
                            text = f"# {text}"
                    full_text.append(text)

            # Combine paragraphs into cohesive document sections or whole document
            combined = "\n\n".join(full_text)
            if combined:
                documents.append(
                    {
                        "content": combined,
                        "metadata": {
                            "source": self.file_path.name,
                            "page": 1,  # Word documents don't have hard page boundaries without rendering
                            "file_type": "docx",
                        },
                    }
                )
        except Exception as e:
            logger.error(f"Failed to load DOCX {self.file_path.name}: {e}")
        return documents
