"""
CSV Loader extracting rows and tabular data cleanly into documents with creation date and file type metadata.
"""

import csv
import datetime
from pathlib import Path
from typing import List, Dict, Any
from utils.logger import get_logger

logger = get_logger(__name__)


class CSVLoader:
    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)

    def load(self) -> List[Dict[str, Any]]:
        """Load rows/tables from CSV file into clean document dictionaries."""
        documents = []
        try:
            creation_date = datetime.datetime.fromtimestamp(
                self.file_path.stat().st_ctime
            ).strftime("%Y-%m-%d")
        except Exception:
            creation_date = datetime.datetime.now().strftime("%Y-%m-%d")

        try:
            with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                rows = []
                for idx, row in enumerate(reader):
                    row_text = ", ".join([f"{k}: {v}" for k, v in row.items() if v])
                    if row_text.strip():
                        rows.append(row_text)

                # Batch rows into logical chunks/pages if CSV is very large (every 50 rows = 1 logical page)
                batch_size = 50
                for i in range(0, max(1, len(rows)), batch_size):
                    batch = rows[i : i + batch_size]
                    content = "\n".join(batch)
                    if content.strip():
                        documents.append(
                            {
                                "content": content.strip(),
                                "metadata": {
                                    "source": self.file_path.name,
                                    "page": (i // batch_size) + 1,
                                    "creation_date": creation_date,
                                    "file_type": "csv",
                                },
                            }
                        )
            logger.info(
                f"Loaded {len(documents)} logical sections from CSV: {self.file_path.name}"
            )
        except Exception as e:
            logger.error(f"Failed to read CSV {self.file_path.name}: {e}")
        return documents
