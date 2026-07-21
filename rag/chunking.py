"""
Smart Text Cleaning and Recursive Chunking module.
"""

from typing import List, Dict, Any
from utils.helpers import clean_text_string
from utils.config import CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)


class SmartChunker:
    def __init__(
        self,
        chunk_size: int = CONFIG.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = CONFIG.DEFAULT_CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """Split text recursively by paragraph, sentence, and word boundaries."""
        cleaned = clean_text_string(text)
        if not cleaned:
            return []

        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            return splitter.split_text(cleaned)
        except ImportError:
            # Clean fallback custom recursive splitter
            chunks = []
            words = cleaned.split()
            current_chunk = []
            current_len = 0
            for word in words:
                word_len = len(word) + 1
                if current_len + word_len > self.chunk_size and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    # Overlap handling
                    overlap_words = current_chunk[-max(1, self.chunk_overlap // 5) :]
                    current_chunk = overlap_words + [word]
                    current_len = sum(len(w) + 1 for w in current_chunk)
                else:
                    current_chunk.append(word)
                    current_len += word_len
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            return chunks

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process document objects and emit fine-grained chunks with inherited metadata."""
        import re

        chunked_docs = []
        for doc in documents:
            content = doc.get("content", "")
            meta = doc.get("metadata", {})
            text_chunks = self.split_text(content)
            current_heading = meta.get("heading", "General Context")

            for i, chunk_text in enumerate(text_chunks):
                # Update current heading if a Markdown header is found in this chunk
                headings = re.findall(
                    r"^(#{1,6})\s+(.+)$", chunk_text, flags=re.MULTILINE
                )
                if headings:
                    current_heading = headings[-1][1].strip()

                chunk_meta = meta.copy()
                chunk_meta["chunk_index"] = i
                chunk_meta["document_id"] = meta.get("source", "doc")
                chunk_meta["chunk_id"] = (
                    f"{meta.get('source', 'doc')}_p{meta.get('page', 1)}_c{i}"
                )
                chunk_meta["page"] = meta.get("page", 1)
                chunk_meta["heading"] = current_heading

                chunked_docs.append({"content": chunk_text, "metadata": chunk_meta})
        logger.info(
            f"Generated {len(chunked_docs)} semantic chunks from {len(documents)} document blocks."
        )
        return chunked_docs
