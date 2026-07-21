"""
FAISS Vector Store wrapper with bulletproof in-memory processing and robust JSON persistence.
Zero dependencies on SQLite, ChromaDB, or background telemetry.
"""

import faiss
import numpy as np
import json
import threading
from typing import List, Dict, Any
from pathlib import Path
from utils.config import CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)


def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure metadata only contains primitive types."""
    clean = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean


class ChromaVectorStore:
    """
    Renamed to ChromaVectorStore for API compatibility, but heavily rewritten internally
    to use pure FAISS + JSON persistence for maximum speed and zero OS lock issues.
    """

    _lock = threading.RLock()

    def __init__(
        self,
        persist_directory: Path = CONFIG.CHROMA_DB_DIR,
        collection_name: str = "knowledgeforge_chunks",
    ):
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.index_path = self.persist_directory / f"{self.collection_name}.faiss"
        self.meta_path = self.persist_directory / f"{self.collection_name}_meta.json"

        self.index = None
        self.documents = []  # List of {id, content, metadata}
        self.dimension = 384  # Default for all-MiniLM-L6-v2

        self._load_from_disk()

    def _load_from_disk(self):
        with self._lock:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            if self.index_path.exists() and self.meta_path.exists():
                try:
                    self.index = faiss.read_index(str(self.index_path))
                    with open(self.meta_path, "r", encoding="utf-8") as f:
                        self.documents = json.load(f)
                    logger.info(
                        f"Loaded FAISS index with {len(self.documents)} vectors."
                    )
                except Exception as e:
                    logger.error(f"Failed to load FAISS index: {e}. Starting fresh.")
                    self.reset_collection()
            else:
                logger.info("No existing FAISS index found. Starting fresh.")
                self.index = faiss.IndexFlatL2(self.dimension)
                self.documents = []

    def _save_to_disk(self):
        with self._lock:
            try:
                if self.index is not None:
                    faiss.write_index(self.index, str(self.index_path))
                with open(self.meta_path, "w", encoding="utf-8") as f:
                    json.dump(self.documents, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"Failed to save FAISS index: {e}")

    def reset_collection(self):
        """Wipe and recreate the FAISS index."""
        with self._lock:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.documents = []
            if self.index_path.exists():
                self.index_path.unlink()
            if self.meta_path.exists():
                self.meta_path.unlink()
            logger.info(f"Re-created fresh FAISS collection '{self.collection_name}'.")

    def add_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 500) -> int:
        """Add embedded chunks directly into the FAISS memory index."""
        if not chunks:
            return 0

        with self._lock:
            valid_embeddings = []
            valid_docs = []

            existing_ids = {doc["id"] for doc in self.documents}

            for i, chunk in enumerate(chunks):
                content = chunk.get("content", "").strip()
                if not content or not chunk.get("embedding"):
                    continue

                meta = _sanitize_metadata(chunk.get("metadata", {}))
                doc_id = meta.get("chunk_id", f"id_{len(self.documents) + i}")

                # Simple deduplication
                if doc_id in existing_ids:
                    continue

                meta["chunk_id"] = doc_id
                existing_ids.add(doc_id)

                valid_embeddings.append(chunk["embedding"])
                valid_docs.append({"id": doc_id, "content": content, "metadata": meta})

            if not valid_embeddings:
                return 0

            # Determine dimension if this is the first insert
            if self.index.ntotal == 0 and valid_embeddings:
                self.dimension = len(valid_embeddings[0])
                self.index = faiss.IndexFlatL2(self.dimension)

            # Convert to numpy and add to FAISS
            emb_array = np.array(valid_embeddings).astype("float32")
            self.index.add(emb_array)
            self.documents.extend(valid_docs)

            # Sync to disk
            self._save_to_disk()

            logger.info(
                f"Successfully upserted {len(valid_docs)} chunks into FAISS collection."
            )
            return len(valid_docs)

    def delete_document(self, filename: str) -> bool:
        """Completely remove all chunks associated with a specific file from FAISS."""
        with self._lock:
            if not self.documents or self.index is None or self.index.ntotal == 0:
                return False

            # Identify indices to keep
            keep_indices = []
            keep_docs = []
            for i, doc in enumerate(self.documents):
                source = doc.get("metadata", {}).get("source", "")
                if source != filename:
                    keep_indices.append(i)
                    keep_docs.append(doc)

            if len(keep_docs) == len(self.documents):
                logger.info(f"File {filename} not found in vector store.")
                return False

            # Reconstruct valid embeddings
            valid_embeddings = []
            import numpy as np

            for idx in keep_indices:
                emb = self.index.reconstruct(idx)
                valid_embeddings.append(emb)

            # Recreate FAISS index
            self.index = faiss.IndexFlatL2(self.dimension)
            self.documents = keep_docs

            if valid_embeddings:
                emb_array = np.array(valid_embeddings).astype("float32")
                self.index.add(emb_array)

            self._save_to_disk()
            logger.info(
                f"Successfully deleted {filename}. Remaining chunks: {len(self.documents)}"
            )
            return True

    def query(
        self, query_embeddings: List[List[float]], n_results: int = 5
    ) -> Dict[str, Any]:
        """Pass-through FAISS query."""
        if not query_embeddings or self.index.ntotal == 0:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        with self._lock:
            q_array = np.array(query_embeddings).astype("float32")
            k = min(n_results, self.index.ntotal)
            distances, indices = self.index.search(q_array, k)

            res_ids, res_docs, res_metas, res_dists = [], [], [], []

            for i in range(len(query_embeddings)):
                batch_ids, batch_docs, batch_metas, batch_dists = [], [], [], []
                for j in range(k):
                    idx = indices[i][j]
                    if idx != -1 and idx < len(self.documents):
                        doc = self.documents[idx]
                        batch_ids.append(doc["id"])
                        batch_docs.append(doc["content"])
                        batch_metas.append(doc["metadata"])
                        batch_dists.append(float(distances[i][j]))
                res_ids.append(batch_ids)
                res_docs.append(batch_docs)
                res_metas.append(batch_metas)
                res_dists.append(batch_dists)

            return {
                "ids": res_ids,
                "documents": res_docs,
                "metadatas": res_metas,
                "distances": res_dists,
            }

    def dense_search(
        self, query_embedding: List[float], top_k: int = CONFIG.DENSE_TOP_K
    ) -> List[Dict[str, Any]]:
        """Perform dense vector L2 search natively via FAISS."""
        if not query_embedding or self.index.ntotal == 0:
            return []

        results = self.query([query_embedding], n_results=top_k)

        candidates = []
        if results and results.get("ids") and len(results["ids"][0]) > 0:
            for idx, doc_id in enumerate(results["ids"][0]):
                dist = results["distances"][0][idx]
                # Convert L2 distance to a 0-1 similarity score loosely
                sim_score = max(0.0, min(1.0, 1.0 - (dist / 10.0)))

                candidates.append(
                    {
                        "id": doc_id,
                        "content": results["documents"][0][idx],
                        "metadata": results["metadatas"][0][idx],
                        "dense_distance": dist,
                        "dense_score": sim_score,
                    }
                )
        return candidates

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Retrieve all stored chunks for BM25 sparse keyword indexing."""
        with self._lock:
            return list(self.documents)
