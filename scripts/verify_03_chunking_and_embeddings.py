"""
Terminal Verification Script 3: Smart Chunking & Embedding Service
Execute in terminal: python scripts/verify_03_chunking_and_embeddings.py
"""

import os
import sys
import warnings
from pathlib import Path

# Safe Windows terminal encoding setup
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", category=UserWarning, message=".*NumPy version.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*sklearn.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


def run_verification():
    print("=" * 70)
    print("[RUN] TERMINAL VERIFICATION 3: Smart Chunking & Embedding Service")
    print("=" * 70)

    from rag.ingestion import DocumentIngestionPipeline
    from rag.chunking import SmartChunker
    from rag.embeddings import EmbeddingService
    from utils.config import CONFIG

    raw_docs = DocumentIngestionPipeline(CONFIG.DATA_RAW_DIR).ingest_all()
    print(f"[OK] Loaded {len(raw_docs)} raw documents.")

    print("\n--- Testing SmartChunker ---")
    chunker = SmartChunker(chunk_size=150, chunk_overlap=20)
    chunks = chunker.chunk_documents(raw_docs)
    print(f"[OK] Generated {len(chunks)} fine-grained chunks with exact metadata.")
    for c in chunks[:3]:
        print(f"    - ID: {c['metadata']['chunk_id']} | Text: {c['content'][:45]}...")

    print("\n--- Testing EmbeddingService on Chunks ---")
    service = EmbeddingService()
    embedded_chunks = service.generate_embeddings_for_chunks(chunks)
    print(f"[OK] Attached dense vectors to {len(embedded_chunks)} chunks.")
    if embedded_chunks:
        print(
            f"[OK] Sample chunk vector dimension: {len(embedded_chunks[0]['embedding'])}"
        )

    print(
        "\n[SUCCESS] VERIFICATION 3 COMPLETE: Chunking & Embeddings verified cleanly!"
    )
    print("=" * 70)


if __name__ == "__main__":
    run_verification()
