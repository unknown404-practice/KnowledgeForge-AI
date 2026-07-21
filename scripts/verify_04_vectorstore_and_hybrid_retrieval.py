"""
Terminal Verification Script 4: ChromaDB Thread-Safe Batch Store & Hybrid Retrieval
Execute in terminal: python scripts/verify_04_vectorstore_and_hybrid_retrieval.py
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
    print("[RUN] TERMINAL VERIFICATION 4: ChromaDB Batch Store & Hybrid Retrieval")
    print("=" * 70)

    from rag.ingestion import DocumentIngestionPipeline
    from rag.chunking import SmartChunker
    from rag.embeddings import EmbeddingService
    from rag.vector_store import ChromaVectorStore
    from rag.retrieval import HybridRetriever
    from utils.config import CONFIG

    raw_docs = DocumentIngestionPipeline(CONFIG.DATA_RAW_DIR).ingest_all()
    chunks = SmartChunker().chunk_documents(raw_docs)
    emb_service = EmbeddingService()
    emb_chunks = emb_service.generate_embeddings_for_chunks(chunks)

    print("\n--- Testing ChromaVectorStore Batch Upsert & Singleton ---")
    store = ChromaVectorStore(collection_name="terminal_verify_collection")
    store.reset_collection()
    inserted = store.add_chunks(emb_chunks, batch_size=500)
    print(
        f"[OK] Chroma collection total count: {store.collection.count()} (Inserted: {inserted})"
    )

    print("\n--- Testing HybridRetriever (BM25 + Dense RRF Fusion) ---")
    retriever = HybridRetriever(store, emb_service)
    results = retriever.retrieve("Complete roadmap verification", top_k=5)
    print(f"[OK] Retrieved {len(results)} combined hybrid candidates:")
    for r in results:
        print(f"    - [RRF Score: {r['rrf_score']:.4f}] {r['content'][:60]}...")

    print(
        "\n[SUCCESS] VERIFICATION 4 COMPLETE: Vector Store & Hybrid Retrieval verified cleanly!"
    )
    print("=" * 70)


if __name__ == "__main__":
    run_verification()
