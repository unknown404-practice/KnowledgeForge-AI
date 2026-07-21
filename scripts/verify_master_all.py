"""
Terminal Master Verification Script: All 5 Architecture Layers
Execute in terminal: python scripts/verify_master_all.py
"""

import os
import sys
import time
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


def run_all_verifications():
    start_time = time.time()
    print("\n" + "=" * 70)
    print("[MASTER] KNOWLEDGEFORGE AI -- TERMINAL VERIFICATION SUITE (v2.0)")
    print("=" * 70)
    print(f"Project Root: {root_dir}\n")

    from scripts import (
        verify_01_utils_and_models,
        verify_02_loaders_and_ingestion,
        verify_03_chunking_and_embeddings,
        verify_04_vectorstore_and_hybrid_retrieval,
        verify_05_reranker_and_generator,
    )

    verify_01_utils_and_models.run_verification()
    print("\n")
    verify_02_loaders_and_ingestion.run_verification()
    print("\n")
    verify_03_chunking_and_embeddings.run_verification()
    print("\n")
    verify_04_vectorstore_and_hybrid_retrieval.run_verification()
    print("\n")
    verify_05_reranker_and_generator.run_verification()
    print("\n" + "=" * 70)
    elapsed = time.time() - start_time
    print(
        f"[SUCCESS] MASTER TERMINAL SUITE COMPLETE: All 5 layers verified cleanly in {elapsed:.2f}s!"
    )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_verifications()
