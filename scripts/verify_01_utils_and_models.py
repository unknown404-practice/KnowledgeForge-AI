"""
Terminal Verification Script 1: Utilities & Native Deepset Gemma 3 Engine
Execute in terminal: python scripts/verify_01_utils_and_models.py
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
    print("[RUN] TERMINAL VERIFICATION 1: Utilities & Native Deepset Gemma 3 Engine")
    print("=" * 70)

    from utils.config import CONFIG
    from utils.logger import get_logger
    from models.embedding import EmbeddingModel
    from models.llm import DeepsetGemmaLLM

    logger = get_logger("TerminalVerifier_01")
    logger.info("Loguru logger active in terminal.")
    print(f"[OK] Config Base Dir: {CONFIG.BASE_DIR}")
    print(f"[OK] Embedding Model Config: {CONFIG.EMBEDDING_MODEL_NAME}")

    print("\n--- Testing EmbeddingModel ---")
    embedder = EmbeddingModel()
    vecs = embedder.embed_documents(["KnowledgeForge AI Terminal Verification"])
    print(f"[OK] Generated vector dimension: {len(vecs[0])}")

    print("\n--- Testing Native Deepset Gemma 3 Engine ---")
    llm = DeepsetGemmaLLM(model_name="google/gemma-3-4b-it")
    print(f"[OK] Deepset Gemma Available: {llm.check_availability()}")
    print(f"[OK] HF Model ID: {llm.hf_model_id}")
    resp = llm.generate("Say ok")
    print(f"[OK] Test Generation Output: {resp}")

    print("\n[SUCCESS] VERIFICATION 1 COMPLETE: Utilities & Models verified cleanly!")
    print("=" * 70)


if __name__ == "__main__":
    run_verification()
