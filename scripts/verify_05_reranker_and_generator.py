"""
Terminal Verification Script 5: BGE Reranker, RAG Generator Confidence Estimation, & FastAPI REST Layer
Execute in terminal: python scripts/verify_05_reranker_and_generator.py
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
    print(
        "[RUN] TERMINAL VERIFICATION 5: BGE Reranker, Confidence Estimation, & FastAPI"
    )
    print("=" * 70)

    from rag.reranker import CrossEncoderReranker
    from rag.generator import RAGGenerator
    from models.llm import DeepsetGemmaLLM
    from app.api import app as fastapi_app, health_check

    print("\n--- Testing CrossEncoderReranker (BAAI/bge-reranker-base) ---")
    reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-base")
    candidates = [
        {
            "id": "1",
            "content": "Navigation test verification snippet for document extraction",
            "metadata": {"page": 1, "source": "test.pdf"},
        },
        {
            "id": "2",
            "content": "Completely unrelated recipe for cooking pasta with tomato sauce",
            "metadata": {"page": 2, "source": "test.pdf"},
        },
    ]
    top_chunks = reranker.rerank("Navigation test verification", candidates, top_k=2)
    print(
        f"[OK] Top reranked result: {top_chunks[0]['content']} | Score: {top_chunks[0]['rerank_score']:.4f}"
    )

    print("\n--- Testing RAGGenerator Confidence Estimation ---")
    generator = RAGGenerator(DeepsetGemmaLLM())
    conf_est = generator.calculate_confidence_estimate(top_chunks)
    print(
        f"[OK] Confidence Estimation: {conf_est['score_percentage']}% | Label: {conf_est['label']}"
    )
    print(f"    Details: {conf_est['details']}")

    print("\n--- Testing FastAPI REST Layer Interface ---")
    print(f"[OK] FastAPI App Title: {fastapi_app.title}")
    print(f"[OK] Health Check Endpoint output: {health_check()}")

    print(
        "\n[SUCCESS] VERIFICATION 5 COMPLETE: Reranker, Generator, & FastAPI verified cleanly!"
    )
    print("=" * 70)


if __name__ == "__main__":
    run_verification()
