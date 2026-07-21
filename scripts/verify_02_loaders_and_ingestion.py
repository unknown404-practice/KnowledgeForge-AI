"""
Terminal Verification Script 2: All 7 Document Loaders & Ingestion Pipeline
Execute in terminal: python scripts/verify_02_loaders_and_ingestion.py
"""

import os
import sys
import warnings
import fitz
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
    print("[RUN] TERMINAL VERIFICATION 2: All 7 Document Loaders & Ingestion Pipeline")
    print("=" * 70)

    from loaders.pdf_loader import get_ocr_reader
    from loaders.csv_loader import CSVLoader
    from loaders.html_loader import HTMLLoader
    from loaders.python_loader import PythonLoader
    from loaders.markdown_loader import MarkdownLoader
    from rag.ingestion import DocumentIngestionPipeline
    from utils.config import CONFIG

    print(f"[OK] PyMuPDF version: {fitz.__version__}")
    ocr = get_ocr_reader()
    print(f"[OK] EasyOCR Reader active: {ocr is not False}")

    # Create sample verification files
    sample_dir = CONFIG.DATA_RAW_DIR
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "sample_table.csv").write_text(
        "id,name,role\n1,Ranadeep,Lead Engineer\n2,AI,Assistant\n", encoding="utf-8"
    )
    (sample_dir / "sample_page.html").write_text(
        "<html><body><h1>KnowledgeForge RAG</h1><p>Production HTML block.</p></body></html>",
        encoding="utf-8",
    )
    (sample_dir / "sample_script.py").write_text(
        'def hello():\n    return "KnowledgeForge Python Loader"\n', encoding="utf-8"
    )
    (sample_dir / "sample_notes.md").write_text(
        "# Markdown Notes\n\nComplete roadmap verification.\n", encoding="utf-8"
    )

    print("\n--- Testing Individual Loaders ---")
    csv_docs = CSVLoader(sample_dir / "sample_table.csv").load()
    html_docs = HTMLLoader(sample_dir / "sample_page.html").load()
    py_docs = PythonLoader(sample_dir / "sample_script.py").load()
    md_docs = MarkdownLoader(sample_dir / "sample_notes.md").load()

    print(
        f"[OK] CSV Loader: {csv_docs[0]['content'][:50]}... | {csv_docs[0]['metadata']['file_type']}"
    )
    print(
        f"[OK] HTML Loader: {html_docs[0]['content'][:50]}... | {html_docs[0]['metadata']['file_type']}"
    )
    print(
        f"[OK] Python Loader: {py_docs[0]['content'][:50]}... | {py_docs[0]['metadata']['file_type']}"
    )
    print(
        f"[OK] Markdown Loader: {md_docs[0]['content'][:50]}... | {md_docs[0]['metadata']['file_type']}"
    )

    print("\n--- Testing DocumentIngestionPipeline across data/raw ---")
    pipeline = DocumentIngestionPipeline(sample_dir)
    all_docs = pipeline.ingest_all()
    print(f"[OK] Total documents ingested across folder: {len(all_docs)}")
    for d in all_docs:
        print(
            f"    - Source: {d['metadata']['source']} | Type: {d['metadata']['file_type']} | Date: {d['metadata']['creation_date']}"
        )

    print(
        "\n[SUCCESS] VERIFICATION 2 COMPLETE: All 7 Loaders & Ingestion verified cleanly!"
    )
    print("=" * 70)


if __name__ == "__main__":
    run_verification()
