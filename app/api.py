"""
FastAPI Backend Layer providing REST API endpoints for the KnowledgeForge AI RAG Pipeline.
Run with: `uvicorn app.api:app --port 8000 --reload`
"""

from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from utils.config import CONFIG
from utils.logger import get_logger
from rag.ingestion import DocumentIngestionPipeline
from rag.chunking import SmartChunker
from rag.embeddings import EmbeddingService
from rag.vector_store import ChromaVectorStore
from rag.retrieval import HybridRetriever
from rag.reranker import CrossEncoderReranker
from rag.generator import RAGGenerator
from models.llm import DeepsetGemmaLLM
import uuid

logger = get_logger(__name__)

app = FastAPI(
    title="KnowledgeForge AI • RAG REST API",
    description="High-performance backend API for Hybrid RAG (Dense + BM25), BGE CrossEncoder Reranking, and Deepset Gemma 3 generation.",
    version="2.0.0",
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_id = str(uuid.uuid4())
    logger.error("Unexpected API error [{}]: {}", error_id, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "error_id": error_id,
            "details": str(exc),
        },
    )


# Shared singletons
_vector_store = ChromaVectorStore(persist_directory=CONFIG.CHROMA_DB_DIR)
_embedding_service = EmbeddingService(CONFIG.EMBEDDING_MODEL_NAME)
_retriever = HybridRetriever(_vector_store, _embedding_service)
_reranker = CrossEncoderReranker()
_llm = DeepsetGemmaLLM(model_name="google/gemma-3-4b-it", use_native_weights=True)
_generator = RAGGenerator(_retriever, _reranker, _llm)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    llm_model: Optional[str] = "google/gemma-3-4b-it"


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Health check endpoint checking ChromaDB index and Deepset Gemma availability."""
    logger.info("Health check requested.")
    return {
        "status": "healthy",
        "chromadb_chunks": len(_vector_store.documents),
        "gemma_deepset_available": True,
        "native_weights_loaded": _llm.is_native_loaded(),
        "ollama_available": _llm.check_availability(),
    }


from pathlib import Path


@app.post("/upload", status_code=status.HTTP_201_CREATED)
def upload_documents(files: List[UploadFile] = File(...)):
    """Ingest uploaded files through PyMuPDF/EasyOCR, split into smart chunks, and index into ChromaDB + BM25."""
    logger.info(f"Uploading and processing {len(files)} files.")

    ALLOWED_EXTENSIONS = {
        ".pdf",
        ".txt",
        ".md",
        ".docx",
        ".csv",
        ".html",
        ".htm",
        ".py",
    }
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB limits

    saved_paths = []
    for file in files:
        # 1. Path Traversal Protection
        safe_filename = Path(file.filename).name
        if not safe_filename or safe_filename == "." or safe_filename == "..":
            raise HTTPException(status_code=400, detail="Invalid filename provided.")

        # 2. File Extension Validation
        ext = Path(safe_filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File extension {ext} not allowed. Allowed types: {ALLOWED_EXTENSIONS}",
            )

        # 3. File Size Validation (read into memory)
        file_bytes = file.file.read()
        if len(file_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, detail=f"File {safe_filename} exceeds 50MB limit."
            )

        dest = CONFIG.DATA_RAW_DIR / safe_filename
        with open(dest, "wb") as f:
            f.write(file_bytes)
        saved_paths.append(dest)

    pipeline = DocumentIngestionPipeline(CONFIG.DATA_RAW_DIR)
    raw_docs = []
    for path in saved_paths:
        raw_docs.extend(pipeline.ingest_file(path))

    chunker = SmartChunker()
    chunked_docs = chunker.chunk_documents(raw_docs)
    embedded_chunks = _embedding_service.generate_embeddings_for_chunks(chunked_docs)
    upsert_count = _vector_store.add_chunks(embedded_chunks)
    _retriever.refresh_bm25_index()

    return {
        "message": f"Successfully processed and indexed {upsert_count} chunks from {len(files)} files.",
        "upsert_count": upsert_count,
        "total_chunks_in_db": len(_vector_store.documents),
    }


@app.post("/query", response_model=QueryResponse)
def execute_query(request: QueryRequest):
    """Run Hybrid Retrieval (BM25 + Dense) -> BGE Reranker -> Local Ollama LLM."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if request.llm_model and request.llm_model != _llm.model_name:
        _llm.model_name = request.llm_model

    answer, sources = _generator.generate_response(request.query, top_k=request.top_k)
    return QueryResponse(answer=answer, sources=sources)


@app.post("/reset_db", status_code=status.HTTP_200_OK)
def reset_database():
    """Wipe and recreate the ChromaDB collection."""
    _vector_store.reset_collection()
    _retriever.refresh_bm25_index()
    return {"message": "ChromaDB collection wiped and re-created cleanly."}


@app.get("/documents", status_code=status.HTTP_200_OK)
def list_documents():
    """List all unique documents currently indexed in the vector store."""
    try:
        if not _vector_store or not hasattr(_vector_store, "documents"):
            return {"documents": []}

        chunks = _vector_store.documents
        files = {}
        for chunk in chunks:
            source = chunk.get("metadata", {}).get("source", "Unknown")
            if source not in files:
                files[source] = {"name": source, "chunks": 0}
            files[source]["chunks"] += 1

        documents = []
        for idx, (name, data) in enumerate(files.items()):
            ext = name.split(".")[-1] if "." in name else "txt"
            documents.append(
                {
                    "id": idx + 1,
                    "name": name,
                    "ext": ext,
                    "chunks": data["chunks"],
                    "status": "indexed",
                }
            )

        return {"documents": documents}
    except Exception as e:
        logger.error(f"Error fetching document list: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{filename}", status_code=status.HTTP_200_OK)
def delete_document(filename: str):
    """Securely and permanently delete a document from the vector store and disk."""
    try:
        logger.info(f"Deleting document {filename} via API...")
        if not _vector_store:
            raise HTTPException(status_code=500, detail="Database not initialized.")

        success = _vector_store.delete_document(filename)
        if not success:
            raise HTTPException(
                status_code=404, detail=f"Document {filename} not found in database."
            )

        _retriever.refresh_bm25_index()

        raw_file = Path(CONFIG.DATA_RAW_DIR) / filename
        if raw_file.exists():
            raw_file.unlink()

        return {"message": f"Document {filename} completely deleted."}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
