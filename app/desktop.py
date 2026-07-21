import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import webview
from utils.config import CONFIG
from utils.logger import get_logger
from rag.embeddings import EmbeddingService
from rag.vector_store import ChromaVectorStore
from rag.retrieval import HybridRetriever
from rag.reranker import CrossEncoderReranker
from rag.generator import RAGGenerator
from models.llm import DeepsetGemmaLLM

logger = get_logger(__name__)


class KnowledgeForgeAPI:
    """Backend API bridge for pywebview."""

    def __init__(self):
        # Initialize the same singletons as api.py
        self.vector_store = ChromaVectorStore(persist_directory=CONFIG.CHROMA_DB_DIR)
        self.embedding_service = EmbeddingService(CONFIG.EMBEDDING_MODEL_NAME)
        self.retriever = HybridRetriever(self.vector_store, self.embedding_service)
        self.reranker = CrossEncoderReranker()
        self.llm = DeepsetGemmaLLM(model_name="gemma:2b", use_native_weights=False)
        self.generator = RAGGenerator(self.retriever, self.reranker, self.llm)
        logger.info("Desktop API initialized.")

    def send_query(self, query: str, settings: dict = None):
        try:
            logger.info(f"Received query from UI: {query}")
            settings = settings or {}
            top_k = int(settings.get("topK", 5))
            use_reranker = settings.get("reranker", True)

            # Generate RAG response
            answer, sources_data = self.generator.generate_response(
                query, top_k=top_k, use_reranker=use_reranker
            )

            # Format sources for the frontend
            sources = []
            for src in sources_data:
                # The reranker output usually has confidence info
                confidence = (
                    src.get("confidence_estimate", {}).get("score_percentage", 0)
                    / 100.0
                )
                if "rerank_score" in src:
                    confidence = src[
                        "rerank_score"
                    ]  # fallback if confidence estimate wasn't formatted properly
                rrf_score = src.get("rrf_score", 0.0)
                sources.append(
                    {
                        "file": f"{src.get('metadata', {}).get('source', 'Unknown')} (Page {src.get('metadata', {}).get('page', 1)} | Chunk #{src.get('metadata', {}).get('chunk_index', 0)})",
                        "ce": f"{confidence:.4f}",
                        "rrf": f"{rrf_score:.4f}",
                        "heading": src.get("metadata", {}).get(
                            "heading", "General Context"
                        ),
                        "snippet": src.get("content", ""),
                    }
                )

            return {"success": True, "text": answer, "sources": sources}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing query: {error_msg}")
            if "offline" in error_msg.lower() or "ollama" in error_msg.lower():
                friendly_msg = "👋 **Welcome to KnowledgeForge AI!**<br><br>It looks like the local Ollama AI engine hasn't been started in the background yet.<br><br>**To start the engine and test the app:**<br>1. Run `docker-compose up -d` in your terminal to boot the backend.<br>2. Or manually start Ollama on your host machine.<br><br>Once the backend is running, try asking your question again!"
                # Return success=True so it renders the markdown in the chat window normally instead of an ugly system error
                return {"success": True, "text": friendly_msg, "sources": []}
            return {"success": False, "error": error_msg}

    def get_database_status(self):
        try:
            if not self.vector_store or not hasattr(self.vector_store, "documents"):
                return {"success": True, "documents": []}

            chunks = self.vector_store.documents
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
                        "added": "Synced from DB",
                        "content": "",
                    }
                )
            return {"success": True, "documents": documents}
        except Exception as e:
            logger.error(f"Error getting DB status: {e}")
            return {"success": False, "error": str(e)}

    def wipe_database(self):
        try:
            logger.info("Wiping ChromaDB collection...")
            if self.vector_store:
                self.vector_store.reset_collection()
                # Rebuild retriever/generator to clear any cached states
                self.retriever = HybridRetriever(
                    self.vector_store, self.embedding_service
                )
                self.generator = RAGGenerator(self.retriever, self.reranker, self.llm)
                logger.info("Database wiped and recreated successfully.")
            return {"success": True}
        except Exception as e:
            logger.error(f"Error wiping database: {e}")
            return {"success": False, "error": str(e)}

    def delete_document(self, filename: str):
        try:
            logger.info(f"Deleting document {filename} from database...")
            if not self.vector_store:
                return {"success": False, "error": "Database not initialized."}

            # Delete from FAISS
            success = self.vector_store.delete_document(filename)
            if not success:
                logger.warning(f"File {filename} was not found in FAISS.")

            # Rebuild BM25
            if self.retriever:
                self.retriever.refresh_bm25_index()

            # Physically delete raw file
            raw_file = Path(CONFIG.DATA_RAW_DIR) / filename
            if raw_file.exists():
                raw_file.unlink()

            return {"success": True}
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return {"success": False, "error": str(e)}

    def open_upload_dialog(self):
        try:
            import webview
            import shutil
            from rag.ingestion import DocumentIngestionPipeline
            from rag.chunking import SmartChunker

            # Note: the active window can be accessed from webview.windows
            if not webview.windows:
                return {"success": False, "error": "No active window."}

            window = webview.windows[0]
            file_types = (
                "Document Files (*.pdf;*.docx;*.txt;*.md;*.csv;*.py;*.html)",
                "All files (*.*)",
            )

            # This opens native dialog and returns tuple of paths
            result = window.create_file_dialog(
                webview.OPEN_DIALOG, allow_multiple=True, file_types=file_types
            )

            if not result:
                return {"success": True, "files": [], "chunks": 0}

            # Initialize ingestion pipeline and chunker
            pipeline = DocumentIngestionPipeline(Path(CONFIG.DATA_RAW_DIR))
            chunker = SmartChunker()

            all_chunks = []
            files_processed = []

            for file_path_str in result:
                file_path = Path(file_path_str)
                # Ensure the raw directory exists
                pipeline.raw_dir.mkdir(parents=True, exist_ok=True)

                # Copy file to raw data directory if it's not already there
                dest_path = pipeline.raw_dir / file_path.name
                if file_path.resolve() != dest_path.resolve():
                    shutil.copy2(file_path, dest_path)

                docs = pipeline.ingest_file(dest_path)
                chunks = chunker.chunk_documents(docs)
                all_chunks.extend(chunks)
                files_processed.append(file_path.name)

            if all_chunks:
                # Need to add embeddings first
                all_chunks = self.embedding_service.generate_embeddings_for_chunks(
                    all_chunks
                )
                self.vector_store.add_chunks(all_chunks)
                self.retriever.refresh_bm25_index()

            logger.info(
                f"Processed {len(files_processed)} files, generated {len(all_chunks)} chunks."
            )

            return {
                "success": True,
                "files": files_processed,
                "chunks": len(all_chunks),
            }

        except Exception as e:
            logger.error(f"Error uploading files: {e}")
            return {"success": False, "error": str(e)}


def main():
    # Load the 2.0 frontend HTML
    html_path = project_root / "knowledgeforge_frontend2.0.html"

    if not html_path.exists():
        logger.error(f"Frontend file not found: {html_path}")
        sys.exit(1)

    with open(html_path, "r", encoding="utf-8") as f:
        f.read()

    api = KnowledgeForgeAPI()

    # Create the standalone desktop window
    window = webview.create_window(
        "KnowledgeForge AI - Desktop Edition",
        url=str(html_path.resolve()),
        js_api=api,
        width=1280,
        height=850,
        min_size=(900, 600),
        background_color="#090d16",  # matches CRT bg-deep
    )

    # Start the desktop application
    webview.start(debug=False)


if __name__ == "__main__":
    main()
