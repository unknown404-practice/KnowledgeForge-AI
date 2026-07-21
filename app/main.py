"""
KnowledgeForge AI Application Entrypoint.
Run using: `streamlit run app/main.py`
"""

import sys
from pathlib import Path
import streamlit as st

# Add project root to sys.path so modules can import seamlessly
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from utils.config import CONFIG
from utils.logger import get_logger
from app.settings import render_sidebar_settings
from app.ui import render_header, render_upload_section
from app.chat import render_chat_interface
from rag.ingestion import DocumentIngestionPipeline
from rag.chunking import SmartChunker
from rag.embeddings import EmbeddingService
from rag.vector_store import ChromaVectorStore
from rag.retrieval import HybridRetriever
from rag.reranker import CrossEncoderReranker
from rag.generator import RAGGenerator
from models.llm import DeepsetGemmaLLM

logger = get_logger(__name__)

# Set Streamlit page config
st.set_page_config(
    page_title="KnowledgeForge AI • Local RAG Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Initialize singletons in session_state for speed & memory optimization
@st.cache_resource
def get_rag_engine_components():
    """Cache and return core RAG backend components."""
    logger.info("Initializing core RAG engine singletons...")
    vector_store = ChromaVectorStore(persist_directory=CONFIG.CHROMA_DB_DIR)
    embedding_service = EmbeddingService(CONFIG.EMBEDDING_MODEL_NAME)
    retriever = HybridRetriever(vector_store, embedding_service)
    reranker = CrossEncoderReranker()
    return vector_store, embedding_service, retriever, reranker


def main():
    try:
        render_header()
    
        # Sidebar settings
        settings = render_sidebar_settings()
        CONFIG.DEFAULT_CHUNK_SIZE = settings["chunk_size"]
        CONFIG.DEFAULT_CHUNK_OVERLAP = settings["chunk_overlap"]
        CONFIG.TEMPERATURE = settings["temperature"]
    
        # Initialize backend services
        vector_store, embedding_service, retriever, reranker = get_rag_engine_components()
        llm = DeepsetGemmaLLM(
            model_name=settings["llm_model"],
            use_native_weights=settings.get("use_native_weights", False),
        )
        generator = RAGGenerator(retriever, reranker, llm)
    except Exception as e:
        st.error(f"🚨 **Engine Initialization Failed:** {str(e)}")
        st.warning("Please ensure you have an active internet connection on the first run to download the required models, and that the database isn't locked by another process.")
        return

    # Ingestion callbacks for UI uploader
    def handle_ingestion(paths: list) -> list:
        pipeline = DocumentIngestionPipeline(CONFIG.DATA_RAW_DIR)
        all_docs = []
        for path in paths:
            all_docs.extend(pipeline.ingest_file(path))
        return all_docs

    def handle_chunking_and_indexing(raw_docs: list) -> int:
        chunker = SmartChunker(
            chunk_size=settings["chunk_size"], chunk_overlap=settings["chunk_overlap"]
        )
        chunked_docs = chunker.chunk_documents(raw_docs)
        embedded_chunks = embedding_service.generate_embeddings_for_chunks(chunked_docs)
        count = vector_store.add_chunks(embedded_chunks)
        # Refresh BM25 index after new chunks added
        retriever.refresh_bm25_index()
        return count

    def handle_reset_database():
        vector_store.reset_collection()
        retriever.refresh_bm25_index()
        st.session_state.doc_count = 0
        st.session_state.chunk_count = 0
        import shutil

        if CONFIG.DATA_RAW_DIR.exists():
            shutil.rmtree(CONFIG.DATA_RAW_DIR)
            CONFIG.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        return True

    # Navigation Tabs
    tab_chat, tab_upload = st.tabs(["💬 Chat Interface", "📁 Upload Documents"])

    with tab_upload:
        render_upload_section(
            ingestion_callback=handle_ingestion,
            chunking_callback=handle_chunking_and_indexing,
            reset_callback=handle_reset_database,
        )

    with tab_chat:
        render_chat_interface(generator=generator, top_k=settings["top_k"])


if __name__ == "__main__":
    main()
