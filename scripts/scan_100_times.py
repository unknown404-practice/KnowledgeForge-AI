import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from utils.config import CONFIG
from utils.logger import get_logger
from rag.embeddings import EmbeddingService
from rag.vector_store import ChromaVectorStore
from rag.retrieval import HybridRetriever
from rag.reranker import CrossEncoderReranker
import time

logger = get_logger(__name__)


def run_scan():
    logger.info("Initializing 100x Hacker Scan of KnowledgeForge AI Pipeline...")

    # 1. Initialize services
    vector_store = ChromaVectorStore(persist_directory=CONFIG.CHROMA_DB_DIR)
    embedding_service = EmbeddingService(CONFIG.EMBEDDING_MODEL_NAME)
    retriever = HybridRetriever(vector_store, embedding_service)
    reranker = CrossEncoderReranker()

    logger.info(
        f"Loaded FAISS Vector Store. Total chunks: {len(vector_store.documents)}"
    )

    if len(vector_store.documents) == 0:
        logger.warning("Vector store is empty! Creating a dummy document to test...")
        dummy_chunk = {
            "content": "The knowledge base system relies on advanced FAISS and CrossEncoder reranking for hybrid search capabilities.",
            "metadata": {"source": "dummy_test.txt", "chunk_id": "dummy_1"},
        }
        dummy_chunk["embedding"] = embedding_service.embed_query(dummy_chunk["content"])
        vector_store.add_chunks([dummy_chunk])
        retriever.refresh_bm25_index()
        logger.info(f"Added dummy chunk. Total chunks: {len(vector_store.documents)}")

    success_count = 0
    fail_count = 0
    start_time = time.time()

    logger.info("==================================================")
    logger.info("       INITIATING 100x STRESS SCAN HACK         ")
    logger.info("==================================================")

    for i in range(1, 101):
        query = (
            f"Test query iteration {i}: What is FAISS and CrossEncoder hybrid search?"
        )
        try:
            candidates = retriever.retrieve(query, top_k=5)
            reranked_chunks = reranker.rerank(query, candidates, top_k=5)
            success_count += 1
            if i % 10 == 0:
                logger.info(
                    f"Scan {i}/100 Completed Successfully (Found {len(reranked_chunks)} chunks)"
                )
        except Exception as e:
            logger.error(f"Scan {i}/100 FAILED! Bug detected: {e}")
            fail_count += 1

    end_time = time.time()

    logger.info("==================================================")
    logger.info(f"       SCAN COMPLETE in {end_time - start_time:.2f} seconds")
    logger.info(f"       Success: {success_count}/100")
    logger.info(f"       Errors:  {fail_count}/100")
    logger.info("==================================================")

    if fail_count > 0:
        logger.error("HACKER REPORT: FLAWS DETECTED! Pipeline is NOT bug-free.")
        sys.exit(1)
    else:
        logger.info("HACKER REPORT: 100% ERROR-FREE! Architecture is ROCK SOLID.")
        sys.exit(0)


if __name__ == "__main__":
    run_scan()
