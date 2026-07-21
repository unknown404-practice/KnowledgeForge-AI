import sys
from pathlib import Path

# Add the project root to the path so we can import the modules
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from rag.vector_store import ChromaVectorStore as FAISSVectorStore
from rag.embeddings import EmbeddingService
from rag.retrieval import HybridRetriever
from rag.reranker import CrossEncoderReranker
from rag.generator import RAGGenerator
from models.llm import DeepsetGemmaLLM

print("All core KnowledgeForge AI modules loaded successfully!")

# 1. Load the extremely fast FAISS database
vector_store = FAISSVectorStore()

# 2. Load the embedding model (all-MiniLM-L6-v2 by default)
embedding_service = EmbeddingService()

# 3. Create the Hybrid Retriever (combines BM25 sparse search with FAISS dense search)
retriever = HybridRetriever(vector_store, embedding_service)

# 4. Load the Cross-Encoder for advanced contextual reranking
reranker = CrossEncoderReranker()

# 5. Boot up the local LLM Synthesizer (connects to local Ollama API)
# use_native_weights=False forces it to use Ollama via REST, which is faster for the notebook tutorial
llm = DeepsetGemmaLLM(model_name="gemma:2b", use_native_weights=False)

# 6. Assemble the final RAG Generator pipeline
generator = RAGGenerator(retriever, reranker, llm)

print("Engine initialized. Ready for queries.")

query = "What is KnowledgeForge AI?"
print(f"User Query: {query}\n")

answer, sources = generator.generate_response(query, top_k=5, use_reranker=True)

print("==================== AI RESPONSE ====================")
print(answer)
print("=====================================================\n")

print("--- Citations & Sources ---")
for i, source in enumerate(sources, 1):
    meta = source.get('metadata', {})
    print(f"Source {i}: {meta.get('source', 'Unknown')} (Page {meta.get('page', 1)})")
    
    conf = source.get('confidence_estimate', {}).get('score_percentage', 0) / 100.0
    if 'rerank_score' in source:
        conf = source['rerank_score']
    print(f"Confidence Score: {conf:.4f}")
    print("-" * 30)
