"""
System Prompt templates and Context structuring utilities.
"""

from typing import List, Dict, Any


def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """Format top-k retrieved chunks into a clean context block with citations."""
    if not chunks:
        return "No relevant context found in the uploaded documents."

    formatted_parts = []
    for idx, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source_name = meta.get("source", "Document")
        page_num = meta.get("page", 1)
        content = chunk.get("content", "").strip()
        formatted_parts.append(
            f"[Source {idx}: {source_name} | Page {page_num}]\n{content}"
        )
    return "\n\n---\n\n".join(formatted_parts)


def get_rag_system_prompt() -> str:
    """Return the core system prompt enforcing precision and grounding."""
    return (
        "You are KnowledgeForge AI, an expert analytical AI coding and RAG assistant.\n"
        "Your role is to answer user queries accurately using the provided CONTEXT INFORMATION.\n"
        "The CONTEXT INFORMATION provided to you represents the exact contents of the documents the user has uploaded.\n"
        "Rules:\n"
        "1. Base your answer strictly on facts and information explicitly mentioned in the context.\n"
        "2. Do NOT say the context does not provide information if it is present in the context. Read the context carefully!\n"
        "3. If the answer isn't supported by the context at all, explicitly say 'I don't know' and state that the information is not present in the uploaded documents.\n"
        "4. Cite your sources clearly using inline square brackets like [Source 1: filename | Page X] when referencing specific points.\n"
        "5. Be concise, well-structured, and highly analytical. Always use Markdown lists, bold text, and formatting to structure your reply."
    )


def format_final_prompt(query: str, context_str: str) -> str:
    """Wrap context and query into a coherent user prompt block."""
    return (
        f"CONTEXT INFORMATION:\n"
        f"====================\n"
        f"{context_str}\n\n"
        f"USER QUESTION:\n"
        f"==============\n"
        f"{query}\n\n"
        f"Answer the question clearly based strictly on the context above:"
    )
