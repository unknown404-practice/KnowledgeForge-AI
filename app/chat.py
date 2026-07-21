"""
Interactive Chat Interface with citations and confidence score inspection.
"""

import streamlit as st
from typing import List, Dict, Any
from rag.generator import RAGGenerator
from utils.helpers import format_confidence_score


def render_sources_accordion(sources: List[Dict[str, Any]]):
    """Render expandable accordion showing exact source metadata and confidence scores."""
    if not sources:
        return

    with st.expander(
        f"📚 Sources + Confidence Scores ({len(sources)} Chunks Retrieved)",
        expanded=False,
    ):
        for idx, item in enumerate(sources, 1):
            meta = item.get("metadata", {})
            source_file = meta.get("source", "Unknown Document")
            page = meta.get("page", 1)
            heading = meta.get("heading", "General Context")
            raw_s = item.get("rerank_score", item.get("rrf_score", 0.0))
            conf_pct = format_confidence_score(item.get("confidence_score", 0.85))

            st.markdown(
                f"""
            <div class="source-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <b>🔍 Source #{idx}: {source_file} (Page {page})</b>
                    <span class="confidence-badge">Confidence: {conf_pct}</span>
                </div>
                <div style="font-size: 0.8rem; color: #38bdf8; margin-top: 4px;">
                    <b>Section:</b> {heading}
                </div>
                <div style="font-size: 0.9rem; color: #cbd5e1; margin-top: 6px; margin-bottom: 0; white-space: pre-wrap; word-wrap: break-word;">
                    {item.get('content', '')}
                </div>
                <div style="font-size: 0.75rem; color: #64748b; margin-top: 6px;">
                    Chunk ID: <code>{meta.get('chunk_id', 'N/A')}</code> | Rerank Score: <code>{raw_s:.4f}</code>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )


def render_chat_interface(generator: RAGGenerator, top_k: int):
    """Render chat input and message history."""
    st.markdown("### 💬 Chat Interface")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I am KnowledgeForge AI. Upload your documents in the tab above and ask me anything about them with verified citations!",
                "sources": [],
            }
        ]

    # Display existing message history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources_accordion(message["sources"])

    # Handle user prompt input
    if prompt := st.chat_input("Ask a question based on uploaded documents..."):
        # Append user message
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "sources": []}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner(
                "🔍 Running Hybrid Retrieval + Cross-Encoder Reranking + Local Ollama Generation..."
            ):
                try:
                    try:
                        stream, sources = generator.stream_response(prompt, top_k=top_k)
                        full_response = st.write_stream(stream)
                    except (NotImplementedError, AttributeError):
                        full_response, sources = generator.generate_response(
                            prompt, top_k=top_k
                        )
                        st.markdown(full_response)
                except Exception as e:
                    error_msg = str(e)
                    if "offline" in error_msg.lower() or "ollama" in error_msg.lower():
                        st.warning("👋 **Welcome to KnowledgeForge AI!**\n\nIt looks like the local Ollama AI engine hasn't been started in the background yet.\n\n**To start the engine and test the app:**\n1. Run `docker-compose up -d` in your terminal to boot the backend.\n2. Or manually start Ollama on your host machine.\n\nOnce the backend is running, try asking your question again!")
                    else:
                        st.error(f"🚨 **Engine Error:** {error_msg}")
                    full_response = f"System Error: {error_msg}"
                    sources = []

            render_sources_accordion(sources)

            # Save assistant response to session state
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response, "sources": sources}
            )
