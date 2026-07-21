"""
UI components and styling for Streamlit Dashboard.
"""

import streamlit as st
from utils.config import CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)


def inject_custom_css():
    """Inject vibrant modern UI styling."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        .main-header {
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
            font-size: 2.5rem;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            color: #94a3b8;
            font-size: 1.1rem;
            margin-bottom: 1.5rem;
        }
        .source-card {
            background-color: rgba(30, 41, 59, 0.6);
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 12px;
            margin-top: 8px;
            margin-bottom: 8px;
        }
        .confidence-badge {
            background: linear-gradient(90deg, #10b981, #059669);
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )


def render_header():
    """Render top application header."""
    inject_custom_css()
    st.markdown(
        '<div class="main-header">🧠 KnowledgeForge AI</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-header">Local RAG Dashboard • Hybrid Search (BM25 + Dense) • Cross-Encoder Reranking • Ollama LLM</div>',
        unsafe_allow_html=True,
    )


def render_upload_section(
    ingestion_callback, chunking_callback, reset_callback=None
) -> bool:
    """Render drag-and-drop document upload interface."""
    st.markdown("### 📁 Upload Documents into Pipeline")
    st.markdown(
        "Upload **PDF**, **DOCX**, **TXT**, **Markdown**, **CSV**, **HTML**, or **Python** files to ingest into local ChromaDB with Hybrid indexing."
    )

    if reset_callback is not None:
        if st.button("🗑️ Wipe Database & Start Fresh", type="secondary"):
            reset_callback()
            st.success(
                "Database and memory cache wiped successfully! You can now start fresh."
            )

    uploaded_files = st.file_uploader(
        "Choose document files",
        type=["pdf", "docx", "txt", "md", "csv", "html", "htm", "py"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button(
            "🚀 Process & Ingest Documents", type="primary", use_container_width=True
        ):
            with st.status("⚙️ Running Document Pipeline...", expanded=True) as status:
                st.write("📁 Saving raw files...")
                saved_paths = []
                for file in uploaded_files:
                    dest_path = CONFIG.DATA_RAW_DIR / file.name
                    with open(dest_path, "wb") as f:
                        f.write(file.getbuffer())
                    saved_paths.append(dest_path)

                st.write("🧹 Ingesting & Cleaning Text...")
                raw_docs = ingestion_callback(saved_paths)

                st.write("✂️ Smart Chunking & Generating Dense Embeddings...")
                chunks_inserted = chunking_callback(raw_docs)

                st.session_state.doc_count = len(uploaded_files)
                st.session_state.chunk_count += chunks_inserted

                status.update(
                    label=f"✅ Successfully ingested {chunks_inserted} chunks!",
                    state="complete",
                    expanded=False,
                )
            st.success(
                f"🎉 Pipeline Complete! Indexed **{chunks_inserted}** high-precision chunks."
            )
            return True
    return False
