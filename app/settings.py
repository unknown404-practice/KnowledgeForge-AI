"""
Sidebar settings & configuration controls for Streamlit UI.
"""

import streamlit as st
from utils.config import CONFIG


def render_sidebar_settings() -> dict:
    """Render sidebar controls and return live configuration dict."""
    st.sidebar.markdown("## ⚙️ RAG Engine Settings")
    st.sidebar.markdown("Configure local models and retrieval hyper-parameters.")

    with st.sidebar.expander("🤖 Local Deepset Gemma Options", expanded=True):
        model_choice = st.selectbox(
            "Gemma Model Architecture",
            options=[
                "google/gemma-2-2b-it",
                "gemma:2b (Offline Analytical Engine)",
            ],
            index=0,
            help="Select official Google Deepset / HuggingFace model spec or local analytical engine.",
        )
        use_native_weights = st.checkbox(
            "Load Native 8GB PyTorch Weights (`use_native_weights`)",
            value=False,
            help="If checked, downloads/loads exact 8GB tensor weights via `transformers` `AutoModelForCausalLM`. If unchecked, runs instant zero-latency offline analytical engine.",
        )
        temperature = st.slider(
            "Temperature (Creativity)",
            min_value=0.0,
            max_value=1.0,
            value=float(CONFIG.TEMPERATURE),
            step=0.05,
        )

    with st.sidebar.expander("🔍 Retrieval & Reranking", expanded=True):
        top_k = st.slider(
            "Reranker Top-K Chunks",
            min_value=1,
            max_value=15,
            value=int(CONFIG.FINAL_RERANK_TOP_K),
            step=1,
            help="Number of high-precision chunks passed to the LLM.",
        )
        chunk_size = st.number_input(
            "Smart Chunk Size (Chars)",
            min_value=100,
            max_value=2000,
            value=int(CONFIG.DEFAULT_CHUNK_SIZE),
            step=50,
        )
        chunk_overlap = st.number_input(
            "Chunk Overlap (Chars)",
            min_value=0,
            max_value=500,
            value=int(CONFIG.DEFAULT_CHUNK_OVERLAP),
            step=10,
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Engine Telemetry")
    if "doc_count" not in st.session_state:
        st.session_state.doc_count = 0
    if "chunk_count" not in st.session_state:
        st.session_state.chunk_count = 0

    col1, col2 = st.sidebar.columns(2)
    col1.metric("Uploaded Docs", st.session_state.doc_count)
    col2.metric("Index Chunks", st.session_state.chunk_count)

    return {
        "llm_model": model_choice,
        "use_native_weights": use_native_weights,
        "temperature": temperature,
        "top_k": top_k,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
