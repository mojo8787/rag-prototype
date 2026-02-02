"""Streamlit UI for RAG prototype: ingest, Q&A, extraction."""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Streamlit Cloud: inject secrets into env so LangChain/OpenAI pick them up
try:
    if not os.getenv("OPENAI_API_KEY") and "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = str(st.secrets["OPENAI_API_KEY"])
except Exception:
    pass

from src.extraction import DefaultExtractionSchema, run_extraction
from src.ingest import ingest_documents
from src.qa import run_qa

st.set_page_config(
    page_title="RAG Prototype",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a polished UI
st.markdown("""
<style>
    /* Main container padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }
    /* Section cards */
    .stApp [data-testid="stVerticalBlock"] > div {
        border-radius: 12px;
    }
    /* Headers */
    h1 {
        font-weight: 600;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem !important;
    }
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .status-ok {
        background: rgba(34, 197, 94, 0.2);
        color: #22c55e;
    }
    .status-review {
        background: rgba(234, 179, 8, 0.2);
        color: #eab308;
    }
    /* Source cards */
    .source-card {
        background: var(--background-color);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    /* Confidence meter */
    .confidence-bar {
        height: 6px;
        border-radius: 3px;
        background: rgba(255,255,255,0.1);
        overflow: hidden;
        margin-top: 0.25rem;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 3px;
        background: linear-gradient(90deg, #22c55e, #4ade80);
        transition: width 0.3s ease;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(30,30,35,0.98) 0%, rgba(20,20,25,0.98) 100%);
    }
    [data-testid="stSidebar"] .stRadio > label {
        font-weight: 500;
    }
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.03);
        border: 1px dashed rgba(255,255,255,0.15);
        border-radius: 12px;
        padding: 1.5rem;
    }
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        padding: 0.5rem 1.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Session state
if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None
if "llm" not in st.session_state:
    st.session_state["llm"] = None


def get_llm():
    """Lazy init LLM (OpenAI). Reads OPENAI_API_KEY from env."""
    if st.session_state["llm"] is None:
        from langchain_openai import ChatOpenAI
        st.session_state["llm"] = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return st.session_state["llm"]


def main():
    # Sidebar
    with st.sidebar:
        st.markdown("### 📄 RAG Prototype")
        st.markdown("---")

        api_key = os.getenv("OPENAI_API_KEY", "") or getattr(st.secrets, "OPENAI_API_KEY", "") or ""
        api_key_set = bool(str(api_key).strip())

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**API Key**")
        with col2:
            if api_key_set:
                st.success("✓ Set", icon="✅")
            else:
                st.error("Not set", icon="⚠️")

        st.caption("Add OPENAI_API_KEY to .env or Streamlit secrets")
        st.markdown("---")

        st.markdown("**Navigation**")
        page = st.radio(
            "Page",
            ["Ingest", "Q&A", "Extraction"],
            label_visibility="collapsed",
            format_func=lambda x: {"Ingest": "📥 Ingest", "Q&A": "💬 Q&A", "Extraction": "📋 Extraction"}[x],
        )

        st.markdown("---")
        vs = st.session_state.get("vector_store")
        if vs is not None:
            st.success("Documents loaded", icon="📚")
        else:
            st.info("No documents yet", icon="📭")

    if page == "Ingest":
        render_ingest()
    elif page == "Q&A":
        render_qa()
    else:
        render_extraction()


def render_ingest():
    st.markdown("## 📥 Ingest Documents")
    st.markdown("Upload `.txt` or `.pdf` files. They will be chunked, embedded, and stored for Q&A and extraction.")
    st.markdown("")

    uploaded = st.file_uploader(
        "Upload files",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        help="Drag and drop or click to browse. Max 200MB per file.",
    )

    if not uploaded:
        with st.container():
            st.info("👆 Upload one or more files to get started. Supported: `.txt`, `.pdf`")
        return

    # Show uploaded files
    with st.expander(f"📎 {len(uploaded)} file(s) ready", expanded=True):
        for f in uploaded:
            st.caption(f"• {f.name} ({f.size / 1024:.1f} KB)")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        ingest_btn = st.button("🚀 Ingest", type="primary", use_container_width=True)

    if ingest_btn:
        progress_bar = st.progress(0.0, text="Preparing…")
        status = st.empty()
        try:
            paths = []
            for f in uploaded:
                suffix = Path(f.name).suffix.lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(f.read())
                    paths.append(tmp.name)

            def on_progress(msg: str, p: float) -> None:
                progress_bar.progress(min(1.0, max(0.0, p)), text=msg)
                status.caption(msg)
                time.sleep(0.05)

            display_names = [f.name for f in uploaded]
            vector_store = ingest_documents(paths, progress_callback=on_progress, file_display_names=display_names)
            st.session_state["vector_store"] = vector_store
            for p in paths:
                try:
                    Path(p).unlink(missing_ok=True)
                except Exception:
                    pass
            progress_bar.progress(1.0, text="Done.")
            status.empty()
            st.balloons()
            st.success(f"✅ Ingested {len(uploaded)} file(s). You can now use **Q&A** and **Extraction**.")
        except Exception as e:
            progress_bar.empty()
            status.empty()
            st.error(str(e))


def render_qa():
    st.markdown("## 💬 Document Q&A")
    st.markdown("Ask questions about your ingested documents. Answers are grounded in retrieved chunks.")
    st.markdown("")

    vs = st.session_state.get("vector_store")
    if vs is None:
        st.warning("📭 No documents loaded. Go to **Ingest** and upload files first.")
        return

    question = st.text_area(
        "Your question",
        placeholder="e.g. What is the main topic? What are the key dates?",
        height=80,
        label_visibility="collapsed",
    )

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        run_btn = st.button("🔍 Run Q&A", type="primary", use_container_width=True)

    if question and run_btn:
        with st.spinner("Retrieving and generating..."):
            try:
                llm = get_llm()
                result = run_qa(question, vs, llm)

                # Answer section
                st.markdown("### Answer")
                st.markdown(result["answer"])

                # Confidence
                conf = result.get("confidence")
                if conf is not None:
                    pct = int(conf * 100)
                    st.caption(f"Confidence: {pct}%")
                    st.markdown(
                        f'<div class="confidence-bar"><div class="confidence-fill" style="width:{pct}%"></div></div>',
                        unsafe_allow_html=True,
                    )

                # Sources
                st.markdown("### Sources")
                for i, doc in enumerate(result["source_chunks"], 1):
                    source = doc.metadata.get("source", "unknown")
                    with st.expander(f"Chunk {i} — {source}"):
                        content = doc.page_content[:600] + ("..." if len(doc.page_content) > 600 else "")
                        st.text(content)

                # Review status
                st.markdown("### Review status")
                needs = result["needs_review"]
                badge = "status-ok" if not needs else "status-review"
                label = "OK" if not needs else "Needs review"
                st.markdown(f'<span class="status-badge {badge}">{label}</span>', unsafe_allow_html=True)
                if needs:
                    st.caption(f"Reason: {result['review_reason']}")

            except Exception as e:
                st.error(str(e))


def render_extraction():
    st.markdown("## 📋 Document Extraction")
    st.markdown("Extract structured fields (dates, parties, amounts, summary) from your documents.")
    st.markdown("")

    vs = st.session_state.get("vector_store")
    if vs is None:
        st.warning("📭 No documents loaded. Go to **Ingest** and upload files first.")
        return

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        run_btn = st.button("📤 Run Extraction", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Retrieving and extracting..."):
            try:
                llm = get_llm()
                result = run_extraction(vs, DefaultExtractionSchema, llm)

                # Extracted record
                st.markdown("### Extracted record")
                st.json(result["record"])

                # Warnings
                if result.get("uncertain_fields"):
                    st.warning(f"Uncertain fields: {', '.join(result['uncertain_fields'])}")
                if result.get("validation_errors"):
                    st.warning(f"Validation: {'; '.join(result['validation_errors'][:3])}")

                # Review status
                st.markdown("### Review status")
                needs = result["needs_review"]
                badge = "status-ok" if not needs else "status-review"
                label = "OK" if not needs else "Needs review"
                st.markdown(f'<span class="status-badge {badge}">{label}</span>', unsafe_allow_html=True)
                if needs:
                    st.caption(f"Reason: {result['review_reason']}")

            except Exception as e:
                st.error(str(e))


if __name__ == "__main__":
    main()
