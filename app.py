"""Streamlit UI for RAG prototype: ingest, Q&A, extraction."""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from src.extraction import DefaultExtractionSchema, run_extraction
from src.ingest import ingest_documents
from src.qa import run_qa

st.set_page_config(page_title="RAG Prototype", layout="wide")

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
    page = st.sidebar.radio(
        "Page",
        ["Ingest", "Q&A", "Extraction"],
        label_visibility="collapsed",
    )

    if page == "Ingest":
        render_ingest()
    elif page == "Q&A":
        render_qa()
    else:
        render_extraction()


def render_ingest():
    st.title("Ingest documents")
    st.caption("Upload .txt or .pdf files. They will be chunked, embedded, and stored for Q&A and extraction.")

    uploaded = st.file_uploader("Upload files", type=["txt", "pdf"], accept_multiple_files=True)
    if not uploaded:
        st.info("Upload one or more .txt or .pdf files to get started.")
        return

    if st.button("Ingest"):
        with st.spinner("Chunking and embedding..."):
            try:
                paths = []
                for f in uploaded:
                    suffix = Path(f.name).suffix.lower()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(f.read())
                        paths.append(tmp.name)
                vector_store = ingest_documents(paths)
                for p in paths:
                    try:
                        Path(p).unlink(missing_ok=True)
                    except Exception:
                        pass
                st.session_state["vector_store"] = vector_store
                st.success(f"Ingested {len(uploaded)} file(s). You can now use Q&A and Extraction.")
            except Exception as e:
                st.error(str(e))


def render_qa():
    st.title("Document Q&A")
    st.caption("Ask a question. Answer is based on ingested chunks; 'Needs human review' is shown when the gate triggers.")

    vs = st.session_state.get("vector_store")
    if vs is None:
        st.warning("Ingest documents first (Ingest page).")
        return

    question = st.text_input("Question", placeholder="e.g. What is the main topic?")
    if not question:
        return

    if st.button("Run Q&A"):
        with st.spinner("Retrieving and generating..."):
            try:
                llm = get_llm()
                result = run_qa(question, vs, llm)
                st.subheader("Answer")
                st.write(result["answer"])
                if result.get("confidence") is not None:
                    st.caption(f"Confidence: {result['confidence']:.2f}")
                st.subheader("Sources")
                for i, doc in enumerate(result["source_chunks"], 1):
                    with st.expander(f"Chunk {i} ({doc.metadata.get('source', '')})"):
                        st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
                st.subheader("Needs human review")
                needs = result["needs_review"]
                st.write("**Yes**" if needs else "**No**")
                if needs:
                    st.caption(f"Reason: {result['review_reason']}")
            except Exception as e:
                st.error(str(e))


def render_extraction():
    st.title("Document extraction")
    st.caption("Extract structured fields (default schema: dates, parties, amounts, summary). 'Needs human review' if uncertain or validation fails.")

    vs = st.session_state.get("vector_store")
    if vs is None:
        st.warning("Ingest documents first (Ingest page).")
        return

    if st.button("Run extraction"):
        with st.spinner("Retrieving and extracting..."):
            try:
                llm = get_llm()
                result = run_extraction(vs, DefaultExtractionSchema, llm)
                st.subheader("Extracted record")
                st.json(result["record"])
                if result.get("uncertain_fields"):
                    st.caption(f"Uncertain fields: {result['uncertain_fields']}")
                if result.get("validation_errors"):
                    st.caption(f"Validation errors: {result['validation_errors']}")
                st.subheader("Needs human review")
                needs = result["needs_review"]
                st.write("**Yes**" if needs else "**No**")
                if needs:
                    st.caption(f"Reason: {result['review_reason']}")
            except Exception as e:
                st.error(str(e))


if __name__ == "__main__":
    main()
