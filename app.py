"""
DocMind - Personal Document Q&A Assistant
Streamlit app wiring together all 5 features:
  1. Smart chunking with overlap        (ingest.py)
  2. Source citations with page numbers (llm.py + rendering below)
  3. Hybrid search (vector + BM25)      (retrieval.py)
  4. Conversation memory                (llm.py: reformulate_query)
  5. Confidence / fallback handling     (retrieval.py: is_confident)
"""

import os
import streamlit as st

import config
from ingest import load_embedder, ingest_file
from vectorstore import get_vector_store
from retrieval import hybrid_retrieve
from llm import reformulate_query, generate_answer

st.set_page_config(page_title="DocMind", page_icon="📄", layout="wide")

# --- Cached resources (loaded once per session) ---
@st.cache_resource
def get_resources():
    os.makedirs(config.DATA_DIR, exist_ok=True)
    embedder = load_embedder()
    store = get_vector_store()
    return embedder, store

embedder, store = get_resources()

# --- Session state ---
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"role": ..., "content": ...}

# --- Sidebar: upload + ingest ---
with st.sidebar:
    st.title("📄 DocMind")
    st.caption("Your personal documents, searchable and grounded.")

    st.subheader("Upload documents")
    uploaded_files = st.file_uploader(
        "PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
    )

    if uploaded_files and st.button("Ingest documents"):
        total_chunks = 0
        with st.spinner("Chunking, embedding, and indexing..."):
            for f in uploaded_files:
                save_path = os.path.join(config.DATA_DIR, f.name)
                with open(save_path, "wb") as out:
                    out.write(f.getbuffer())
                total_chunks += ingest_file(save_path, store, embedder)
        st.success(f"Ingested {total_chunks} chunks from {len(uploaded_files)} file(s).")

    st.divider()
    doc_count = store.count()
    st.metric("Chunks indexed", doc_count)

    if st.button("Clear conversation"):
        st.session_state.history = []
        st.rerun()

    if st.button("🗑️ Clear all documents"):
        store.clear()
        # Also remove the raw uploaded files from data/ so re-ingesting
        # the folder later doesn't bring back stale documents.
        for fname in os.listdir(config.DATA_DIR):
            fpath = os.path.join(config.DATA_DIR, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)
        st.session_state.history = []
        st.success("All documents and chat history cleared. Upload a new file to start fresh.")
        st.rerun()

# --- Main chat area ---
st.header("Ask your documents")

if store.count() == 0:
    st.info("Upload and ingest at least one document from the sidebar to get started.")

# Render past turns
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if store.count() == 0:
            answer = "Please upload and ingest some documents first."
            st.markdown(answer)
        else:
            with st.spinner("Thinking..."):
                # Feature 4: reformulate follow-up questions using history
                standalone_query = reformulate_query(question, st.session_state.history[:-1])

                # Feature 3: hybrid retrieval + Feature 5: confidence check
                chunks, confident = hybrid_retrieve(standalone_query, store, embedder)

                if not confident or not chunks:
                    answer = (
                        "I don't have enough relevant information in your documents "
                        "to answer that confidently. Try rephrasing, or upload a document "
                        "that covers this topic."
                    )
                    st.markdown(answer)
                else:
                    answer = generate_answer(
                        standalone_query, chunks, st.session_state.history[:-1]
                    )
                    st.markdown(answer)

                    # Feature 2: citations, rendered below the answer
                    with st.expander("📎 Sources"):
                        for i, chunk in enumerate(chunks, start=1):
                            src = chunk["metadata"].get("source", "unknown")
                            page = chunk["metadata"].get("page", "?")
                            st.markdown(f"**[{i}] {src}** — page {page}")
                            st.caption(chunk["text"][:300] + "...")

    st.session_state.history.append({"role": "assistant", "content": answer})
