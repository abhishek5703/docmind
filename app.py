"""
DocMind - Personal Document Q&A Assistant
Streamlit app wiring together all 5 features:
  1. Smart chunking with overlap        (ingest.py)
  2. Source citations with page numbers (llm.py + rendering below)
  3. Hybrid search (vector + BM25)      (retrieval.py)
  4. Conversation memory                (llm.py: reformulate_query)
  5. Confidence / fallback handling     (retrieval.py: is_confident)

UI layer only — every call into ingest.py / vectorstore.py / retrieval.py /
llm.py below uses the exact same functions and arguments as the original
app, so behavior is unchanged; this file only changes what it looks like.
"""

import os
import time
import datetime as dt

import streamlit as st

import config
from ingest import load_embedder, ingest_file
from vectorstore import get_vector_store
from retrieval import hybrid_retrieve
from llm import reformulate_query, generate_answer

st.set_page_config(
    page_title="DocMind — Chat with your documents",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────
# Premium visual theme (CSS only — no behavior here)
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Sora:wght@600;700;800&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

/* ---------- App background ---------- */
.stApp {
    background:
        radial-gradient(1200px 600px at 10% -10%, rgba(139,124,255,0.16), transparent 60%),
        radial-gradient(1000px 500px at 110% 10%, rgba(34,211,238,0.10), transparent 55%),
        #0B0E14;
}

/* ---------- Hide default chrome ---------- */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #10131D 0%, #0C0F17 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }

/* ---------- Headings ---------- */
h1, h2, h3 { font-family: 'Sora', sans-serif; letter-spacing: -0.02em; }

.dm-brand {
    display: flex; align-items: center; gap: 0.6rem;
    margin-bottom: 0.15rem;
}
.dm-brand .dm-logo {
    font-size: 1.6rem;
    filter: drop-shadow(0 0 12px rgba(139,124,255,0.55));
}
.dm-brand .dm-title {
    font-family: 'Sora', sans-serif;
    font-weight: 800;
    font-size: 1.35rem;
    background: linear-gradient(90deg, #C9C3FF 0%, #8B7CFF 45%, #22D3EE 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.dm-tagline {
    color: #9AA3B7; font-size: 0.82rem; margin-top: -2px; margin-bottom: 1.1rem;
}

/* ---------- Hero header (main area) ---------- */
.dm-hero {
    padding: 1.6rem 1.8rem;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(139,124,255,0.14), rgba(34,211,238,0.07));
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1.4rem;
}
.dm-hero h1 {
    font-size: 1.9rem; margin: 0 0 0.25rem 0; color: #F1F2F8;
}
.dm-hero p { color: #A8B0C2; margin: 0; font-size: 0.95rem; }
.dm-pill-row { margin-top: 0.9rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }
.dm-pill {
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.28rem 0.7rem; border-radius: 999px;
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.09);
    font-size: 0.78rem; color: #C7CCDC;
}
.dm-pill b { color: #E9EBF5; }
.dm-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
.dm-dot.on { background: #34D399; box-shadow: 0 0 8px #34D39980; }
.dm-dot.off { background: #F87171; box-shadow: 0 0 8px #F8717180; }

/* ---------- Cards (stat cards, onboarding, sources) ---------- */
.dm-card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1rem 1.1rem;
}
.dm-stat {
    text-align: center;
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 0.7rem 0.4rem;
}
.dm-stat .n {
    font-family: 'Sora', sans-serif; font-weight: 800; font-size: 1.4rem;
    background: linear-gradient(90deg, #C9C3FF, #22D3EE);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.dm-stat .l { color: #8D96A9; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em; }

/* ---------- Onboarding steps ---------- */
.dm-step {
    background: rgba(255,255,255,0.03);
    border: 1px dashed rgba(255,255,255,0.14);
    border-radius: 16px;
    padding: 1.1rem 1rem;
    height: 100%;
}
.dm-step .num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 26px; height: 26px; border-radius: 50%;
    background: linear-gradient(135deg, #8B7CFF, #22D3EE);
    color: #0B0E14; font-weight: 800; font-size: 0.8rem; margin-bottom: 0.5rem;
}
.dm-step h4 { margin: 0.2rem 0 0.25rem 0; color: #EDEFF6; font-size: 0.98rem; }
.dm-step p { margin: 0; color: #98A1B4; font-size: 0.83rem; line-height: 1.4; }

/* ---------- Source / citation cards ---------- */
.dm-source {
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 0.65rem 0.85rem;
    margin-bottom: 0.5rem;
}
.dm-source .head {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 0.3rem;
}
.dm-source .name { color: #E4E7F1; font-size: 0.86rem; font-weight: 600; }
.dm-source .page {
    font-size: 0.72rem; color: #9AA3B7;
    background: rgba(255,255,255,0.06); padding: 0.1rem 0.5rem; border-radius: 999px;
}
.dm-source .snippet { color: #93A0B5; font-size: 0.79rem; line-height: 1.45; }
.dm-bar-bg { background: rgba(255,255,255,0.07); border-radius: 999px; height: 5px; margin-top: 0.45rem; overflow: hidden; }
.dm-bar-fg { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #8B7CFF, #22D3EE); }

/* ---------- Confidence badge ---------- */
.dm-badge {
    display: inline-flex; align-items: center; gap: 0.35rem;
    font-size: 0.74rem; padding: 0.2rem 0.6rem; border-radius: 999px;
    margin-bottom: 0.55rem; font-weight: 600;
}
.dm-badge.high { background: rgba(52,211,153,0.14); color: #6EE7B7; border: 1px solid rgba(52,211,153,0.3); }
.dm-badge.low  { background: rgba(248,113,113,0.14); color: #FCA5A5; border: 1px solid rgba(248,113,113,0.3); }

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    background: rgba(255,255,255,0.04) !important;
    color: #E7E9F1 !important;
    transition: all 0.15s ease;
    font-weight: 500 !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: rgba(139,124,255,0.55) !important;
    box-shadow: 0 0 0 3px rgba(139,124,255,0.12);
    transform: translateY(-1px);
}
div[data-testid="stFormSubmitButton"] button,
.dm-primary button {
    background: linear-gradient(135deg, #8B7CFF, #22D3EE) !important;
    color: #0B0E14 !important; font-weight: 700 !important; border: none !important;
}

/* ---------- File uploader ---------- */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1.5px dashed rgba(139,124,255,0.35) !important;
    border-radius: 14px !important;
}

/* ---------- Chat bubbles ---------- */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 0.35rem 0.2rem;
    margin-bottom: 0.6rem;
}

/* ---------- Chat input ---------- */
[data-testid="stChatInput"] textarea { border-radius: 14px !important; }

/* ---------- Misc ---------- */
.dm-footer {
    text-align: center; color: #5C6478; font-size: 0.76rem;
    margin-top: 2rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.06);
}
hr { border-color: rgba(255,255,255,0.08) !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────
# Cached resources (loaded once per session) — unchanged logic
# ──────────────────────────────────────────────────────────────────────────
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
if "confirm_clear_docs" not in st.session_state:
    st.session_state.confirm_clear_docs = False
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

SUGGESTED_PROMPTS = [
    "📋 Summarize these documents",
    "🔑 What are the key points?",
    "🧩 What questions can these documents answer?",
]


def _file_icon(filename: str) -> str:
    return "📕" if filename.lower().endswith(".pdf") else "📝"


def _relevance_pct(chunk: dict, rank: int, total: int) -> int:
    """Cosmetic-only relevance bar. Uses the vector 'distance' field when
    present (lower distance = more relevant), otherwise falls back to a
    rank-based estimate. Purely presentational — does not affect retrieval,
    the confidence check, or which chunks are sent to the LLM."""
    dist = chunk.get("distance")
    if dist is not None:
        return max(5, min(100, round((1 - dist) * 100)))
    return max(15, round(100 - (rank / max(total, 1)) * 70))


# ──────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="dm-brand">
            <span class="dm-logo">🧠</span>
            <span class="dm-title">DocMind</span>
        </div>
        <div class="dm-tagline">Chat with your own documents — grounded in real citations.</div>
        """,
        unsafe_allow_html=True,
    )

    if not config.GROQ_API_KEY:
        st.warning("No `GROQ_API_KEY` found. Add it in `.env` or your Streamlit secrets to enable answers.", icon="⚠️")

    st.markdown("##### 📤 Upload documents")
    uploaded_files = st.file_uploader(
        "PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        total_size_kb = sum(f.size for f in uploaded_files) / 1024
        st.caption(f"{len(uploaded_files)} file(s) selected · {total_size_kb:,.0f} KB")

    ingest_clicked = st.button("⚡ Ingest documents", use_container_width=True, disabled=not uploaded_files)

    if ingest_clicked and uploaded_files:
        total_chunks = 0
        progress = st.progress(0, text="Starting…")
        for idx, f in enumerate(uploaded_files, start=1):
            progress.progress(
                (idx - 1) / len(uploaded_files),
                text=f"Chunking & embedding {f.name}…",
            )
            save_path = os.path.join(config.DATA_DIR, f.name)
            with open(save_path, "wb") as out:
                out.write(f.getbuffer())
            total_chunks += ingest_file(save_path, store, embedder)
            progress.progress(idx / len(uploaded_files), text=f"Indexed {f.name}")
        time.sleep(0.2)
        progress.empty()
        st.success(f"✅ Ingested **{total_chunks}** chunks from **{len(uploaded_files)}** file(s).")

    st.markdown("---")

    doc_count = store.count()
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f'<div class="dm-stat"><div class="n">{doc_count}</div><div class="l">Chunks indexed</div></div>',
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f'<div class="dm-stat"><div class="n">{len(st.session_state.history)//2}</div><div class="l">Questions asked</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("⚙️ Under the hood"):
        st.caption(f"**Embedding model:** `{config.EMBEDDING_MODEL}`")
        st.caption(f"**LLM:** `{config.GROQ_MODEL or 'not set'}`")
        st.caption(f"**Chunk size / overlap:** `{config.CHUNK_SIZE}` / `{config.CHUNK_OVERLAP}` chars")
        st.caption(f"**Retrieval:** hybrid (vector + BM25), top **{config.TOP_K_FINAL}** after fusion")
        st.caption(f"**Confidence threshold:** `{config.DISTANCE_THRESHOLD}`")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🧹 Clear chat", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    with c2:
        if st.button("🗑️ Clear docs", use_container_width=True):
            st.session_state.confirm_clear_docs = True

    if st.session_state.confirm_clear_docs:
        st.warning("This deletes **all** indexed documents and chat history. Are you sure?")
        cc1, cc2 = st.columns(2)
        with cc1:
            if st.button("Yes, delete everything", use_container_width=True, type="primary"):
                store.clear()
                for fname in os.listdir(config.DATA_DIR):
                    fpath = os.path.join(config.DATA_DIR, fname)
                    if os.path.isfile(fpath):
                        os.remove(fpath)
                st.session_state.history = []
                st.session_state.confirm_clear_docs = False
                st.success("All documents and chat history cleared.")
                st.rerun()
        with cc2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.confirm_clear_docs = False
                st.rerun()

    if st.session_state.history:
        transcript = "\n\n".join(
            f"{t['role'].upper()}: {t['content']}" for t in st.session_state.history
        )
        st.download_button(
            "⬇️ Download transcript",
            transcript,
            file_name=f"docmind_chat_{dt.date.today().isoformat()}.txt",
            use_container_width=True,
        )

    st.markdown(
        '<div class="dm-footer">Built with local embeddings + Groq · $0 cost</div>',
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────────
# Main area — hero header
# ──────────────────────────────────────────────────────────────────────────
key_dot = "on" if config.GROQ_API_KEY else "off"
key_label = "Connected" if config.GROQ_API_KEY else "Not configured"

st.markdown(
    f"""
    <div class="dm-hero">
        <h1>Ask your documents 🧠✨</h1>
        <p>Upload a PDF or note, then ask anything — every answer is grounded in your own text, with citations back to the exact page.</p>
        <div class="dm-pill-row">
            <span class="dm-pill">📚 <b>{doc_count}</b>&nbsp;chunks indexed</span>
            <span class="dm-pill">🔎 Hybrid search (vector + BM25)</span>
            <span class="dm-pill"><span class="dm-dot {key_dot}"></span>&nbsp;Groq: {key_label}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Empty state / onboarding ---
if doc_count == 0:
    st.markdown("#### Get started in 3 steps")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(
            """<div class="dm-step"><div class="num">1</div>
            <h4>📤 Upload</h4><p>Drop a PDF or .txt file into the sidebar uploader.</p></div>""",
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            """<div class="dm-step"><div class="num">2</div>
            <h4>⚡ Ingest</h4><p>Click "Ingest documents" — it's chunked, embedded, and indexed locally.</p></div>""",
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            """<div class="dm-step"><div class="num">3</div>
            <h4>💬 Ask</h4><p>Ask a question below and get a grounded answer with citations.</p></div>""",
            unsafe_allow_html=True,
        )
    st.markdown("<br>", unsafe_allow_html=True)

# --- Render past turns ---
for turn in st.session_state.history:
    avatar = "🧑‍💻" if turn["role"] == "user" else "🧠"
    with st.chat_message(turn["role"], avatar=avatar):
        st.markdown(turn["content"])

# --- Suggested prompts (only before the first question, once docs exist) ---
if doc_count > 0 and not st.session_state.history:
    st.caption("Try asking:")
    chip_cols = st.columns(len(SUGGESTED_PROMPTS))
    for i, prompt in enumerate(SUGGESTED_PROMPTS):
        with chip_cols[i]:
            if st.button(prompt, key=f"chip_{i}", use_container_width=True):
                st.session_state.pending_question = prompt.split(" ", 1)[1]

# ──────────────────────────────────────────────────────────────────────────
# Question handling — same functional pipeline as the original app
# ──────────────────────────────────────────────────────────────────────────
question = st.chat_input("Ask a question about your documents...")
if not question and st.session_state.pending_question:
    question = st.session_state.pending_question
st.session_state.pending_question = None

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🧠"):
        if store.count() == 0:
            answer = "Please upload and ingest some documents first."
            st.markdown(answer)
        else:
            with st.spinner("Reading through your documents…"):
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
                    st.markdown(
                        '<span class="dm-badge low">🟡 Low confidence — refused to guess</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(answer)
                else:
                    try:
                        answer = generate_answer(
                            standalone_query, chunks, st.session_state.history[:-1]
                        )
                    except Exception as e:
                        answer = (
                            "Something went wrong while generating the answer. "
                            f"({e})"
                        )
                        st.error(answer)
                    else:
                        st.markdown(
                            '<span class="dm-badge high">🟢 Grounded in your documents</span>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(answer)

                        # Feature 2: citations, rendered below the answer
                        with st.expander(f"📎 Sources ({len(chunks)})"):
                            for i, chunk in enumerate(chunks, start=1):
                                src = chunk["metadata"].get("source", "unknown")
                                page = chunk["metadata"].get("page", "?")
                                snippet = chunk["text"][:300].strip().replace("\n", " ")
                                pct = _relevance_pct(chunk, i - 1, len(chunks))
                                st.markdown(
                                    f"""
                                    <div class="dm-source">
                                        <div class="head">
                                            <span class="name">{_file_icon(src)} [{i}] {src}</span>
                                            <span class="page">page {page}</span>
                                        </div>
                                        <div class="snippet">{snippet}…</div>
                                        <div class="dm-bar-bg"><div class="dm-bar-fg" style="width:{pct}%"></div></div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

    st.session_state.history.append({"role": "assistant", "content": answer})

st.markdown(
    '<div class="dm-footer">DocMind — local embeddings, hybrid search, zero paid services.</div>',
    unsafe_allow_html=True,
)
