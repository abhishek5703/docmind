<div align="center">

# 🧠 DocMind

### Chat with your own documents — grounded in real citations, zero hallucination tolerance.

A Retrieval-Augmented Generation (RAG) app that turns your PDFs and notes into a
searchable, conversational knowledge base — with hybrid retrieval, follow-up
memory, and an honest **"I don't know"** when the answer just isn't in your files.

<br>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](#)
[![Groq](https://img.shields.io/badge/LLM-Groq%20(free)-F55036?style=for-the-badge&logo=lightning&logoColor=white)](#)
[![Cost](https://img.shields.io/badge/Cost-%240-2EA043?style=for-the-badge)](#)
[![License](https://img.shields.io/badge/License-MIT-8b7cff?style=for-the-badge)](#)

<br>

<img src="docs/screenshot.png" alt="DocMind — chat interface with grounded answers and source citations" width="880">

<sub>📸 Replace <code>docs/screenshot.png</code> with your own screenshot (or <code>docs/demo.gif</code>) — it renders automatically here and on GitHub.</sub>

</div>

<br>

## ✨ Why DocMind

Most "chat with your PDF" demos either hallucinate confidently or bolt on a
paid vector database you don't need for a personal document set. DocMind is
built to be **honest, inspectable, and free**:

- 🔍 **Grounded, not guessed** — every claim in an answer is backed by a
  citation you can click open and verify against the source page.
- 🧩 **Hybrid retrieval** — semantic search *and* exact keyword search, fused
  together, so it finds both vague conceptual questions and precise
  names/dates/IDs.
- 🙅 **Knows when it doesn't know** — a confidence gate refuses to answer
  rather than inventing something plausible-sounding.
- 💸 **Genuinely $0** — local embeddings, a hand-built numpy vector store, and
  Groq's free-tier LLM. No paid API keys, no compiled C++ dependencies, no
  credit card.

<br>

## 🚀 Features

| # | Feature | What it actually does |
|---|---|---|
| 1 | **Smart chunking with overlap** | Splits documents on paragraph/sentence boundaries — not blind character cuts — with overlapping context so information isn't lost at chunk edges. |
| 2 | **Source citations** | Every answer includes `[1]`, `[2]`-style markers that map straight back to the exact source file **and page number**. |
| 3 | **Hybrid search** | Dense vector search (semantic meaning) + BM25 keyword search (exact terms/names/dates), merged with Reciprocal Rank Fusion. Catches both *"what did I write about motivation"* and *"what's my AWS account ID"* style queries. |
| 4 | **Conversation memory** | Follow-ups like *"what about the second one?"* get rewritten into a standalone query using a deterministic rule (pronouns, short fragments) instead of a fragile LLM rewrite — chosen after testing showed small models invent wrong connections. |
| 5 | **Confidence-based fallback** | If retrieval similarity is too low, DocMind says so instead of letting the LLM hallucinate an answer. |

<br>

## 🏗️ Architecture

```
                 ┌──────────────┐
   PDF/TXT  ───► │   ingest.py  │  chunk (overlap) → embed → store
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────────┐
                 │ vectorstore  │  numpy cosine search + metadata,
                 │    .py       │  persisted to a local pickle file
                 └──────┬───────┘
                        │
   query ──► reformulate (llm.py) ──► hybrid_retrieve (retrieval.py)
                                            │
                              vector search ┼ BM25 search
                                            │
                                   Reciprocal Rank Fusion
                                            │
                                  confidence check (Feature 5)
                                            │
                                            ▼
                                   generate_answer (llm.py)
                                            │
                                            ▼
                                  answer + citations (app.py)
```

<br>

## 🧰 Tech stack

<div align="center">

| Layer | Choice | Why |
|---|---|---|
| UI | **Streamlit** | Fast, free hosting, zero frontend build step |
| Embeddings | **`all-MiniLM-L6-v2`** (local) | Small, fast, runs on CPU, no API cost |
| Vector store | **Hand-built numpy store** | No `chroma-hnswlib` C++ build step — pure `pip install numpy` |
| Keyword search | **`rank-bm25`** | Lightweight BM25 for exact-match queries |
| LLM | **Groq (free tier)** | Fast inference, generous free quota |
| PDF parsing | **PyMuPDF** | Page-aware extraction for accurate citations |

</div>

<br>

## ⚡ Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Get a free Groq API key → https://console.groq.com/keys (no card required)

# 3. Configure environment
cp .env.example .env
# then paste your GROQ_API_KEY into .env

# 4. Run it
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`), upload a
PDF or `.txt` file from the sidebar, hit **Ingest documents**, and start
asking questions.

<details>
<summary><b>Optional: bulk ingest via CLI</b></summary>
<br>

Drop files into the `data/` folder and run:

```bash
python ingest.py
```

</details>

<br>

## ☁️ Deploying for free

1. Push this repo to GitHub.
2. Deploy on [Streamlit Community Cloud](https://streamlit.io/cloud) — free tier.
3. Add `GROQ_API_KEY` (and optionally `GROQ_MODEL`) under the app's **Secrets**
   panel instead of committing `.env`.

That's the whole deployment story — no infra, no Docker, no paid tier required.

<br>

## 📁 Project structure

```
docmind/
├── app.py              # Streamlit UI — ties all 5 features together
├── ingest.py           # PDF/text parsing, chunking, embedding, storage
├── vectorstore.py       # Self-built numpy cosine-similarity vector store
├── retrieval.py         # Hybrid search (vector + BM25) + confidence check
├── llm.py               # Groq API calls, query reformulation, citations
├── config.py             # All tunable settings in one place
├── requirements.txt
├── .env.example
├── .streamlit/
│   └── config.toml     # App theme
├── docs/
│   └── screenshot.png   # ← drop your app screenshot here
└── data/                # Uploaded documents land here
```

<br>

## 🤔 Why no ChromaDB?

An earlier version used ChromaDB, but its `chroma-hnswlib` dependency needs a
C++ compiler to build from source on Windows unless you install Visual C++
Build Tools. Since this project is meant to be a **zero-friction, zero-cost**
demo, `vectorstore.py` implements the same core idea — store embeddings,
search by similarity, keep metadata for citations — in ~80 lines of plain
numpy. No compiler required, and it's a genuinely good interview talking
point: you understand what a vector index does under the hood instead of
treating it as a black box.

<br>

## 🎯 Tuning knobs

Everything that affects retrieval quality lives in `config.py`:

| Setting | Default | Effect |
|---|---|---|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `150` | Larger chunks = more context per citation, fewer total chunks |
| `TOP_K_VECTOR` / `TOP_K_BM25` / `TOP_K_FINAL` | `10` / `10` / `8` | How many candidates each retriever pulls before fusion |
| `DISTANCE_THRESHOLD` | `0.9` | Lower = stricter "I don't know" gate; higher = more willing to answer |
| `MAX_HISTORY_TURNS` | `4` | How much chat history feeds query reformulation |

<br>

## 🗣️ Talking points for interviews

- **Chunking strategy** — why paragraph/sentence-aware chunking with overlap
  beats naive fixed-size splitting (no mid-sentence cuts, context preserved
  across boundaries).
- **Hybrid retrieval** — why pure vector search misses exact-match queries
  (names, IDs, dates), and how BM25 + Reciprocal Rank Fusion fixes that
  without a re-ranker model.
- **Confidence fallback** — the trade-off in picking a distance threshold:
  too strict refuses valid questions, too loose hallucinates.
- **Conversation memory** — replacing an LLM-based query rewriter with a
  deterministic rule after testing showed the LLM version was unreliable on
  a small/fast model — robustness over sophistication when correctness
  matters.

<br>

## 🛣️ Roadmap

- [ ] Cross-encoder re-ranking on top of the fused candidates
- [ ] Streaming responses token-by-token
- [ ] Retrieval evaluation harness (recall@k, answer faithfulness)
- [ ] Multi-user / multi-workspace document isolation

<br>

## 📄 License

MIT — use it, fork it, put it in your portfolio.

<div align="center">
<sub>Built with local embeddings, hybrid search, and $0 in API costs.</sub>
</div>