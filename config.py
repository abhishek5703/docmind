"""
Central configuration for DocMind.
Tweak these values to change chunking, retrieval, and model behavior
without hunting through the codebase.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
DATA_DIR = "data"
STORE_PATH = os.path.join("vector_store", "store.pkl")

# --- Chunking (Feature 1: smart chunking with overlap) ---
CHUNK_SIZE = 800          # characters per chunk (roughly ~150-200 words)
CHUNK_OVERLAP = 150       # overlap between consecutive chunks

# --- Embeddings ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast, free, runs locally

# --- Retrieval (Feature 3: hybrid search) ---
TOP_K_VECTOR = 10         # chunks to pull from vector search
TOP_K_BM25 = 10           # chunks to pull from keyword search
TOP_K_FINAL = 8           # final chunks sent to the LLM after fusion

# --- Confidence fallback (Feature 5) ---
# Distance here is (1 - cosine_similarity), so 0 = identical, 1 = unrelated,
# 2 = opposite. If the best chunk's distance is above this, we treat
# retrieval as "not confident enough" and refuse to answer.
# NOTE: all-MiniLM-L6-v2 tends to produce higher baseline distances than
# larger embedding models, even for genuinely relevant matches - so this
# threshold is intentionally generous. Tighten it later if you notice the
# app confidently answering things it shouldn't.
DISTANCE_THRESHOLD = 0.9

# --- Conversation memory (Feature 4) ---
MAX_HISTORY_TURNS = 4     # how many previous turns to use for query reformulation

# --- LLM (Groq - free tier) ---
# Locally, these come from .env (via python-dotenv). On Streamlit Community
# Cloud, there's no .env file - instead you set these in the app's "Secrets"
# panel, and Streamlit exposes them through st.secrets. This checks both,
# so the same code works in both places.
def _get_secret(key, default=None):
    value = os.getenv(key)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


GROQ_API_KEY = _get_secret("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"