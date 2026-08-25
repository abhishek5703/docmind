"""
Ingestion pipeline for DocMind.

Handles:
- Parsing PDFs (page-aware, so citations can point to a page) and .txt files
- Feature 1: chunking with overlap, split on paragraph/sentence boundaries
  where possible instead of blind fixed-size cuts
- Embedding chunks locally with sentence-transformers
- Storing chunks + metadata in ChromaDB (metadata powers Feature 2: citations)
"""

import os
import re
import uuid

import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

import config
from vectorstore import get_vector_store


def load_embedder():
    return SentenceTransformer(config.EMBEDDING_MODEL)


def extract_pages(filepath: str):
    """Returns a list of (page_number, text) tuples. Page 1-indexed.
    Falls back to a single 'page' for plain text files."""
    if filepath.lower().endswith(".pdf"):
        doc = fitz.open(filepath)
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                pages.append((i + 1, text))
        doc.close()
        return pages
    else:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return [(1, f.read())]


def chunk_text(text: str, chunk_size: int = None, overlap: int = None):
    """
    Feature 1: Smart chunking.
    Splits on paragraph boundaries first, then sentence boundaries, and only
    falls back to a hard character cut if a single paragraph is too long.
    Adjacent chunks overlap so context isn't lost at the boundary.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    # Normalize whitespace, split into paragraphs
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 1 <= chunk_size:
            current = f"{current}\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            if len(para) > chunk_size:
                # paragraph itself is too long -> split on sentences
                sentences = re.split(r"(?<=[.!?])\s+", para)
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) + 1 <= chunk_size:
                        current = f"{current} {sent}".strip()
                    else:
                        if current:
                            chunks.append(current)
                        current = sent
            else:
                current = para

    if current:
        chunks.append(current)

    # Add overlap by prepending the tail of the previous chunk
    overlapped = []
    for i, c in enumerate(chunks):
        if i == 0 or overlap == 0:
            overlapped.append(c)
        else:
            tail = chunks[i - 1][-overlap:]
            overlapped.append(f"{tail} {c}")

    return overlapped


def ingest_file(filepath: str, store, embedder):
    """Parses, chunks, embeds, and stores a single file. Returns #chunks added."""
    filename = os.path.basename(filepath)
    pages = extract_pages(filepath)

    ids, docs, metadatas = [], [], []

    for page_num, page_text in pages:
        for chunk in chunk_text(page_text):
            if len(chunk.strip()) < 20:
                continue  # skip near-empty chunks
            ids.append(str(uuid.uuid4()))
            docs.append(chunk)
            metadatas.append({
                "source": filename,
                "page": page_num,
            })

    if not docs:
        return 0

    embeddings = embedder.encode(docs, show_progress_bar=False).tolist()

    store.add(
        ids=ids,
        documents=docs,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return len(docs)


def ingest_directory(directory: str, store, embedder):
    """Ingests every .pdf/.txt file in a directory. Returns total chunk count."""
    total = 0
    for fname in os.listdir(directory):
        if fname.lower().endswith((".pdf", ".txt")):
            fpath = os.path.join(directory, fname)
            total += ingest_file(fpath, store, embedder)
    return total


if __name__ == "__main__":
    # CLI usage: python ingest.py
    os.makedirs(config.DATA_DIR, exist_ok=True)
    embedder = load_embedder()
    store = get_vector_store()
    count = ingest_directory(config.DATA_DIR, store, embedder)
    print(f"Ingested {count} chunks from '{config.DATA_DIR}' into the vector store.")
