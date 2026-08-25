"""
Retrieval logic for DocMind.

Feature 3: Hybrid search - combines dense vector search (semantic meaning)
with BM25 keyword search (exact terms, names, dates), merged via
Reciprocal Rank Fusion (RRF). Pure vector search often misses exact
keyword matches; pure keyword search misses paraphrased meaning. Combining
both covers more query types.

Feature 5: Confidence fallback - if the best vector match is too distant,
we flag the result as low-confidence so the app can say "I don't know"
instead of letting the LLM hallucinate an answer.
"""

from rank_bm25 import BM25Okapi

import config


def _tokenize(text: str):
    return text.lower().split()


def vector_search(query: str, store, embedder, top_k: int = None):
    top_k = top_k or config.TOP_K_VECTOR
    query_embedding = embedder.encode([query]).tolist()[0]
    return store.query(query_embedding, top_k)


def bm25_search(query: str, all_docs: list, all_metadatas: list, top_k: int = None):
    """Keyword search over every stored chunk. Fine for personal-scale
    collections (hundreds/low thousands of chunks); for larger corpora
    you'd want a proper inverted index (e.g. Elasticsearch)."""
    top_k = top_k or config.TOP_K_BM25
    if not all_docs:
        return []

    tokenized_corpus = [_tokenize(d) for d in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(_tokenize(query))

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {"text": all_docs[i], "metadata": all_metadatas[i], "score": scores[i]}
        for i in ranked
        if scores[i] > 0
    ]


def reciprocal_rank_fusion(vector_hits: list, bm25_hits: list, k: int = 60):
    """
    Merges two ranked lists into one, using Reciprocal Rank Fusion:
    score(doc) = sum(1 / (k + rank)) across the lists it appears in.
    This rewards chunks that show up (even at different ranks) in BOTH
    the semantic and keyword results.
    """
    scores = {}
    doc_lookup = {}

    for rank, hit in enumerate(vector_hits):
        key = hit["text"]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_lookup[key] = hit

    for rank, hit in enumerate(bm25_hits):
        key = hit["text"]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_lookup.setdefault(key, hit)

    ranked_keys = sorted(scores.keys(), key=lambda k_: scores[k_], reverse=True)
    return [doc_lookup[k_] for k_ in ranked_keys[: config.TOP_K_FINAL]]


def is_confident(vector_hits: list) -> bool:
    """Feature 5: confidence check based on the closest vector match."""
    if not vector_hits:
        return False
    best_distance = min(h["distance"] for h in vector_hits)
    return best_distance <= config.DISTANCE_THRESHOLD


def hybrid_retrieve(query: str, store, embedder):
    """
    Runs vector + BM25 search, fuses results, and checks confidence.
    Returns (fused_chunks, confident: bool).
    """
    vector_hits = vector_search(query, store, embedder)

    # Pull the full corpus for BM25 (fine for personal-scale document sets)
    all_data = store.get_all()
    all_docs = all_data.get("documents", [])
    all_metadatas = all_data.get("metadatas", [])

    bm25_hits = bm25_search(query, all_docs, all_metadatas)

    fused = reciprocal_rank_fusion(vector_hits, bm25_hits)
    confident = is_confident(vector_hits)

    return fused, confident
