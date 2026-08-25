"""
A minimal, dependency-free vector store to replace ChromaDB.

Why this exists: ChromaDB pulls in `chroma-hnswlib`, which needs a C++
compiler to build on Windows (Visual C++ Build Tools). For a personal-scale
document collection (hundreds to low-thousands of chunks), a plain numpy
cosine-similarity search is fast enough and has zero compiled dependencies -
it just works with `pip install numpy`, which you already have.

Everything is persisted to a single pickle file so your index survives
restarts, just like ChromaDB's persistent client did.
"""

import os
import pickle

import numpy as np

import config


class SimpleVectorStore:
    def __init__(self, path: str):
        self.path = path
        self.ids = []
        self.documents = []
        self.metadatas = []
        self.embeddings = None  # numpy array, shape (n_chunks, dim)
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                data = pickle.load(f)
            self.ids = data["ids"]
            self.documents = data["documents"]
            self.metadatas = data["metadatas"]
            self.embeddings = data["embeddings"]

    def _save(self):
        with open(self.path, "wb") as f:
            pickle.dump(
                {
                    "ids": self.ids,
                    "documents": self.documents,
                    "metadatas": self.metadatas,
                    "embeddings": self.embeddings,
                },
                f,
            )

    def count(self) -> int:
        return len(self.ids)

    def add(self, ids: list, documents: list, metadatas: list, embeddings: list):
        new_embeddings = np.array(embeddings, dtype=np.float32)

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        self.ids.extend(ids)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self._save()

    def get_all(self):
        return {"documents": self.documents, "metadatas": self.metadatas}

    def clear(self):
        """Wipes the entire index - all chunks, metadata, and embeddings -
        and removes the persisted file so a fresh set of documents can be
        ingested without old ones lingering in search results."""
        self.ids = []
        self.documents = []
        self.metadatas = []
        self.embeddings = None
        if os.path.exists(self.path):
            os.remove(self.path)

    def query(self, query_embedding: list, top_k: int):
        """Returns top_k hits sorted by similarity, each with a 'distance'
        field defined as (1 - cosine_similarity) so lower = more similar,
        matching the convention used elsewhere in the app (e.g. config.DISTANCE_THRESHOLD)."""
        if self.embeddings is None or len(self.ids) == 0:
            return []

        q = np.array(query_embedding, dtype=np.float32)

        # Cosine similarity = dot product of normalized vectors
        doc_norms = self.embeddings / (
            np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10
        )
        q_norm = q / (np.linalg.norm(q) + 1e-10)

        similarities = doc_norms @ q_norm
        distances = 1 - similarities

        top_k = min(top_k, len(self.ids))
        top_indices = np.argsort(distances)[:top_k]

        return [
            {
                "text": self.documents[i],
                "metadata": self.metadatas[i],
                "distance": float(distances[i]),
            }
            for i in top_indices
        ]


def get_vector_store() -> SimpleVectorStore:
    os.makedirs(os.path.dirname(config.STORE_PATH) or ".", exist_ok=True)
    return SimpleVectorStore(config.STORE_PATH)
