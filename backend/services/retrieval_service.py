"""
Product retrieval using SentenceTransformer embeddings.
"""

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model once
_model = SentenceTransformer("all-MiniLM-L6-v2")
_index_data = None


def _load_index():
    """Load the product index once."""
    global _index_data
    if _index_data is None:
        with open("retrieval/product_index.pkl", "rb") as f:
            _index_data = pickle.load(f)
    return _index_data


def _embed_query(query: str):
    """Embed the user's search query."""
    return _model.encode(query)


def _cosine_similarity(query_vec, all_vecs):
    """Compute cosine similarity."""
    query_vec = np.array(query_vec)
    query_norm = query_vec / np.linalg.norm(query_vec)

    all_vecs = np.array(all_vecs)
    all_norms = all_vecs / np.linalg.norm(all_vecs, axis=1, keepdims=True)

    return all_norms @ query_norm


def search_products(query: str, top_n: int = 5):
    """Return the top matching products."""
    index_data = _load_index()

    query_embedding = _embed_query(query)

    similarities = _cosine_similarity(
        query_embedding,
        index_data["embeddings"]
    )

    top_indices = np.argsort(similarities)[::-1][:top_n]

    results = []
    for idx in top_indices:
        product = index_data["products"][idx]
        results.append({
            **product,
            "relevance_score": float(similarities[idx])
        })

    return results