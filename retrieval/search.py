"""
Given a customer's query, finds the most relevant products from the index.

Usage (as a module):
    from retrieval.search import search_products
    results = search_products("do you have organic milk", top_n=5)
"""

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

_model = None
_index_data = None

def _load():
    """Lazy-load the model and index once, reused across calls."""
    global _model, _index_data
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    if _index_data is None:
        with open("retrieval/product_index.pkl", "rb") as f:
            _index_data = pickle.load(f)
    return _model, _index_data

def cosine_similarity(query_vec, all_vecs):
    """Computes similarity between the query and every product embedding."""
    query_norm = query_vec / np.linalg.norm(query_vec)
    all_norms = all_vecs / np.linalg.norm(all_vecs, axis=1, keepdims=True)
    return all_norms @ query_norm

def search_products(query, top_n=5):
    """Returns the top_n most relevant products for a given customer query."""
    model, index_data = _load()

    query_embedding = model.encode([query])[0]
    similarities = cosine_similarity(query_embedding, index_data["embeddings"])

    top_indices = np.argsort(similarities)[::-1][:top_n]

    results = []
    for idx in top_indices:
        product = index_data["products"][idx]
        results.append({**product, "relevance_score": float(similarities[idx])})

    return results


# Quick manual test — run this file directly to try a few example searches
if __name__ == "__main__":
    test_queries = [
        "do you have organic milk",
        "cheap snacks",
        "something for breakfast",
    ]
    for q in test_queries:
        print(f"\nQuery: '{q}'")
        for r in search_products(q, top_n=3):
            print(f"  - {r['name']} (${r['price']}, score={r['relevance_score']:.3f})")