"""
Wraps the product retrieval/search logic (from retrieval/search.py, build_index.py).
"""
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

_model = None
_index_data = None

def _load():
    global _model, _index_data
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    if _index_data is None:
        with open("retrieval/product_index.pkl", "rb") as f:
            _index_data = pickle.load(f)
    return _model, _index_data

def _cosine_similarity(query_vec, all_vecs):
    query_norm = query_vec / np.linalg.norm(query_vec)
    all_norms = all_vecs / np.linalg.norm(all_vecs, axis=1, keepdims=True)
    return all_norms @ query_norm

def search_products(query: str, top_n: int = 5):
    model, index_data = _load()
    query_embedding = model.encode([query])[0]
    similarities = _cosine_similarity(query_embedding, index_data["embeddings"])
    top_indices = np.argsort(similarities)[::-1][:top_n]
    results = []
    for idx in top_indices:
        product = index_data["products"][idx]
        results.append({**product, "relevance_score": float(similarities[idx])})
    return results
