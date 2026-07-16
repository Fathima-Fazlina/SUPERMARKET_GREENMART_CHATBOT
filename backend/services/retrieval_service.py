"""
Wraps the product retrieval/search logic using Gemini Embeddings API.
Replaces sentence-transformers to keep deployment bundle size small.
"""
import os
import pickle

import numpy as np
from google import genai

_client = None
_index_data = None

EMBEDDING_MODEL = "text-embedding-004"


def _get_client():
    """Lazy-initialize the Gemini client once."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _client


def _load_index():
    """Lazy-load the product index once."""
    global _index_data
    if _index_data is None:
        with open("retrieval/product_index.pkl", "rb") as f:
            _index_data = pickle.load(f)
    return _index_data


def _embed_query(query: str) -> list[float]:
    """Embed a single query string using Gemini text-embedding-004."""
    client = _get_client()
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[query],
    )
    return result.embeddings[0].values


def _cosine_similarity(query_vec: np.ndarray, all_vecs: np.ndarray) -> np.ndarray:
    """Computes cosine similarity between a query and all product embeddings."""
    query_norm = query_vec / np.linalg.norm(query_vec)
    all_norms = all_vecs / np.linalg.norm(all_vecs, axis=1, keepdims=True)
    return all_norms @ query_norm


def search_products(query: str, top_n: int = 5) -> list[dict]:
    """
    Returns the top_n most relevant products for a given customer query,
    using Gemini embeddings for semantic search.
    """
    index_data = _load_index()
    query_embedding = np.array(_embed_query(query))
    similarities = _cosine_similarity(query_embedding, index_data["embeddings"])
    top_indices = np.argsort(similarities)[::-1][:top_n]
    results = []
    for idx in top_indices:
        product = index_data["products"][idx]
        results.append({**product, "relevance_score": float(similarities[idx])})
    return results
