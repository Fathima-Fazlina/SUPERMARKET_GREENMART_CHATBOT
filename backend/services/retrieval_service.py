import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

_index = None


def _load():
    global _index

    if _index is None:
        with open("retrieval/product_index.pkl", "rb") as f:
            _index = pickle.load(f)

    return _index


def search_products(query, top_n=5):

    index = _load()

    query_vec = index["vectorizer"].transform([query])

    similarities = cosine_similarity(
        query_vec,
        index["matrix"]
    )[0]

    top = np.argsort(similarities)[::-1][:top_n]

    results = []

    for i in top:
        product = index["products"][i]
        results.append({
            **product,
            "relevance_score": float(similarities[i])
        })

    return results