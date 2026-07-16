"""
Builds a searchable vector index from the product catalog.

Run: python build_index.py
Reads: data/raw/products.csv
Saves: retrieval/product_index.pkl
"""

import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def product_to_text(row):
    """Convert a product row into a descriptive text."""
    stock = "in stock" if row["in_stock"] else "out of stock"
    organic = "organic " if row["organic"] else ""
    return (
        f"{organic}{row['name']}, "
        f"category {row['category']}, "
        f"brand {row['brand']}, "
        f"price ${row['price']}, "
        f"{stock}"
    )


# 1. Load products
products = pd.read_csv("data/raw/products.csv")
print(f"Loaded {len(products)} products")

# 2. Convert each product to text
products["text"] = products.apply(product_to_text, axis=1)
texts = products["text"].tolist()

# 3. Generate embeddings
print("Generating embeddings...")
embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True
)

# 4. Save index
index_data = {
    "embeddings": embeddings,
    "products": products.to_dict("records")
}

with open("retrieval/product_index.pkl", "wb") as f:
    pickle.dump(index_data, f)

print(f"Saved index with {len(products)} products to retrieval/product_index.pkl")
print(f"Embedding dimensions: {embeddings.shape[1]}")