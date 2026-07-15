"""
Builds a searchable vector index from the product catalog.
Converts each product into a short text description, embeds it using a local
sentence-transformers model (free, no API needed), and saves the index to disk.

Run: python build_index.py
Reads: data/raw/products.csv
Saves: retrieval/product_index.pkl
"""

import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# 1. Load the product catalog
products = pd.read_csv("data/raw/products.csv")
print(f"Loaded {len(products)} products")

# 2. Turn each product row into one descriptive text chunk
#    This is what actually gets embedded and searched over
def product_to_text(row):
    stock = "in stock" if row["in_stock"] else "out of stock"
    organic = "organic " if row["organic"] else ""
    return f"{organic}{row['name']}, category {row['category']}, brand {row['brand']}, price ${row['price']}, {stock}"

products["text"] = products.apply(product_to_text, axis=1)

# 3. Load a small, fast, local embedding model
#    all-MiniLM-L6-v2 is ~80MB, runs on CPU, no API key needed
print("Loading embedding model (first run downloads it, ~80MB)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Embed all product texts at once (batched for speed)
print("Embedding all products...")
embeddings = model.encode(products["text"].tolist(), show_progress_bar=True, batch_size=64)

# 5. Save the index: embeddings + the product data they correspond to
index_data = {
    "embeddings": np.array(embeddings),
    "products": products.to_dict("records"),
}
with open("retrieval/product_index.pkl", "wb") as f:
    pickle.dump(index_data, f)

print(f"Saved index with {len(products)} products to retrieval/product_index.pkl")