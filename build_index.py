"""
Builds a searchable vector index from the product catalog using Gemini Embeddings.
Replaces sentence-transformers to keep the deployment bundle size small.

Run: python build_index.py
Reads: data/raw/products.csv
Saves: retrieval/product_index.pkl
"""

import os
import pickle
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
EMBEDDING_MODEL = "text-embedding-004"
BATCH_SIZE = 100  # Gemini supports up to 100 texts per request


def product_to_text(row):
    """Convert a product row into a single descriptive text for embedding."""
    stock = "in stock" if row["in_stock"] else "out of stock"
    organic = "organic " if row["organic"] else ""
    return f"{organic}{row['name']}, category {row['category']}, brand {row['brand']}, price ${row['price']}, {stock}"


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using Gemini text-embedding-004."""
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
    )
    return [e.values for e in result.embeddings]


# 1. Load the product catalog
products = pd.read_csv("data/raw/products.csv")
print(f"Loaded {len(products)} products")

# 2. Convert each product row to descriptive text
products["text"] = products.apply(product_to_text, axis=1)
texts = products["text"].tolist()

# 3. Embed all products in batches
print(f"Embedding {len(texts)} products using Gemini {EMBEDDING_MODEL}...")
all_embeddings = []
for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i: i + BATCH_SIZE]
    print(f"  Embedding batch {i // BATCH_SIZE + 1}/{(len(texts) + BATCH_SIZE - 1) // BATCH_SIZE}...")
    embeddings = embed_batch(batch)
    all_embeddings.extend(embeddings)
    # Small delay to respect rate limits
    if i + BATCH_SIZE < len(texts):
        time.sleep(0.5)

# 4. Save the index: embeddings + the product data
index_data = {
    "embeddings": np.array(all_embeddings),
    "products": products.to_dict("records"),
    "model": EMBEDDING_MODEL,
}
with open("retrieval/product_index.pkl", "wb") as f:
    pickle.dump(index_data, f)

print(f"Saved index with {len(products)} products to retrieval/product_index.pkl")
print(f"Embedding dimensions: {len(all_embeddings[0])}")