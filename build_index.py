import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def product_to_text(row):
    stock = "in stock" if row["in_stock"] else "out of stock"
    organic = "organic " if row["organic"] else ""

    return (
        f"{organic}{row['name']} "
        f"{row['category']} "
        f"{row['brand']} "
        f"{row['price']} "
        f"{stock}"
    )


products = pd.read_csv("data/raw/products.csv")

products["text"] = products.apply(product_to_text, axis=1)

vectorizer = TfidfVectorizer(stop_words="english")

matrix = vectorizer.fit_transform(products["text"])

index = {
    "vectorizer": vectorizer,
    "matrix": matrix,
    "products": products.to_dict("records")
}

with open("retrieval/product_index.pkl", "wb") as f:
    pickle.dump(index, f)

print("Index built successfully.")