"""
Trains a text classifier to detect customer intent (product_search, price_comparison,
order_tracking, promotions, store_hours, policy, loyalty_program, issue_resolution,
general_faq, chitchat) from a customer's message.

Run: python train_intent_model.py
Reads: classifier/labeled_intents.csv
Saves: classifier/intent_model.pkl
"""

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Load the labeled data
data = pd.read_csv("classifier/labeled_intents.csv")
print(f"Loaded {len(data)} labeled examples across {data['intent'].nunique()} intents")

# 2. Split into training and testing sets (80% train, 20% test)
#    This lets us check how well the model does on messages it hasn't seen
X_train, X_test, y_train, y_test = train_test_split(
    data["message"], data["intent"],
    test_size=0.2, random_state=42, stratify=data["intent"]
)

# 3. Convert text into numbers the model can learn from (TF-IDF)
#    TF-IDF scores each word by how distinctive it is to a message
vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Train a simple, fast classifier (Logistic Regression works well for this)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# 5. Check how well it performs on the held-out test messages
y_pred = model.predict(X_test_vec)
print("\n--- Performance on unseen test messages ---")
print(classification_report(y_test, y_pred))

# 6. Save the trained model + vectorizer together so the app can use them later
with open("classifier/intent_model.pkl", "wb") as f:
    pickle.dump({"model": model, "vectorizer": vectorizer}, f)

print("Saved trained classifier to classifier/intent_model.pkl")

# 7. Quick manual test with a few example messages
print("\n--- Quick test ---")
test_messages = [
    "do you have milk",
    "when do you close",
    "where is my delivery",
    "hi there",
]
for msg in test_messages:
    vec = vectorizer.transform([msg])
    prediction = model.predict(vec)[0]
    print(f"'{msg}' -> {prediction}")