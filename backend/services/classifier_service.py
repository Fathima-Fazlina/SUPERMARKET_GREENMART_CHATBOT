"""
Wraps the trained intent classifier (from train_intent_model.py).
"""
import pickle

_model = None
_vectorizer = None

def _load():
    global _model, _vectorizer
    if _model is None:
        with open("classifier/intent_model.pkl", "rb") as f:
            data = pickle.load(f)
            _model = data["model"]
            _vectorizer = data["vectorizer"]
    return _model, _vectorizer

def detect_intent(message: str) -> str:
    """Classifies a customer message into one of our intent categories."""
    model, vectorizer = _load()
    vec = vectorizer.transform([message])
    return model.predict(vec)[0]
