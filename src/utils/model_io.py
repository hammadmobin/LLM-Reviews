import joblib
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# =====================
# Save & Load Baseline Models
# =====================

def save_baseline_model(model, vectorizer, path="models/baseline"):
    os.makedirs(path, exist_ok=True)
    
    joblib.dump(model, f"{path}/classifier.pkl")
    joblib.dump(vectorizer, f"{path}/vectorizer.pkl")

    print(f"[SAVED] Baseline model stored at {path}")


def load_baseline_model(path="models/baseline"):
    clf = joblib.load(f"{path}/classifier.pkl")
    vec = joblib.load(f"{path}/vectorizer.pkl")
    print(f"[LOADED] Baseline model loaded from {path}")
    return clf, vec


# =====================
# Save & Load Transformer Models
# =====================

def save_transformer_model(model, tokenizer, path="models/transformer"):
    os.makedirs(path, exist_ok=True)

    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    print(f"[SAVED] Transformer model stored at {path}")


def load_transformer_model(path="models/transformer"):
    model = AutoModelForSequenceClassification.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)

    print(f"[LOADED] Transformer model loaded from {path}")

    return model, tokenizer
