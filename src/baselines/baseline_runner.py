import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

from src.config.config_baselines import (
    BASELINE_CLASSIFIER,
    BASELINE_SIMPLE_TFIDF,
    BASELINE_CLEAN,
    BASELINE_CHAR,
    BASELINE_STRONG,
    BASELINE_STOPWORDS_ONLY
)

from src.preprocessing.text_cleaning import (
    identity,
    stopwords_only,
    clean,
    normalize_strong
)

from src.baselines.tfidf_vectorizers import build_vectorizer


# Preprocessing mapping
PREPROCESS_MAP = {
    None: identity,
    "clean": clean,
    "normalize_strong": normalize_strong,
    "stopwords": stopwords_only
}


def run_baseline(X, y, baseline_config, name="Baseline"):
    print(f"\n======== Running {name} ========")

    preprocess_func = PREPROCESS_MAP[baseline_config["preprocess"]]

    # Apply preprocessing
    X_proc = X.apply(preprocess_func)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=0.20, random_state=42, stratify=y
    )

    # Build vectorizer
    vectorizer = build_vectorizer(baseline_config["vectorizer"])

    # Vectorization
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Fit classifier
    clf = BASELINE_CLASSIFIER
    clf.fit(X_train_vec, y_train)

    preds = clf.predict(X_test_vec)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("F1-score:", f1_score(y_test, preds))
    print(classification_report(y_test, preds))


def run_all_baselines(auth_df, suspicious_df):
    """
    Takes authentic & suspicious dataframe text columns and runs all baselines.
    """
    X = pd.concat([auth_df["text"], suspicious_df["text"]], ignore_index=True).astype(str)
    y = pd.Series([0] * len(auth_df) + [1] * len(suspicious_df))

    run_baseline(X, y, BASELINE_SIMPLE_TFIDF, name="Simple TF-IDF")
    run_baseline(X, y, BASELINE_CLEAN, name="Clean TF-IDF")
    run_baseline(X, y, BASELINE_CHAR, name="Char TF-IDF")
    run_baseline(X, y, BASELINE_STRONG, name="Strong Baseline")
    run_baseline(X, y, BASELINE_STOPWORDS_ONLY, name="Stopwords Baseline")
