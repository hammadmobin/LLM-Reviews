from sklearn.svm import LinearSVC

BASELINE_CLASSIFIER = LinearSVC()

TRAIN_TEST_SPLIT = {
    "test_size": 0.20,
    "random_state": 42,
    "stratify": True
}

BASELINE_SIMPLE_TFIDF = {
    "vectorizer": {"max_features": 50000, "ngram_range": (1,2)},
    "preprocess": None
}

BASELINE_CLEAN = {
    "vectorizer": {"max_features": 50000, "ngram_range": (1,2)},
    "preprocess": "clean"
}

BASELINE_CHAR = {
    "vectorizer": {"analyzer": "char", "ngram_range": (3,5), "max_features": 50000},
    "preprocess": None
}

BASELINE_STRONG = {
    "vectorizer": {"max_features": 50000, "ngram_range": (1,2)},
    "preprocess": "normalize_strong"
}

BASELINE_STOPWORDS_ONLY = {
    "vectorizer": {"max_features": 50000, "ngram_range": (1,2)},
    "preprocess": "stopwords"
}
