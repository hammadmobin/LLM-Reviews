from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(config):
    """
    Creates a TF-IDF vectorizer using the config dict.
    """
    return TfidfVectorizer(**config)
