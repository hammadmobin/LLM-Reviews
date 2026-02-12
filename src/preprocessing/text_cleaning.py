import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download assets once
nltk.download("stopwords")
nltk.download("wordnet")

STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()


def identity(text):
    """No preprocessing"""
    return text


def stopwords_only(text):
    """Lowercase + remove stopwords"""
    return " ".join([w for w in text.lower().split() if w not in STOPWORDS])


def clean(text):
    """Lowercase + stopword removal + stemming + lemmatization"""
    words = [w for w in text.lower().split() if w not in STOPWORDS]
    words = [STEMMER.stem(w) for w in words]
    words = [LEMMATIZER.lemmatize(w) for w in words]
    return " ".join(words)


def normalize_strong(text):
    """Full cleaning pipeline used in strong baseline"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    words = [STEMMER.stem(w) for w in words]
    words = [LEMMATIZER.lemmatize(w) for w in words]
    return " ".join(words)
