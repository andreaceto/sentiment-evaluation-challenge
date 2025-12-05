from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from .utils import clean_text
from .config import VECTORIZER_CONFIG, MODEL_CONFIG


def build_model():
    """
    Build the final pipeline TF–IDF + LinearSVC.
    """

    tfidf = TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=VECTORIZER_CONFIG["ngram_range"],
        min_df=VECTORIZER_CONFIG["min_df"],
        max_df=VECTORIZER_CONFIG["max_df"],
        max_features=VECTORIZER_CONFIG["max_features"],
    )

    model_type = MODEL_CONFIG["type"]
    params = MODEL_CONFIG["params"]

    if model_type == "svm":
        clf = LinearSVC(**params)
    else:
        raise ValueError(f"Model not supported: {model_type}")

    pipeline = Pipeline([
        ("tfidf", tfidf),
        ("clf", clf),
    ])

    return pipeline
