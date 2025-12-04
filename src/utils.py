import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
from pathlib import Path


# ============================================================
# Basic Utilities
# ============================================================
def load_training_data(path: str | Path) -> pd.DataFrame:
    return pd.read_excel(path)


def load_eval_data(path: str | Path) -> pd.DataFrame:
    return pd.read_excel(path)

# ============================================================
# Data Preprocessing Utilities
# ============================================================

def clean_text_ml(text: str) -> str:
    """
    Minimal cleaning ONLY for classical ML workflows:
    - lowercase
    - strip whitespace
    - normalize spaces
    - keep punctuation
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def build_cleaner_ml():
    """
    Returns a sklearn FunctionTransformer that applies clean_text_ml.
    Suitable for TF–IDF pipelines.
    """
    return FunctionTransformer(
        lambda x: [clean_text_ml(t) for t in x],
        validate=False
    )

def clean_text_transformer(text: str) -> str:
    """
    Very light cleaning for Transformers:
    - strip whitespace ONLY
    - DO NOT lowercase unless model is uncased
    - DO NOT remove punctuation / stopwords
    """
    if not isinstance(text, str):
        return ""
    return text.strip()


def build_cleaner_transformer():
    """
    A FunctionTransformer that preserves the raw text (strip only).
    """
    return FunctionTransformer(
        lambda x: [clean_text_transformer(t) for t in x],
        validate=False
    )


def get_cleaner(mode: str):
    """
    Returns the appropriate cleaning function/transformer.

    mode="ml" → cleaning for TF–IDF
    mode="transformer" → minimal cleaning for BERT-like models
    """
    if mode == "ml":
        return build_cleaner_ml()
    elif mode == "transformer":
        return build_cleaner_transformer()
    else:
        raise ValueError(f"Unknown mode: {mode}")


def deduplicate_reviews(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    """
    Updated deduplication pipeline:

    1. Clean text using ML-cleaning (lowercase + normalize);
       this ensures identical texts compare correctly.
    2. Group by cleaned text.
    3. If group labels conflict → drop the entire group.
    4. If group labels are consistent → keep ONE copy.

    Returns a cleaned DataFrame.
    """

    # Step 1 — Clean the text BEFORE grouping
    df = df.copy()
    df["_clean_text"] = df[text_col].apply(clean_text_ml)

    keep_indices = []

    grouped = df.groupby("_clean_text")
    for clean_txt, group in grouped:
        unique_labels = group[label_col].unique()

        if len(unique_labels) > 1:
            # Conflict → remove whole group
            continue

        # Keep one example
        keep_indices.append(group.index[0])

    cleaned_df = df.loc[keep_indices].copy()
    cleaned_df.drop(columns=["_clean_text"], inplace=True)

    return cleaned_df


def build_tfidf_vectorizer(
    ngram_range=(1,2),
    min_df=3,
    max_df=0.95,
    max_features=100_000
):
    """
    Builds a TF–IDF vectorizer using ML-cleaning rules.
    """
    return TfidfVectorizer(
        preprocessor=clean_text_ml,      # clean text BEFORE tokenizing
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features
    )

