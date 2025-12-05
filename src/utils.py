import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import os


# ============================================================
# Basic Utilities
# ============================================================
def load_training_data() -> pd.DataFrame:
    return pd.read_excel(os.path.join("data", "raw", "Allegato 1 - data_classification.xlsx"))


def load_eval_data() -> pd.DataFrame:
    return pd.read_excel(os.path.join("data", "raw", "Allegato 2 - data_evaluation.xlsx"))

# ============================================================
# Data Preprocessing Utilities
# ============================================================
def clean_text(text: str) -> str:
    """
    Minimal cleaning for classical ML workflows:
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


def build_cleaner():
    """
    Returns a sklearn FunctionTransformer that applies clean_text.
    Suitable for TF–IDF pipelines.
    """
    return FunctionTransformer(
        lambda x: [clean_text(t) for t in x],
        validate=False
    )


def deduplicate_reviews(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    """
    Updated deduplication pipeline:

    1. Clean text (lowercase + normalize);
    2. Group by cleaned text.
    3. If group labels conflict → drop the entire group.
    4. If group labels are consistent → keep ONE copy.

    Returns a cleaned DataFrame.
    """

    # Step 1 — Clean the text BEFORE grouping
    df = df.copy()
    df["_clean_text"] = df[text_col].apply(clean_text)

    keep_indices = []

    grouped = df.groupby("_clean_text")
    for clean_txt, group in grouped:
        unique_labels = group[label_col].unique()

        if len(unique_labels) > 1:
            # Conflict results in removing whole group
            continue

        # Keep one example
        keep_indices.append(group.index[0])

    cleaned_df = df.loc[keep_indices].copy()
    cleaned_df.drop(columns=["_clean_text"], inplace=True)

    return cleaned_df

# ============================================================
# TF–IDF Utilities
# ============================================================
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
        preprocessor=clean_text,      # clean text BEFORE tokenizing
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features
    )
