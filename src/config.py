RANDOM_STATE = 42
TEST_SIZE = 0.20

# === BEST PARAMS - TF–IDF ===
VECTORIZER_CONFIG = {
    "ngram_range": (1, 2),
    "min_df": 3,
    "max_df": 0.95,
    "max_features": 100_000
}

# === BEST PARAMS - LINEAR SVC ===
MODEL_CONFIG = {
    "type": "svm",
    "params": {
        "C": 0.5,
        "max_iter": 2000,
        "random_state": RANDOM_STATE,
    }
}
