import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)

from src.utils import load_training_data, deduplicate_reviews
from src.models import build_model
from src.config import RANDOM_STATE, TEST_SIZE

FIRSTNAME = "Andrea"
LASTNAME = "Aceto"


def main():
    print("\n=== CARICAMENTO TRAINING DATA ===")
    df = load_training_data()
    print("Shape iniziale:", df.shape)

    # Deduplica
    df = deduplicate_reviews(df, text_col="Review", label_col="Promotore")
    print("Shape dopo dedup:", df.shape)

    X = df["Review"]
    y = df["Promotore"]

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    print("\n=== COSTRUZIONE MODELLO (Linear SVC) ===")
    model = build_model()

    print("\n=== TRAINING ===")
    model.fit(X_train, y_train)

    print("\n=== VALUTAZIONE ===")
    y_pred = model.predict(X_test)

    # Evaluation metrics
    y_scores = model.decision_function(X_test)

    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_scores)
    cm = confusion_matrix(y_test, y_pred)

    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC : {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # ROC curve plot
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.title(f"ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig("roc_curve.png")
    plt.close()

    # Confusion matrix plot
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.savefig("confusion_matrix.png")
    plt.close()

    print("\nGrafici salvati: roc_curve.png, confusion_matrix.png")

    print("\n=== SALVATAGGIO MODELLO ===")
    model_filename = f"{FIRSTNAME}_{LASTNAME}_model.pickle"
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)

    print(f"Modello salvato in: {model_filename}")


if __name__ == "__main__":
    main()
