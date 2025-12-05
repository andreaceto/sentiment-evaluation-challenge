import pickle
from pathlib import Path

from src.utils import load_eval_data

FIRSTNAME = "Andrea"
LASTNAME = "Aceto"


def main():
    model_path = Path(f"{FIRSTNAME}_{LASTNAME}_model.pickle")

    if not model_path.exists():
        raise FileNotFoundError("Modello non trovato. Esegui prima main_train.py")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    df_eval = load_eval_data()
    X_eval = df_eval["Review"]

    preds = model.predict(X_eval)

    df_out = df_eval.copy()
    df_out["promotore_pred"] = preds

    out_path = f"output_pred_{FIRSTNAME}_{LASTNAME}.xlsx"
    df_out.to_excel(out_path, index=False)

    print(f"Output generato: {out_path}")


if __name__ == "__main__":
    main()
