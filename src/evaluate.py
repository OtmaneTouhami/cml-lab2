import argparse
import os
from pathlib import Path

# 0) forcer un backend qui n'a pas besoin d'écran
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import pandas as pd

import yaml
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def load_config(root: Path) -> dict:
    cfg_path = root / "params.yaml"
    with cfg_path.open("r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model and save figures"
    )
    parser.add_argument(
        "--exp-name", required=True, help="Experiment name (same as used in training)"
    )
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    REPORTS_DIR = ROOT / "reports"
    MODELS_DIR = ROOT / "models"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("📂 cwd        :", os.getcwd())
    print("📂 root dir   :", ROOT)
    print("📂 reports    :", REPORTS_DIR)

    # test d'écriture simple
    (REPORTS_DIR / f"{args.exp_name}__DEBUG_EVALUATE_RAN.txt").write_text(
        "evaluate.py ran successfully.\n"
    )

    # données
    cfg = load_config(ROOT)
    data_cfg = cfg.get("data", {})
    iris = load_iris()
    X, y = iris.data, iris.target
    X_df = pd.DataFrame(X, columns=iris.feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X_df,
        y,
        test_size=data_cfg.get("test_size", 0.2),
        random_state=data_cfg.get("random_state", 42),
        stratify=y,
    )

    # modèle
    model_path = MODELS_DIR / f"{args.exp_name}_model.pkl"
    print("🔎 loading model from :", model_path)
    model = joblib.load(model_path)

    # prédiction
    y_pred = model.predict(X_test)
    print("✅ prediction OK")

    # matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=iris.target_names,
        yticklabels=iris.target_names,
    )
    plt.title(f"Confusion Matrix - {args.exp_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    cm_path = REPORTS_DIR / f"{args.exp_name}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"✅ confusion matrix saved to {cm_path}")

    # feature importance si disponible
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        idx = np.argsort(fi)[::-1]
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(fi)), fi[idx])
        plt.xticks(
            range(len(fi)),
            [iris.feature_names[i] for i in idx],
            rotation=45,
        )
        plt.title(f"Feature importance - {args.exp_name}")
        plt.tight_layout()
        fi_path = REPORTS_DIR / f"{args.exp_name}_feature_importance.png"
        plt.savefig(fi_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"✅ feature importance saved to {fi_path}")
    else:
        print("ℹ️ modèle sans feature_importances_ → on saute la 2e figure.")

    print("🎉 Terminé.")


if __name__ == "__main__":
    main()
