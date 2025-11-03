import argparse
import json
from pathlib import Path

import yaml
import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_config(root: Path) -> dict:
    cfg_path = root / "params.yaml"
    with cfg_path.open("r") as f:
        return yaml.safe_load(f)


def build_model(model_name: str, params: dict):
    if model_name == "random_forest":
        return RandomForestClassifier(**params)
    elif model_name == "logistic_regression":
        # Map keys that differ from sklearn defaults if needed
        return LogisticRegression(**params)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Train Iris model using params.yaml")
    parser.add_argument(
        "--exp-name",
        required=True,
        help="Experiment name defined under experiments in params.yaml",
    )
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    MODELS_DIR = ROOT / "models"
    METRICS_DIR = ROOT / "metrics"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_config(ROOT)
    data_cfg = cfg.get("data", {})
    exps = cfg.get("experiments", {})
    if args.exp_name not in exps:
        raise SystemExit(
            f"Experiment '{args.exp_name}' not found. Available: {list(exps.keys())}"
        )

    exp_cfg = exps[args.exp_name]
    model_name = exp_cfg["model"]
    model_params = exp_cfg.get("params", {})

    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=data_cfg.get("test_size", 0.2),
        random_state=data_cfg.get("random_state", 42),
        stratify=y,
    )

    model = build_model(model_name, model_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    model_path = MODELS_DIR / f"{args.exp_name}_model.pkl"
    joblib.dump(model, model_path)

    # Persist metrics with full config context for traceability
    metrics = {
        "experiment": args.exp_name,
        "model": model_name,
        "model_params": model_params,
        "data_params": data_cfg,
        "accuracy": float(accuracy),
        "test_size_samples": int(len(X_test)),
    }

    metrics_path = METRICS_DIR / f"{args.exp_name}.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[train] exp={args.exp_name} model={model_name} accuracy={accuracy:.4f}")


if __name__ == "__main__":
    main()
