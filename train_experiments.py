import pandas as pd
import yaml
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import joblib
import os
from datetime import datetime


def load_config() -> dict:
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def build_model(model_name: str, params: dict):
    if model_name == "random_forest":
        return RandomForestClassifier(**params)
    elif model_name == "svm":
        return SVC(**params)
    elif model_name == "xgboost":
        return XGBClassifier(**params)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def main():
    # Charger la configuration
    cfg = load_config()
    data_cfg = cfg.get("data", {})
    experiments = cfg.get("experiments", {})

    # Charger les donnees
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=data_cfg.get("test_size", 0.25),
        random_state=data_cfg.get("random_state", 42),
        stratify=y,
    )

    # Créer les répertoires
    os.makedirs("experiments", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Entraîner tous les modèles
    results = []

    print("=" * 60)
    print("Entraînement de la matrice d'expériences")
    print("=" * 60)

    for exp_name, exp_cfg in experiments.items():
        print(f"\nEntraînement: {exp_name}")

        model_type = exp_cfg["model"]
        model_params = exp_cfg.get("params", {})

        # Construire et entraîner le modèle
        model = build_model(model_type, model_params)
        model.fit(X_train, y_train)

        # Prédictions
        y_pred = model.predict(X_test)

        # Calculer les métriques
        metrics = {
            "name": exp_name,
            "algorithm": model.__class__.__name__,
            "params": model_params,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="weighted")),
            "recall": float(recall_score(y_test, y_pred, average="weighted")),
            "f1_score": float(f1_score(y_test, y_pred, average="weighted")),
            "test_size": len(X_test),
            "timestamp": datetime.now().isoformat(),
        }

        # Sauvegarder le modèle
        model_path = f"models/{exp_name}.pkl"
        joblib.dump(model, model_path)

        # Sauvegarder les métriques individuelles
        metrics_path = f"experiments/{exp_name}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        results.append(metrics)

        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  Modèle sauvegardé: {model_path}")

    print("\n" + "=" * 60)
    print(f"Entraînement terminé: {len(results)} modèles")
    print("=" * 60)

    # Sauvegarder tous les résultats
    with open("experiments/all_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nRésultats sauvegardés dans experiments/all_results.json")


if __name__ == "__main__":
    main()
