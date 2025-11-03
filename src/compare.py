import json
from pathlib import Path
from typing import List, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "metrics"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_metrics() -> List[Dict]:
    rows = []
    for p in sorted(METRICS_DIR.glob("*.json")):
        with p.open("r") as f:
            rows.append(json.load(f))
    return rows


def plot_accuracy_bar(df: pd.DataFrame) -> Path:
    plt.figure(figsize=(8, 5))
    bars = plt.bar(df["experiment"], df["accuracy"], color="#4C78A8")
    plt.title("Accuracy by experiment")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    for b in bars:
        plt.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.01,
            f"{b.get_height():.3f}",
            ha="center",
        )
    out = REPORTS_DIR / "comparison_accuracy.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def main():
    rows = load_metrics()
    if not rows:
        print("No metrics found to compare.")
        return
    df = pd.DataFrame(rows)
    best = df.sort_values("accuracy", ascending=False).iloc[0]
    out_plot = plot_accuracy_bar(df)

    summary_path = ROOT / "best_model.json"
    with summary_path.open("w") as f:
        json.dump(
            {
                "best_experiment": best["experiment"],
                "model": best["model"],
                "accuracy": float(best["accuracy"]),
                "model_params": best["model_params"],
            },
            f,
            indent=2,
        )

    print(f"[compare] best={best['experiment']} acc={best['accuracy']:.4f}")
    print(f"[compare] plot={out_plot}")


if __name__ == "__main__":
    main()
