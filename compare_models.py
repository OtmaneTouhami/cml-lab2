import json
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Charger tous les résultats
with open("experiments/all_results.json", "r") as f:
    results = json.load(f)

# Créer un DataFrame pour faciliter l'analyse
df = pd.DataFrame(results)

# Trier par accuracy décroissante
df_sorted = df.sort_values("accuracy", ascending=False)

print("=" * 70)
print("COMPARAISON DES MODÈLES")
print("=" * 70)

# Afficher le tableau de comparaison
print("\nTableau de comparaison (trié par accuracy):\n")
print(
    df_sorted[
        ["name", "algorithm", "accuracy", "f1_score", "precision", "recall"]
    ].to_string(index=False)
)

# Identifier le meilleur modèle
best_model = df_sorted.iloc[0]
print("\n" + "=" * 70)
print("MEILLEUR MODÈLE")
print("=" * 70)
print(f"Nom: {best_model['name']}")
print(f"Algorithme: {best_model['algorithm']}")
print(f"Paramètres: {best_model['params']}")
print(f"Accuracy: {best_model['accuracy']:.4f}")
print(f"F1-Score: {best_model['f1_score']:.4f}")
print(f"Precision: {best_model['precision']:.4f}")
print(f"Recall: {best_model['recall']:.4f}")

# Sauvegarder le meilleur modèle
best_model_info = {
    "best_model": best_model["name"],
    "algorithm": best_model["algorithm"],
    "params": best_model["params"],
    "accuracy": best_model["accuracy"],
    "f1_score": best_model["f1_score"],
    "precision": best_model["precision"],
    "recall": best_model["recall"],
}

with open("experiments/best_model.json", "w") as f:
    json.dump(best_model_info, f, indent=2)

print("\nMeilleur modèle sauvegardé dans experiments/best_model.json")

# Créer le répertoire reports s'il n'existe pas
os.makedirs("reports", exist_ok=True)

# Visualisation 1: Comparaison des accuracy
plt.figure(figsize=(12, 6))
sns.barplot(data=df_sorted, x="name", y="accuracy", palette="viridis")
plt.title("Comparaison des Accuracy par Modèle", fontsize=14, fontweight="bold")
plt.xlabel("Modèle", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.ylim(0.85, 1.01)
plt.tight_layout()
plt.savefig("reports/accuracy_comparison.png", dpi=120, bbox_inches="tight")
plt.close()

# Visualisation 2: Comparaison de toutes les métriques
metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]
df_melted = df.melt(
    id_vars=["name"], value_vars=metrics_to_plot, var_name="metric", value_name="score"
)

plt.figure(figsize=(14, 7))
sns.barplot(data=df_melted, x="name", y="score", hue="metric", palette="Set2")
plt.title("Comparaison de Toutes les Métriques", fontsize=14, fontweight="bold")
plt.xlabel("Modèle", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.legend(title="Métrique", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.ylim(0.85, 1.01)
plt.tight_layout()
plt.savefig("reports/all_metrics_comparison.png", dpi=120, bbox_inches="tight")
plt.close()

# Visualisation 3: Heatmap des performances
df_heatmap = df_sorted[
    ["name", "accuracy", "precision", "recall", "f1_score"]
].set_index("name")
plt.figure(figsize=(10, 8))
sns.heatmap(
    df_heatmap.T,
    annot=True,
    fmt=".4f",
    cmap="YlGnBu",
    cbar_kws={"label": "Score"},
    linewidths=0.5,
)
plt.title("Heatmap des Performances", fontsize=14, fontweight="bold")
plt.xlabel("Modèle", fontsize=12)
plt.ylabel("Métrique", fontsize=12)
plt.tight_layout()
plt.savefig("reports/performance_heatmap.png", dpi=120, bbox_inches="tight")
plt.close()

print("\nVisualisations générées:")
print("  - reports/accuracy_comparison.png")
print("  - reports/all_metrics_comparison.png")
print("  - reports/performance_heatmap.png")

print("\n" + "=" * 70)
print("Comparaison terminée!")
print("=" * 70)
