import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from src.data import load_esol
from src.features import featurize
from src.splits import get_split
from src.ad_methods import ad_score

SEEDS = [0, 1, 2, 3, 4]
K = 5
Path("figures").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

df = load_esol()
X, valid_idx = featurize(df["smiles"].tolist())
df = df.iloc[valid_idx].reset_index(drop=True)
y = df["y"].values
scaffolds = df["scaffold"].tolist()

rows = []
for method in ["random", "scaffold"]:
    for seed in SEEDS:
        tr, te = get_split(df["smiles"].tolist(), method=method,
                           seed=seed, scaffolds=scaffolds)
        model = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        ad = ad_score(X[te], X[tr], k=K)
        rows.append(pd.DataFrame({
            "split": method, "seed": seed, "test_idx": te,
            "y_true": y[te], "y_pred": pred,
            "abs_error": np.abs(y[te] - pred), "ad_score": ad,
        }))

per = pd.concat(rows, ignore_index=True)
per.to_csv("results/per_compound.csv", index=False)

print("mean AD score by split regime:")
print(per.groupby("split")["ad_score"].agg(["mean", "std", "median"]).round(3))

edges = np.quantile(per["ad_score"], [0, 0.25, 0.5, 0.75, 1.0])
edges[0] -= 1e-9
labels = ["Q1 (least similar)", "Q2", "Q3", "Q4 (most similar)"]
per["ad_quartile"] = pd.cut(per["ad_score"], bins=edges, labels=labels)

def rmse(g):
    return np.sqrt(mean_squared_error(g["y_true"], g["y_pred"]))

binned = (per.groupby(["split", "ad_quartile"], observed=True)
            .apply(lambda g: pd.Series({"rmse": rmse(g), "n": len(g)}))
            .reset_index())
binned.to_csv("results/rmse_by_ad_quartile.csv", index=False)
print("\nRMSE by AD quartile:")
print(binned.to_string(index=False))

fig, ax = plt.subplots(figsize=(6, 4))
for name, color in [("random", "#2a78d6"), ("scaffold", "#eb6834")]:
    ax.hist(per[per["split"] == name]["ad_score"], bins=40, alpha=0.6,
            label=f"{name} split", color=color, density=True)
ax.set_xlabel(f"Applicability domain score (mean Tanimoto to {K} nearest training compounds)")
ax.set_ylabel("Density")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("figures/fig1_ad_distribution.png", dpi=200)

fig, ax = plt.subplots(figsize=(6, 4))
w = 0.38
xpos = np.arange(len(labels))
for i, (name, color) in enumerate([("random", "#2a78d6"), ("scaffold", "#eb6834")]):
    sub = binned[binned["split"] == name].set_index("ad_quartile").reindex(labels)
    ax.bar(xpos + (i - 0.5) * w, sub["rmse"], w, label=f"{name} split", color=color)
ax.set_xticks(xpos)
ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
ax.set_ylabel("Test RMSE (log S)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("figures/fig2_rmse_by_ad_quartile.png", dpi=200)

print("\nfigures written to figures/")
