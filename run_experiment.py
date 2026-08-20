import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from src.data import load_esol
from src.features import featurize
from src.splits import get_split

SEEDS = [0, 1, 2, 3, 4]
N_ESTIMATORS = 100
MAX_DEPTH = None
RADIUS = 2
N_BITS = 2048

Path("results").mkdir(exist_ok=True)

df = load_esol()
X, valid_idx = featurize(df["smiles"].tolist(), radius=RADIUS, n_bits=N_BITS)
df = df.iloc[valid_idx].reset_index(drop=True)
y = df["y"].values
scaffolds = df["scaffold"].tolist()
print(f"featurized: X={X.shape}, y={y.shape}")

def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))

rows = []
splits_record = []

for method in ["random", "scaffold"]:
    for seed in SEEDS:
        tr, te = get_split(df["smiles"].tolist(), method=method,
                           seed=seed, scaffolds=scaffolds)
        splits_record.append(pd.DataFrame({
            "method": method, "seed": seed,
            "idx": np.concatenate([tr, te]),
            "fold": ["train"] * len(tr) + ["test"] * len(te),
        }))

        models = {
            "RandomForest": RandomForestRegressor(
                n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                random_state=seed, n_jobs=-1),
            "XGBoost": XGBRegressor(
                n_estimators=N_ESTIMATORS, max_depth=6,
                random_state=seed, n_jobs=-1, verbosity=0),
        }

        for name, model in models.items():
            model.fit(X[tr], y[tr])
            tr_rmse = rmse(y[tr], model.predict(X[tr]))
            te_rmse = rmse(y[te], model.predict(X[te]))
            rows.append({
                "split": method, "model": name, "seed": seed,
                "n_train": len(tr), "n_test": len(te),
                "train_rmse": tr_rmse, "test_rmse": te_rmse,
                "gap": te_rmse - tr_rmse,
                "n_estimators": N_ESTIMATORS, "radius": RADIUS, "n_bits": N_BITS,
            })
            print(f"{method:9s} {name:13s} seed {seed}  test RMSE {te_rmse:.3f}")

res = pd.DataFrame(rows)
res.to_csv("results/raw_results.csv", index=False)
pd.concat(splits_record).to_csv("results/split_indices.csv", index=False)

summary = (res.groupby(["split", "model"])
             .agg(test_rmse_mean=("test_rmse", "mean"),
                  test_rmse_sd=("test_rmse", "std"),
                  train_rmse_mean=("train_rmse", "mean"),
                  gap_mean=("gap", "mean"))
             .round(3).reset_index())
summary.to_csv("results/summary.csv", index=False)
print("\n", summary.to_string(index=False))
