import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from src.data import load_esol

per = pd.read_csv("results/per_compound.csv")
df = load_esol()
df["mw"] = [Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in df["smiles"]]
df["heavy"] = [Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in df["smiles"]]

per = per.merge(df[["mw", "heavy"]], left_on="test_idx", right_index=True)
edges = np.quantile(per["ad_score"], [0, .25, .5, .75, 1.0]); edges[0] -= 1e-9
per["q"] = pd.cut(per["ad_score"], bins=edges,
                  labels=["Q1", "Q2", "Q3", "Q4"])

print(per.groupby(["split", "q"], observed=True).agg(
    mw=("mw", "mean"), heavy=("heavy", "mean"),
    y_mean=("y_true", "mean"), y_sd=("y_true", "std"),
    rmse=("abs_error", lambda e: np.sqrt((e**2).mean())),
).round(2).to_string())

print("\nper-seed scaffold RMSE by quartile:")
sc = per[per["split"] == "scaffold"]
print(sc.groupby(["seed", "q"], observed=True)["abs_error"]
        .apply(lambda e: np.sqrt((e**2).mean())).unstack().round(2).to_string())
