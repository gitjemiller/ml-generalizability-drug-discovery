import pandas as pd
from src.splits import get_scaffolds

df = pd.read_csv("data/delaney-processed.csv")
print("shape:", df.shape)
print("columns:", df.columns.tolist())

sc = get_scaffolds(df["smiles"])
n_acyclic = sum(s == "" for s in sc)
n_unique = len(set(s for s in sc if s != ""))
print(f"acyclic: {n_acyclic} of {len(sc)}")
print(f"unique ring scaffolds: {n_unique}")
