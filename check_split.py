from src.data import load_esol
from src.splits import get_split

df = load_esol()
print(f"n = {len(df)}, unique scaffolds = {df['scaffold'].nunique()}")
print(f"y range: {df['y'].min():.2f} to {df['y'].max():.2f}")

for seed in range(3):
    tr, te = get_split(df["smiles"].tolist(), method="scaffold", seed=seed)
    tr_sc = set(df.loc[tr, "scaffold"])
    te_sc = set(df.loc[te, "scaffold"])
    overlap = tr_sc & te_sc
    print(f"seed {seed}: train {len(tr)}, test {len(te)}, scaffold overlap {len(overlap)}")

print("\nlargest scaffold groups:")
print(df["scaffold"].value_counts().head(5))
