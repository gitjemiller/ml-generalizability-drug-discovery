import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

TARGET_COL = "measured log solubility in mols per litre"
# NOTE: 'ESOL predicted log solubility...' is Delaney's own model output.
# It is NOT the target and must never enter X.


def load_esol(path="data/delaney-processed.csv", drop_acyclic=True):
    """Load ESOL. Returns df with columns: smiles, y, scaffold."""
    df = pd.read_csv(path)
    df = df[["smiles", TARGET_COL]].rename(columns={TARGET_COL: "y"})

    scaffolds = []
    for smi in df["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scaffolds.append(None)
            continue
        try:
            scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False))
        except Exception:
            scaffolds.append(None)
    df["scaffold"] = scaffolds

    df = df[df["scaffold"].notnull()]
    if drop_acyclic:
        df = df[df["scaffold"] != ""]

    return df.reset_index(drop=True)
