import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split


def get_scaffolds(smiles_list):
    """Bemis-Murcko scaffold SMILES. Empty string for acyclic molecules."""
    out = []
    for smi in smiles_list:
        try:
            out.append(MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=False))
        except Exception:
            out.append("")
    return out


def get_split(smiles_list, method="random", seed=0, test_size=0.2, scaffolds=None):
    """Returns (train_idx, test_idx). Same interface for every method.

    Scaffold split uses randomized greedy fill: scaffold groups are shuffled,
    then assigned to test until the target test fraction is reached. This keeps
    test-set size stable across seeds while varying which scaffolds are held out.
    """
    n = len(smiles_list)
    idx = np.arange(n)

    if method == "random":
        return train_test_split(idx, test_size=test_size, random_state=seed)

    if method == "scaffold":
        if scaffolds is None:
            scaffolds = get_scaffolds(smiles_list)

        groups = defaultdict(list)
        for i, s in enumerate(scaffolds):
            groups[s].append(i)

        keys = list(groups.keys())
        rng = np.random.default_rng(seed)
        rng.shuffle(keys)

        n_test_target = int(round(test_size * n))
        test_idx, train_idx = [], []
        for k in keys:
            if len(test_idx) < n_test_target:
                test_idx.extend(groups[k])
            else:
                train_idx.extend(groups[k])

        return np.array(sorted(train_idx)), np.array(sorted(test_idx))

    raise ValueError(f"unknown method: {method}")
