import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def featurize(smiles_list, radius=2, n_bits=2048):
    """Morgan fingerprints (ECFP4). Returns (X, valid_idx)."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps, valid_idx = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fps.append(np.array(gen.GetFingerprint(mol)))
        valid_idx.append(i)
    return np.array(fps), np.array(valid_idx)
