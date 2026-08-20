import numpy as np


def tanimoto_matrix(X_test, X_train):
    """Pairwise Tanimoto similarity between binary fingerprint sets."""
    inter = X_test @ X_train.T
    test_bits = X_test.sum(axis=1)[:, None]
    train_bits = X_train.sum(axis=1)[None, :]
    union = test_bits + train_bits - inter
    return np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union > 0)


def ad_score(X_test, X_train, k=5):
    """Mean Tanimoto to the k nearest training compounds. Higher = more familiar."""
    sim = tanimoto_matrix(X_test, X_train)
    k = min(k, sim.shape[1])
    topk = np.partition(sim, -k, axis=1)[:, -k:]
    return topk.mean(axis=1)
