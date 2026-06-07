"""Regenerate tests/fixtures/tiny_synthetic.npz.

A 200-row balanced binary classification problem, pre-split 160/40, used by the
fast (no-network) evaluate tests. Deterministic: fixed seed, no randomness
beyond sklearn's generators.

Run:  uv run python tests/fixtures/make_tiny_synthetic.py
"""

import os

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

SEED = 0
OUT = os.path.join(os.path.dirname(__file__), "tiny_synthetic.npz")


def main():
    X, y = make_classification(
        n_samples=200,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        weights=[0.5, 0.5],
        random_state=SEED,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=40, random_state=SEED, stratify=y
    )
    np.savez(
        OUT,
        X_train=X_train.astype(np.float64),
        y_train=y_train.astype(np.int64),
        X_val=X_val.astype(np.float64),
        y_val=y_val.astype(np.int64),
    )
    print(f"wrote {OUT}: train={X_train.shape} val={X_val.shape}")


if __name__ == "__main__":
    main()
