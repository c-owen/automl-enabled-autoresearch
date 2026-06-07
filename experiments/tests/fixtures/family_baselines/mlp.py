"""MLP (neural) family baseline — a TEST FIXTURE, not a repo-root file.

A minimal PyTorch MLP variant of train.py: nn.Sequential, Adam, cross-entropy,
fixed max epochs with internal early-stopping on a carved-out validation split.
A thin wrapper exposes predict_proba/classes_ so the locked evaluate() scores it
exactly like the sklearn families. Like the others it is NOT in the repo root
and NOT referenced by program.md.

Same print contract as train.py. Run with PYTHONPATH pointed at the repo root.
"""

import copy
import os
import sys
import threading
import time

import numpy as np

from prepare import (
    ALLOWED_FAMILIES,
    RANDOM_SEED,
    TASK_NAME,
    TIME_BUDGET,
    evaluate,
    load_task,
)
from logging_lib import peak_rss_mb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "mlp"

HIDDEN_SIZES = (128, 64)
DROPOUT = 0.1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
MAX_EPOCHS = 30
PATIENCE = 5

# ---------------------------------------------------------------------------
# Family guard
# ---------------------------------------------------------------------------

if MODEL not in ALLOWED_FAMILIES:
    sys.stderr.write(
        f"ERROR: MODEL={MODEL!r} is not in ALLOWED_FAMILIES={ALLOWED_FAMILIES}\n"
    )
    sys.exit(2)

TIMEOUT_EXIT_CODE = 124


def _on_timeout():
    print("TIMEOUT", flush=True)
    os._exit(TIMEOUT_EXIT_CODE)


class _TorchMLPClassifier:
    """Adapter so the locked evaluate() can score the torch net."""

    def __init__(self, net, preprocessor, classes):
        self.net = net
        self.preprocessor = preprocessor
        self.classes_ = classes

    def _forward_proba(self, X):
        import torch

        self.net.eval()
        Xt = np.asarray(self.preprocessor.transform(X), dtype=np.float32)
        with torch.no_grad():
            logits = self.net(torch.from_numpy(Xt))
            return torch.softmax(logits, dim=1).numpy()

    def predict_proba(self, X):
        return self._forward_proba(X)

    def predict(self, X):
        return self.classes_[self._forward_proba(X).argmax(axis=1)]


def _build_net(input_dim, hidden_sizes, dropout, n_classes):
    import torch.nn as nn

    layers = []
    prev = input_dim
    for width in hidden_sizes:
        layers += [nn.Linear(prev, width), nn.ReLU(), nn.Dropout(dropout)]
        prev = width
    layers.append(nn.Linear(prev, n_classes))  # one logit per class
    return nn.Sequential(*layers)


def main():
    import torch
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

    torch.manual_seed(RANDOM_SEED)
    t_start = time.perf_counter()

    X_train, y_train, X_val, y_val = load_task()

    cat_cols = list(X_train.select_dtypes(include=["object", "category"]).columns)
    num_cols = [c for c in X_train.columns if c not in cat_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)

    # Carve an internal validation split for early stopping (no leakage of the
    # authoritative X_val, which only the final evaluate() touches).
    X_fit_raw, X_es_raw, y_fit, y_es = train_test_split(
        X_train, y_train_enc, test_size=0.1,
        random_state=RANDOM_SEED, stratify=y_train_enc,
    )

    X_fit = np.asarray(preprocessor.fit_transform(X_fit_raw), dtype=np.float32)
    X_es = np.asarray(preprocessor.transform(X_es_raw), dtype=np.float32)

    X_fit_t = torch.from_numpy(X_fit)
    y_fit_t = torch.from_numpy(y_fit.astype(np.int64))
    X_es_t = torch.from_numpy(X_es)
    y_es_t = torch.from_numpy(y_es.astype(np.int64))

    net = _build_net(X_fit.shape[1], HIDDEN_SIZES, DROPOUT, len(label_encoder.classes_))
    optimizer = torch.optim.Adam(
        net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    n = X_fit_t.shape[0]
    best_val = float("inf")
    best_state = copy.deepcopy(net.state_dict())
    epochs_without_improvement = 0

    watchdog = threading.Timer(TIME_BUDGET, _on_timeout)
    watchdog.daemon = True
    watchdog.start()
    try:
        t_fit = time.perf_counter()
        for _epoch in range(MAX_EPOCHS):
            net.train()
            perm = torch.randperm(n)
            for start in range(0, n, BATCH_SIZE):
                idx = perm[start:start + BATCH_SIZE]
                optimizer.zero_grad()
                loss = loss_fn(net(X_fit_t[idx]), y_fit_t[idx])
                loss.backward()
                optimizer.step()

            net.eval()
            with torch.no_grad():
                val_loss = loss_fn(net(X_es_t), y_es_t).item()
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_state = copy.deepcopy(net.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= PATIENCE:
                    break
        train_seconds = time.perf_counter() - t_fit
    finally:
        watchdog.cancel()

    net.load_state_dict(best_state)
    model = _TorchMLPClassifier(net, preprocessor, label_encoder.classes_)

    metrics = evaluate(model, X_val, y_val)
    total_seconds = time.perf_counter() - t_start
    n_params = int(sum(p.numel() for p in net.parameters()))

    print("---")
    print(f"val_logloss:    {metrics['val_logloss']:.6f}")
    print(f"val_acc:        {metrics['val_acc']:.6f}")
    print(f"val_auc:        {metrics['val_auc']:.6f}")
    print(f"train_seconds:  {train_seconds:.1f}")
    print(f"total_seconds:  {total_seconds:.1f}")
    print(f"peak_mem_mb:    {peak_rss_mb():.1f}")
    print(f"model_family:   {MODEL}")
    print(f"n_params:       {n_params}")
    print(f"task_name:      {TASK_NAME}")
    print("END_OF_TRIAL", flush=True)


if __name__ == "__main__":
    main()
