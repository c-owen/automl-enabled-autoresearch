"""Logistic Regression family baseline — a TEST FIXTURE, not a repo-root file.

A working LogisticRegression variant of train.py, used only to confirm the
locked harness works for a linear family whose preprocessing surface includes
feature scaling and one-hot encoding. Like the other family baselines it is NOT
in the repo root and NOT referenced by program.md.

Same print contract as train.py. Run with PYTHONPATH pointed at the repo root.
"""

import os
import sys
import threading
import time

from prepare import (
    ALLOWED_FAMILIES,
    TASK_NAME,
    TIME_BUDGET,
    evaluate,
    load_task,
)
from logging_lib import peak_rss_mb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "logistic_regression"

C = 1.0
MAX_ITER = 1000
PENALTY = "l2"
SOLVER = "lbfgs"

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


def main():
    t_start = time.perf_counter()

    X_train, y_train, X_val, y_val = load_task()

    # Linear model: scale numerics, one-hot encode categoricals, impute both.
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    cat_cols = list(X_train.select_dtypes(include=["object", "category"]).columns)
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    preprocess = ColumnTransformer(
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
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    model = Pipeline(
        [
            ("preprocess", preprocess),
            (
                "model",
                LogisticRegression(
                    C=C,
                    max_iter=MAX_ITER,
                    penalty=PENALTY,
                    solver=SOLVER,
                ),
            ),
        ]
    )

    watchdog = threading.Timer(TIME_BUDGET, _on_timeout)
    watchdog.daemon = True
    watchdog.start()
    try:
        t_fit = time.perf_counter()
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - t_fit
    finally:
        watchdog.cancel()

    metrics = evaluate(model, X_val, y_val)
    total_seconds = time.perf_counter() - t_start

    lr = model.named_steps["model"]
    n_params = int(lr.coef_.size + lr.intercept_.size)

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
