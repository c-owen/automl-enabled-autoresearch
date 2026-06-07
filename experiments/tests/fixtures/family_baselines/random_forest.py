"""Random Forest family baseline — a TEST FIXTURE, not a repo-root file.

This is a working RandomForest variant of train.py, used only to confirm the
locked harness (prepare.py, logging_lib.py, the print contract) works for a
non-XGBoost family. It is deliberately NOT in the repo root and NOT referenced
by program.md, so the autoresearch LLM produces family swaps from its own
priors rather than copying this.

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

MODEL = "random_forest"

N_ESTIMATORS = 200
MAX_DEPTH = 16
MIN_SAMPLES_LEAF = 2
MAX_FEATURES = "sqrt"

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

    # sklearn RF can't take NaN or raw categoricals: impute + one-hot encode.
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    cat_cols = list(X_train.select_dtypes(include=["object", "category"]).columns)
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
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
                RandomForestClassifier(
                    n_estimators=N_ESTIMATORS,
                    max_depth=MAX_DEPTH,
                    min_samples_leaf=MIN_SAMPLES_LEAF,
                    max_features=MAX_FEATURES,
                    n_jobs=-1,
                    random_state=0,
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

    forest = model.named_steps["model"]
    n_params = int(sum(est.tree_.node_count for est in forest.estimators_))

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
