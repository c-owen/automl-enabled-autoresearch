"""Trial workpiece — XGBoost baseline.

This is the file the autoresearch LLM rewrites freely (Karpathy-style): swap the
model family, tune hyperparameters, change preprocessing — all fair game. The
only contracts are:
    1. MODEL must be in ALLOWED_FAMILIES.
    2. The script must end with the parseable `---` summary block + END_OF_TRIAL.
    3. Family integrity (protocol §9): the fitted estimator must be the family's
       canonical class (xgboost -> XGBClassifier, random_forest ->
       RandomForestClassifier, logistic_regression -> LogisticRegression, mlp -> a
       torch net). The end-of-script check enforces this — keep it when you rewrite.

Run a trial via the wrapper (records it):  uv run python run_trial.py
(Running `python train.py` directly just prints the summary; nothing is logged.)
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
from logging_lib import family_violation, peak_rss_mb

# ---------------------------------------------------------------------------
# Constants (the LLM tunes these)
# ---------------------------------------------------------------------------

MODEL = "xgboost"

N_ESTIMATORS = 300
MAX_DEPTH = 6
LEARNING_RATE = 0.1
SUBSAMPLE = 0.9
COLSAMPLE_BYTREE = 0.9
REG_LAMBDA = 1.0
REG_ALPHA = 0.0

# ---------------------------------------------------------------------------
# Family guard — fail fast if the chosen family is out of scope.
# ---------------------------------------------------------------------------

if MODEL not in ALLOWED_FAMILIES:
    sys.stderr.write(
        f"ERROR: MODEL={MODEL!r} is not in ALLOWED_FAMILIES={ALLOWED_FAMILIES}\n"
    )
    sys.exit(2)

# ---------------------------------------------------------------------------
# Wall-clock guard (cross-platform; replaces POSIX signal.alarm).
# ---------------------------------------------------------------------------

TIMEOUT_EXIT_CODE = 124


def _on_timeout():
    # os._exit terminates immediately even mid-C-call; flush the sentinel first.
    print("TIMEOUT", flush=True)
    os._exit(TIMEOUT_EXIT_CODE)


def main():
    t_start = time.perf_counter()

    X_train, y_train, X_val, y_val = load_task()

    # Preprocessing: XGBoost handles missing values natively and categoricals
    # via the pandas `category` dtype (enable_categorical=True below).
    X_train = X_train.copy()
    X_val = X_val.copy()
    for col in X_train.select_dtypes(include=["object", "category"]).columns:
        X_train[col] = X_train[col].astype("category")
        X_val[col] = X_val[col].astype("category")

    # XGBoost wants integer-encoded targets; encode consistently for train/val.
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)

    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        reg_lambda=REG_LAMBDA,
        reg_alpha=REG_ALPHA,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="logloss",
    )

    watchdog = threading.Timer(TIME_BUDGET, _on_timeout)
    watchdog.daemon = True
    watchdog.start()
    try:
        t_fit = time.perf_counter()
        model.fit(X_train, y_train_enc)
        train_seconds = time.perf_counter() - t_fit
    finally:
        watchdog.cancel()

    metrics = evaluate(model, X_val, y_val_enc)
    total_seconds = time.perf_counter() - t_start
    n_params = N_ESTIMATORS * MAX_DEPTH  # informational, family-defined

    # Family integrity (protocol §9): the fitted estimator must be the family's
    # canonical class — a substitute (ExtraTrees, HistGradientBoosting, ...) is not
    # allowed even if MODEL names the family. Checked against the live object, not
    # just the MODEL string.
    estimator_class = type(model).__name__
    if family_violation(MODEL, estimator_class):
        sys.stderr.write(
            f"ERROR: MODEL={MODEL!r} requires its canonical estimator, but the "
            f"fitted model is {estimator_class!r} — substitute families are not "
            "allowed (protocol §9).\n"
        )
        sys.exit(2)

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
    print(f"estimator_class: {estimator_class}")
    print("END_OF_TRIAL", flush=True)


if __name__ == "__main__":
    main()
