"""Locked harness for the tabular-classification autoresearch port.

This file is the *fixed* part of the experiment — the LLM does not modify it.
It supplies clean train/val arrays (``load_task``) and the authoritative
scoring function (``evaluate``). The mutable workpiece lives in ``train.py``.

Usage:
    python -c "from prepare import load_task; print([a.shape for a in load_task()])"

Task data is fetched from a pinned source (UCI) on the first call, cached under
~/.cache/autoresearch_tabular/<task>/, and served offline thereafter.
"""

import io
import os
import pickle
import urllib.request

import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

RANDOM_SEED = 42                # seed for the train/val split (pinned)
VAL_FRAC = 0.2                  # fraction of rows held out for validation
TIME_BUDGET = 300              # per-trial wall-clock cap in seconds (5 minutes)
TRIAL_BUDGET = 50              # total trials per search session

ALLOWED_FAMILIES = ["xgboost", "random_forest", "logistic_regression", "mlp"]

DEFAULT_TASK = "adult"

# The active task. Overridable per session via the AUTORESEARCH_TASK env var so
# the harness need not be forked to switch datasets. Read at import so train.py
# (run as a subprocess that inherits the env) prints the right task_name.
TASK_NAME = os.environ.get("AUTORESEARCH_TASK", DEFAULT_TASK)

# Column names for the headerless UCI Adult data file. Target is `income`.
_ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]

# Column names for the headerless UCI German Credit data file (statlog/german).
_GERMAN_COLUMNS = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment", "installment_commitment", "personal_status",
    "other_parties", "residence_since", "property_magnitude", "age",
    "other_payment_plans", "housing", "existing_credits", "job",
    "num_dependents", "own_telephone", "foreign_worker", "class",
]

# Task registry. Each entry is a *pinned, host-agnostic* data source described
# by a config dict (direct URL + parse options) so load_task is reproducible and
# not coupled to any one dataset platform (no OpenML). After the first fetch the
# data is cached locally and never re-downloaded. The three starter tasks vary
# deliberately in size, feature count, and class balance.
#
# Per-entry fields: url, target_col, and optional sep / header / columns /
# na_values / archive ("zip") / member (zip entry) / target_map (value remap).
_TASK_REGISTRY = {
    # ~32.5k rows, 14 features (8 cat / 6 num), ~24% positive. Comma CSV.
    "adult": {
        "url": (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "adult/adult.data"
        ),
        "header": None,
        "columns": _ADULT_COLUMNS,
        "target_col": "income",
        "na_values": "?",
    },
    # 1k rows, 20 features (13 cat / 7 num), 30% positive. Whitespace, target 1/2.
    "credit-g": {
        "url": (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "statlog/german/german.data"
        ),
        "sep": r"\s+",
        "header": None,
        "columns": _GERMAN_COLUMNS,
        "target_col": "class",
        "target_map": {1: "good", 2: "bad"},
    },
    # 4.5k rows, 16 features (10 cat / 6 num), ~11.5% positive. Zip -> ; CSV.
    "bank-marketing": {
        "url": (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "00222/bank.zip"
        ),
        "archive": "zip",
        "member": "bank.csv",
        "sep": ";",
        "header": 0,
        "target_col": "y",
    },
}

# ---------------------------------------------------------------------------
# Cache configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "autoresearch_tabular"
)


def _task_cache_dir(task: str) -> str:
    return os.path.join(CACHE_DIR, task)


def _download_task(task: str):
    """Fetch a task's full (X, y) from its pinned source and persist to cache.

    Network is only touched here, on a cold cache. Returns (X, y) as a pandas
    DataFrame and Series. Kept as a separate helper so tests can assert the
    cached path never reaches the network.
    """
    entry = _TASK_REGISTRY[task]

    req = urllib.request.Request(
        entry["url"], headers={"User-Agent": "autoresearch/0.1"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw_bytes = resp.read()

    if entry.get("archive") == "zip":
        import zipfile

        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            buffer = io.BytesIO(zf.read(entry["member"]))
    else:
        buffer = io.BytesIO(raw_bytes)

    frame = pd.read_csv(
        buffer,
        sep=entry.get("sep", ","),
        header=entry.get("header", "infer"),
        names=entry.get("columns"),
        skipinitialspace=True,
        na_values=entry.get("na_values"),
    )

    target = entry["target_col"]
    y = frame[target]
    if entry.get("target_map"):
        y = y.map(entry["target_map"])
    X = frame.drop(columns=[target])

    cache_dir = _task_cache_dir(task)
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, "Xy.pkl"), "wb") as fh:
        pickle.dump((X, y), fh)
    return X, y


def _load_cached_task(task: str):
    """Return cached (X, y) for ``task``, downloading on a cache miss."""
    cache_path = os.path.join(_task_cache_dir(task), "Xy.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)
    return _download_task(task)


def load_task(task: str = None):
    """Load ``task`` and return a pinned ``(X_train, y_train, X_val, y_val)`` split.

    X parts are pandas DataFrames (mixed feature types and missing values
    preserved — the LLM decides how to encode/impute them); y parts are pandas
    Series. The split is stratified and pinned by ``RANDOM_SEED`` / ``VAL_FRAC``
    so it is identical across calls and across trials.

    ``task`` defaults to ``TASK_NAME`` (which honors the AUTORESEARCH_TASK env
    var); pass an explicit name to override.
    """
    if task is None:
        task = TASK_NAME
    if task not in _TASK_REGISTRY:
        raise ValueError(
            f"Unknown task {task!r}. Registered tasks: {sorted(_TASK_REGISTRY)}"
        )
    X, y = _load_cached_task(task)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_FRAC, random_state=RANDOM_SEED, stratify=y
    )
    return X_train, y_train, X_val, y_val


def evaluate(model, X_val, y_val) -> dict:
    """Score a fitted classifier on the validation split.

    Returns ``{val_logloss, val_acc, val_auc}``. ``val_logloss`` is the
    selection metric. Handles **binary and multiclass** classification. Raises
    ``ValueError`` if the model cannot produce class probabilities (the print
    contract requires a probabilistic score).

    For binary tasks ``val_auc`` is the ROC AUC of the positive class; for
    multiclass it is the macro one-vs-rest ROC AUC. If AUC can't be computed
    (e.g. a class missing from the val split) it is reported as NaN.
    """
    if not hasattr(model, "predict_proba"):
        raise ValueError(
            f"Model {type(model).__name__} has no predict_proba; evaluate() "
            "requires a classifier that outputs class probabilities."
        )

    proba = model.predict_proba(X_val)
    classes = list(getattr(model, "classes_", []))
    if len(classes) < 2:
        raise ValueError(
            f"evaluate() needs at least 2 classes; got {len(classes)}: {classes!r}"
        )

    # log_loss with explicit labels so column order matches `classes`.
    val_logloss = float(log_loss(y_val, proba, labels=classes))
    val_acc = float(accuracy_score(y_val, model.predict(X_val)))
    val_auc = _val_auc(y_val, proba, classes)

    return {"val_logloss": val_logloss, "val_acc": val_acc, "val_auc": val_auc}


def _val_auc(y_val, proba, classes) -> float:
    """ROC AUC: positive-class AUC for binary, macro one-vs-rest for multiclass."""
    try:
        if len(classes) == 2:
            y_true_pos = (pd.Series(list(y_val)) == classes[1]).astype(int)
            return float(roc_auc_score(y_true_pos, proba[:, 1]))
        return float(
            roc_auc_score(y_val, proba, multi_class="ovr",
                          average="macro", labels=classes)
        )
    except (ValueError, IndexError):
        return float("nan")
