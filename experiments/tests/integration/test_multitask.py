"""Every registered task loads offline with its pinned shape + class count, and
``evaluate()`` produces finite multiclass metrics on the new 3-/9-class tasks.

(Step 3: registry + multiclass coverage; protocol §4, gate 1.)
"""

import math

import pytest
from sklearn.ensemble import RandomForestClassifier

import prepare

# (task, n_classes, n_total_rows). The split is pinned by RANDOM_SEED / VAL_FRAC,
# so the row count is exact (±0) — a change here means the data changed.
_EXPECTED = [
    ("adult", 2, 32561),
    ("credit-g", 2, 1000),
    ("bank-marketing", 2, 4521),
    ("electricity", 2, 45312),
    ("balance-scale", 3, 625),
    ("cnae-9", 9, 1080),
]


@pytest.mark.integration
@pytest.mark.parametrize("task,n_classes,n_total", _EXPECTED)
def test_load_task_each_registered(task, n_classes, n_total):
    X_train, y_train, X_val, y_val = prepare.load_task(task)

    assert len(X_train) > 0 and len(X_val) > 0
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)

    # Exact pinned row count.
    assert len(X_train) + len(X_val) == n_total

    # Validation fraction honored (within stratification rounding).
    expected_val = round(n_total * prepare.VAL_FRAC)
    assert abs(len(X_val) - expected_val) <= 1

    assert list(X_train.columns) == list(X_val.columns)
    target_col = prepare._TASK_REGISTRY[task]["target_col"]
    assert target_col not in X_train.columns  # target is split out into y

    # Expected class count, present in both splits (stratified).
    assert y_train.nunique() == n_classes
    assert y_val.nunique() == n_classes


@pytest.mark.integration
@pytest.mark.parametrize("task", ["balance-scale", "cnae-9"])
def test_multiclass_evaluate(task):
    """evaluate() returns finite logloss/acc/macro-OVR AUC on a multiclass task."""
    X_train, y_train, X_val, y_val = prepare.load_task(task)

    # A trivial but proper-probability multiclass classifier.
    model = RandomForestClassifier(n_estimators=25, random_state=0)
    model.fit(X_train, y_train)

    scores = prepare.evaluate(model, X_val, y_val)
    assert set(scores) == {"val_logloss", "val_acc", "val_auc"}
    assert math.isfinite(scores["val_logloss"])
    assert math.isfinite(scores["val_acc"])
    assert math.isfinite(scores["val_auc"])  # macro one-vs-rest AUC
