"""Step 10: multi-task switching. Each registered task loads with sane shapes."""

import pytest

import prepare

_INFRA_MARKERS = ("URLError", "ConnectionError", "ConnectionResetError",
                  "Timeout", "Gateway", "502", "503", "504")


def _load_or_skip(task):
    try:
        return prepare.load_task(task)
    except Exception as exc:  # noqa: BLE001
        blob = f"{type(exc).__name__}: {exc}"
        if any(m in blob for m in _INFRA_MARKERS):
            pytest.skip(f"dataset host unreachable for {task}: {exc}")
        raise


@pytest.mark.integration
@pytest.mark.parametrize("task", ["adult", "credit-g", "bank-marketing"])
def test_load_task_each_registered(task):
    X_train, y_train, X_val, y_val = _load_or_skip(task)

    assert len(X_train) > 0 and len(X_val) > 0
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)

    n_total = len(X_train) + len(X_val)
    expected_val = round(n_total * prepare.VAL_FRAC)
    assert abs(len(X_val) - expected_val) <= 1

    assert list(X_train.columns) == list(X_val.columns)
    target_col = prepare._TASK_REGISTRY[task]["target_col"]
    assert target_col not in X_train.columns  # target is split out into y
    assert y_train.nunique() == 2  # binary classification
