"""Integration tests for load_task — these hit the dataset host on a cold cache.

They depend on the ``task_data_available`` fixture, which warms the cache once
and skips the whole group with a clear reason if the host is unreachable.
"""

import pandas as pd
import pytest

import prepare


@pytest.mark.integration
def test_load_task_shapes(task_data_available):
    """Row counts honor VAL_FRAC (within rounding) and columns are aligned."""
    X_train, y_train, X_val, y_val = prepare.load_task()

    n_total = len(X_train) + len(X_val)
    expected_val = round(n_total * prepare.VAL_FRAC)
    assert abs(len(X_val) - expected_val) <= 1
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert list(X_train.columns) == list(X_val.columns)


@pytest.mark.integration
def test_load_task_deterministic(task_data_available):
    """Two calls return identical splits (pinned seed + cached source)."""
    X1, y1, Xv1, yv1 = prepare.load_task()
    X2, y2, Xv2, yv2 = prepare.load_task()
    pd.testing.assert_frame_equal(X1, X2)
    pd.testing.assert_frame_equal(Xv1, Xv2)
    pd.testing.assert_series_equal(y1, y2)
    pd.testing.assert_series_equal(yv1, yv2)


@pytest.mark.integration
def test_load_task_caches(task_data_available, monkeypatch):
    """After the first fetch, load_task serves from cache without downloading."""

    def _boom(task):
        raise AssertionError("network fetch attempted on a warm cache")

    # Cache is already warm (fixture). Any download attempt now fails loudly;
    # a cache hit stays silent.
    monkeypatch.setattr(prepare, "_download_task", _boom)
    X_train, _, X_val, _ = prepare.load_task()
    assert len(X_train) > 0 and len(X_val) > 0
