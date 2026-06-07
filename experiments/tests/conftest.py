"""Shared pytest fixtures for the tabular-port test suite."""

import os

import numpy as np
import pytest

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def tiny_dataset():
    """Load the 200-row synthetic binary dataset (pre-split 160/40).

    Returns ``(X_train, y_train, X_val, y_val)`` as numpy arrays. Fast and
    offline — used by the evaluate tests so they never touch OpenML.
    """
    data = np.load(os.path.join(FIXTURES_DIR, "tiny_synthetic.npz"))
    return data["X_train"], data["y_train"], data["X_val"], data["y_val"]


# Substrings that identify a transient network/host outage (as opposed to a
# real bug). When the data fetch fails for one of these reasons, the
# integration tests skip rather than fail — an external 504 from the dataset
# host should not redden the suite. A 404 / parse error is NOT listed here, so
# a genuinely broken URL still fails loudly.
_INFRA_ERROR_MARKERS = (
    "URLError",
    "ConnectionError",
    "ConnectionResetError",
    "Timeout",
    "Gateway",
    "502",
    "503",
    "504",
)


def _is_infra_error(exc: BaseException) -> bool:
    blob = f"{type(exc).__name__}: {exc}"
    return any(marker in blob for marker in _INFRA_ERROR_MARKERS)


@pytest.fixture(scope="session")
def task_data_available():
    """Warm the task cache once; skip the group if the dataset host is down.

    The fetch only touches the network on a cold cache. If the host is
    unreachable, the whole integration group skips with a clear reason instead
    of failing.
    """
    import prepare

    try:
        prepare.load_task()
    except Exception as exc:  # noqa: BLE001 — narrowed by _is_infra_error
        if _is_infra_error(exc):
            pytest.skip(f"Dataset host unreachable, skipping data-fetch tests: {exc}")
        raise
    return True
