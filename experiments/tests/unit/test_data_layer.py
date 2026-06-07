"""Step 2: the committed, offline-only data layer.

``prepare.load_task`` reads exclusively from ``data/`` and never opens a socket;
every committed file matches the SHA256 pinned in the registry; a missing file is
a clear operator error pointing at ``tools/fetch_datasets.py``.
"""

import hashlib
import os
import socket

import pytest

import prepare

_ALL_TASKS = sorted(prepare._TASK_REGISTRY)


@pytest.mark.unit
@pytest.mark.parametrize("task", _ALL_TASKS)
def test_load_task_offline(task, monkeypatch):
    """load_task succeeds for every registered task with sockets disabled."""

    def _no_network(*args, **kwargs):
        raise AssertionError(
            "network access attempted during load_task — the experiments harness "
            "must load strictly from data/"
        )

    # Block socket creation outright: any attempt to reach the network raises.
    monkeypatch.setattr(socket, "socket", _no_network)

    X_train, y_train, X_val, y_val = prepare.load_task(task)
    assert len(X_train) > 0 and len(X_val) > 0
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)


@pytest.mark.unit
@pytest.mark.parametrize("task", _ALL_TASKS)
def test_data_checksums(task):
    """Every committed file's bytes match its registry SHA256."""
    files = prepare._TASK_REGISTRY[task].get("files")
    assert files, f"task {task!r} has no `files` registry entry"
    for filename, expected in files.items():
        path = os.path.join(prepare.DATA_DIR, task, filename)
        assert os.path.exists(path), (
            f"committed data file missing: {path} — run tools/fetch_datasets.py"
        )
        with open(path, "rb") as fh:
            actual = hashlib.sha256(fh.read()).hexdigest()
        assert actual == expected, (
            f"{task}/{filename} checksum mismatch: registry={expected} actual={actual}"
        )


@pytest.mark.unit
def test_missing_data_clear_error(tmp_path, monkeypatch):
    """A missing data file raises a clear error naming the fetch script."""
    # Point the data layer at an empty directory: the file no longer exists.
    monkeypatch.setattr(prepare, "DATA_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match=r"fetch_datasets\.py"):
        prepare.load_task("adult")
