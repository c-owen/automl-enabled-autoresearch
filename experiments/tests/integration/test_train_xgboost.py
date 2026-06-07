"""Integration tests for the shipped XGBoost train.py.

These run train.py as a subprocess. The happy-path tests share one real run of
the unmodified, shipped artifact (via a module fixture); the guard/timeout
tests run modified copies. All depend on ``task_data_available`` so they skip
cleanly if the dataset host is down.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from logging_lib import parse_summary_block

REPO = Path(__file__).resolve().parents[2]


def _run_train(work_dir, replacements=None, timeout=120):
    """Write a (possibly modified) copy of train.py and run it as a subprocess.

    Runs with cwd=REPO and PYTHONPATH=REPO so prepare/logging_lib import
    regardless of where the script copy lives.
    """
    src = (REPO / "train.py").read_text(encoding="utf-8")
    for old, new in replacements or []:
        assert old in src, f"anchor not found in train.py: {old!r}"
        src = src.replace(old, new)
    script = Path(work_dir) / "train.py"
    script.write_text(src, encoding="utf-8")

    env = {**os.environ, "PYTHONPATH": str(REPO)}
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )


@pytest.fixture(scope="module")
def xgb_result(task_data_available, tmp_path_factory):
    """One real run of the shipped train.py, shared by the happy-path tests."""
    work_dir = tmp_path_factory.mktemp("xgb_run")
    result = _run_train(work_dir)
    assert result.returncode == 0, (
        f"train.py exited {result.returncode}\nSTDERR:\n{result.stderr}"
    )
    return result


@pytest.mark.integration
def test_train_runs_xgboost(xgb_result):
    """train.py runs clean and beats chance (logloss < 0.69) on the task."""
    summary = parse_summary_block(xgb_result.stdout)
    assert summary["model_family"] == "xgboost"
    assert summary["val_logloss"] < 0.69
    assert "END_OF_TRIAL" in xgb_result.stdout


@pytest.mark.integration
def test_train_print_contract(xgb_result):
    """All nine print-contract keys are present with the correct types."""
    summary = parse_summary_block(xgb_result.stdout)
    expected = {
        "val_logloss": float, "val_acc": float, "val_auc": float,
        "train_seconds": float, "total_seconds": float, "peak_mem_mb": float,
        "model_family": str, "n_params": int, "task_name": str,
    }
    for key, typ in expected.items():
        assert key in summary, f"missing print-contract key: {key}"
        assert isinstance(summary[key], typ), (
            f"{key}={summary[key]!r} is {type(summary[key]).__name__}, expected {typ.__name__}"
        )
    assert summary["task_name"] == "adult"
    assert 0.0 <= summary["val_acc"] <= 1.0
    assert 0.0 <= summary["val_auc"] <= 1.0


@pytest.mark.integration
def test_train_assertion_blocks_bad_family(tmp_path):
    """A MODEL outside ALLOWED_FAMILIES exits non-zero with a clear message."""
    result = _run_train(
        tmp_path, replacements=[('MODEL = "xgboost"', 'MODEL = "lightgbm"')]
    )
    assert result.returncode != 0
    assert "ALLOWED_FAMILIES" in result.stderr
    assert "lightgbm" in result.stderr


@pytest.mark.integration
def test_train_timeout_killed(task_data_available, tmp_path):
    """A trial exceeding TIME_BUDGET is killed via the watchdog: exit 124."""
    result = _run_train(
        tmp_path,
        replacements=[
            ('MODEL = "xgboost"', 'MODEL = "xgboost"\nTIME_BUDGET = 1'),
            ("N_ESTIMATORS = 300", "N_ESTIMATORS = 100000"),
        ],
        timeout=60,
    )
    assert result.returncode == 124, (
        f"expected timeout exit 124, got {result.returncode}\nSTDOUT:\n{result.stdout}"
    )
    assert "TIMEOUT" in result.stdout
