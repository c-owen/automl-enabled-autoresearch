"""Harness-coverage test for the Logistic Regression family baseline fixture.

Exercises the scaling + one-hot preprocessing surface and confirms evaluate()
handles a fitted sklearn Pipeline ending in LogisticRegression.
"""

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from logging_lib import parse_summary_block

REPO = Path(__file__).resolve().parents[2]
LR_FIXTURE = REPO / "tests" / "fixtures" / "family_baselines" / "logistic_regression.py"


@pytest.mark.integration
def test_train_runs_logistic_regression(task_data_available):
    """The LR fixture runs end-to-end and produces a finite val_logloss."""
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    result = subprocess.run(
        [sys.executable, str(LR_FIXTURE)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"LR fixture exited {result.returncode}\nSTDERR:\n{result.stderr}"
    )

    summary = parse_summary_block(result.stdout)
    assert summary["model_family"] == "logistic_regression"
    assert summary["task_name"] == "adult"
    assert math.isfinite(summary["val_logloss"])
    assert summary["val_logloss"] < 0.69
    assert "END_OF_TRIAL" in result.stdout
