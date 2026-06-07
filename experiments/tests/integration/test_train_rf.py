"""Harness-coverage test for the Random Forest family baseline fixture.

Confirms prepare.py/evaluate() and the print contract work for a sklearn
Pipeline-based RandomForest, not just XGBoost. The fixture lives under
tests/fixtures/family_baselines/ and is run as a subprocess.
"""

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from logging_lib import parse_summary_block

REPO = Path(__file__).resolve().parents[2]
RF_FIXTURE = REPO / "tests" / "fixtures" / "family_baselines" / "random_forest.py"


@pytest.mark.integration
def test_train_runs_random_forest(task_data_available):
    """The RF fixture runs end-to-end and produces a finite val_logloss."""
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    result = subprocess.run(
        [sys.executable, str(RF_FIXTURE)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"RF fixture exited {result.returncode}\nSTDERR:\n{result.stderr}"
    )

    summary = parse_summary_block(result.stdout)
    assert summary["model_family"] == "random_forest"
    assert summary["task_name"] == "adult"
    assert math.isfinite(summary["val_logloss"])
    assert summary["val_logloss"] < 0.69  # better than chance
    assert math.isfinite(summary["val_auc"])
    assert "END_OF_TRIAL" in result.stdout
