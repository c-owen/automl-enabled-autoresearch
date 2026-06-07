"""Harness-coverage tests for the MLP (neural) family baseline fixture.

The MLP is the family that can actually overrun, so it also exercises the
wall-clock watchdog: an oversized net with a 1s budget must exit 124.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from logging_lib import parse_summary_block

REPO = Path(__file__).resolve().parents[2]
MLP_FIXTURE = REPO / "tests" / "fixtures" / "family_baselines" / "mlp.py"


def _run_mlp(work_dir, replacements=None, timeout=120):
    src = MLP_FIXTURE.read_text(encoding="utf-8")
    for old, new in replacements or []:
        assert old in src, f"anchor not found in mlp.py: {old!r}"
        src = src.replace(old, new)
    script = Path(work_dir) / "mlp.py"
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


@pytest.mark.integration
def test_train_runs_mlp(task_data_available, tmp_path):
    """A deliberately tiny MLP (1 small layer, 5 epochs) runs end-to-end."""
    result = _run_mlp(
        tmp_path,
        replacements=[
            ("MAX_EPOCHS = 30", "MAX_EPOCHS = 5"),
            ("HIDDEN_SIZES = (128, 64)", "HIDDEN_SIZES = (16,)"),
        ],
    )
    assert result.returncode == 0, (
        f"MLP fixture exited {result.returncode}\nSTDERR:\n{result.stderr}"
    )
    summary = parse_summary_block(result.stdout)
    assert summary["model_family"] == "mlp"
    assert summary["task_name"] == "adult"
    assert summary["val_logloss"] < 0.69
    assert "END_OF_TRIAL" in result.stdout


@pytest.mark.integration
def test_train_mlp_timeout(task_data_available, tmp_path):
    """An oversized MLP with a 1s budget is killed by the watchdog: exit 124."""
    result = _run_mlp(
        tmp_path,
        replacements=[
            ('MODEL = "mlp"', 'MODEL = "mlp"\nTIME_BUDGET = 1'),
            ("MAX_EPOCHS = 30", "MAX_EPOCHS = 100000"),
            ("HIDDEN_SIZES = (128, 64)", "HIDDEN_SIZES = (2048, 2048, 2048)"),
        ],
        timeout=60,
    )
    assert result.returncode == 124, (
        f"expected timeout exit 124, got {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "TIMEOUT" in result.stdout
