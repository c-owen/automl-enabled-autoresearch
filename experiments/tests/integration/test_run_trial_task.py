"""End-to-end: a session's recorded task actually drives the trial.

Confirms run_trial.py injects AUTORESEARCH_TASK from session.json, so
`start_session --task <name>` works without the human setting the env var.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import prepare

REPO = Path(__file__).resolve().parents[2]
_INFRA = ("URLError", "ConnectionError", "Timeout", "Gateway", "502", "503", "504")


@pytest.mark.integration
def test_run_trial_uses_session_task(tmp_path):
    # Warm the non-default task (or skip if the host is down).
    try:
        prepare.load_task("credit-g")
    except Exception as exc:  # noqa: BLE001
        blob = f"{type(exc).__name__}: {exc}"
        if any(m in blob for m in _INFRA):
            pytest.skip(f"dataset host unreachable: {exc}")
        raise

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "session.json").write_text(
        json.dumps({"task": "credit-g"}), encoding="utf-8"
    )

    env = {k: v for k, v in os.environ.items() if k != "AUTORESEARCH_TASK"}
    env.update({
        "LOGS_DIR": str(logs),
        "RESULTS_TSV": str(tmp_path / "results.tsv"),
        "TRIAL_COMMIT": "tasktest",
        "PYTHONPATH": str(REPO),
    })
    result = subprocess.run(
        [sys.executable, "run_trial.py"],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr

    row = json.loads((logs / "runs.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["task"] == "credit-g"   # the trial trained on the session's task
