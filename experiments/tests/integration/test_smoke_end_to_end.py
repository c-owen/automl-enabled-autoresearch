"""The full end-to-end mini-session. Off by default; run with `-m smoke`.

Exercises every layer at once: three real trials through run_trial.py (with
faked decisions), the decision parser, and notebook execution.
"""

import json
from pathlib import Path

import pytest

from tools.smoke_test import run_smoke


@pytest.mark.smoke
def test_smoke_end_to_end(tmp_path):
    result = run_smoke(tmp_path)

    assert result["n_runs"] == 3
    assert result["n_decisions"] == 3
    assert result["notebook_ok"] is True

    logs = Path(result["work_dir"]) / "logs"
    runs = [json.loads(l) for l in (logs / "runs.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    families = [r["model_family"] for r in runs]
    assert families == ["xgboost", "xgboost", "random_forest"]

    # The decision parser sees the mid-session family swap.
    from tools.extract_decisions import extract_decisions
    df = extract_decisions(logs)
    assert list(df["family_changed_from_prior"]) == [False, False, True]
