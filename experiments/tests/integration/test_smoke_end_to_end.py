"""The full end-to-end mini-grid (C0 + C1 + R1). Off by default; run with -m smoke.

Exercises every layer at once: agent trials through run_trial.py, a sealed BO
episode through run_bo.py, a scripted reference through run_reference.py, the
v2-aware decision parser, and notebook execution.
"""

from pathlib import Path

import pytest

from tools.extract_decisions import extract_decisions
from tools.smoke_test import run_smoke
from tools.validate_jsonl import validate_jsonl


@pytest.mark.smoke
def test_smoke_end_to_end(tmp_path):
    r = run_smoke(tmp_path)

    # C0: three agent trials with the mid-session family swap.
    assert len(r["c0_runs"]) == 3
    assert [t["model_family"] for t in r["c0_runs"]] == \
        ["xgboost", "xgboost", "random_forest"]
    c0_logs = Path(r["work_dir"]) / "c0" / "logs"
    df_c0 = extract_decisions(c0_logs)
    assert list(df_c0["family_changed_from_prior"]) == [False, False, True]

    # C1: agent trials + a budget-5 BO episode (5 source="bo" rows, one summary).
    assert r["c1_bo_trials"] == 5
    bo_rows = [t for t in r["c1_runs"] if t.get("source") == "bo"]
    assert {t["bo_episode_id"] for t in bo_rows} == {"bo-ep001"}
    c1_logs = Path(r["work_dir"]) / "c1" / "logs"
    assert validate_jsonl(c1_logs / "runs.jsonl") == []
    tsv = (c1_logs.parent / "results.tsv").read_text(encoding="utf-8")
    assert "bo_episode" in tsv

    # R1: five scripted reference trials.
    assert len(r["r1_runs"]) == 5
    assert all(t["source"] == "reference" for t in r["r1_runs"])

    assert r["notebook_ok"] is True
