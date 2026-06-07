"""Step 6: the v2 ledger — BO-episode trial rows + the single episode summary.

Covers the protocol §3.5 logging contract: agent and bo rows both validate, v1
rows still validate (forward-compat), a bo trial touches only runs.jsonl, the
episode summary is exactly one results.tsv row with status=bo_episode, and the
existing extract_decisions parser doesn't crash on a v2 ledger.
"""

import json

import pytest

from logging_lib import (
    build_bo_run_row,
    build_run_row,
    record_bo_episode_summary,
    record_bo_trial,
)
from tools.extract_decisions import extract_decisions
from tools.validate_jsonl import validate_jsonl, validate_run_row


def _summary(**over):
    s = {
        "val_logloss": 0.30, "val_acc": 0.85, "val_auc": 0.90,
        "train_seconds": 1.0, "total_seconds": 1.2, "peak_mem_mb": 100.0,
        "model_family": "xgboost", "n_params": 10, "task_name": "credit-g",
    }
    s.update(over)
    return s


def _v1_row(trial_id=1):
    return {
        "schema_version": 1, "trial_id": trial_id, "commit": f"c{trial_id}",
        "timestamp": "2026-01-01T00:00:00+00:00", "task": "adult",
        "model_family": "xgboost", "hyperparameters": {},
        "val_logloss": 0.3, "val_acc": 0.8, "val_auc": 0.9,
        "train_seconds": 1.0, "total_seconds": 1.0, "peak_mem_mb": 1.0,
        "status": "keep", "description": "x",
    }


@pytest.mark.unit
def test_v2_schema_roundtrip(tmp_path):
    ts = "2026-01-01T00:00:00+00:00"

    agent = build_run_row("c1", _summary(), "keep", "d",
                          {"n_estimators": 100}, 1, ts)
    assert agent["schema_version"] == 2
    assert agent["source"] == "agent"
    assert agent["bo_episode_id"] is None and agent["bo_trial_index"] is None
    assert validate_run_row(agent) == []

    bo = build_bo_run_row("c1", _summary(), {"max_depth": 4}, "ep-1", 3, 2, ts)
    assert bo["source"] == "bo" and bo["status"] == "bo_trial"
    assert bo["bo_episode_id"] == "ep-1" and bo["bo_trial_index"] == 3
    assert validate_run_row(bo) == []

    # v1 rows from archived sessions remain valid.
    assert validate_run_row(_v1_row()) == []

    # A mixed v1 + v2-agent + v2-bo ledger validates end to end.
    path = tmp_path / "runs.jsonl"
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in [_v1_row(), agent, bo]),
        encoding="utf-8",
    )
    assert validate_jsonl(path) == []


@pytest.mark.unit
def test_record_bo_trial_writes_only_jsonl(tmp_path):
    logs = tmp_path / "logs"
    tsv = tmp_path / "results.tsv"

    row = record_bo_trial("c1", _summary(), {"max_depth": 4},
                          bo_episode_id="ep-1", bo_trial_index=1, logs_dir=logs)

    lines = (logs / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    written = json.loads(lines[0])
    assert written == row
    assert written["source"] == "bo"
    assert validate_run_row(written) == []
    # No results.tsv row and no per-trial run log for a bo trial.
    assert not tsv.exists()
    assert not (logs / "runs").exists()


@pytest.mark.unit
def test_bo_episode_summary_row(tmp_path):
    tsv = tmp_path / "results.tsv"

    record_bo_episode_summary(
        commit="c1", task="credit-g", model_family="xgboost",
        val_logloss=0.27, budget=10,
        space_keys=["n_estimators", "max_depth"],
        results_tsv=str(tsv), best_summary=_summary(val_logloss=0.27),
    )

    lines = tsv.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # header + exactly one summary row
    header = lines[0].split("\t")
    fields = dict(zip(header, lines[1].split("\t")))
    assert fields["status"] == "bo_episode"
    assert fields["model_family"] == "xgboost"
    assert fields["val_logloss"] == "0.27"
    assert fields["description"] == "bo n=10 space=n_estimators,max_depth"


@pytest.mark.unit
def test_extract_decisions_ignores_bo_gracefully(tmp_path):
    """The existing parser doesn't crash on a v2 ledger (full support: Step 10)."""
    logs = tmp_path / "logs"
    logs.mkdir()
    ts = "2026-01-01T00:00:01+00:00"
    agent = build_run_row("c1", _summary(), "keep", "d", {}, 1, ts)
    bo = build_bo_run_row("c2", _summary(), {"max_depth": 4}, "ep-1", 1, 2, ts)
    (logs / "runs.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in [agent, bo]), encoding="utf-8"
    )

    df = extract_decisions(logs, repo_dir=None)  # must not raise
    assert len(df) == 2
