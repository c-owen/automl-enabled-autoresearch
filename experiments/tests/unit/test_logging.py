import json
import os

import pytest

import logging_lib
from logging_lib import (
    append_run_row,
    extract_hyperparameters,
    record_trial,
    start_session,
)
from tools.validate_jsonl import validate_run_row


def _valid_summary():
    """A parsed print-contract summary (the output shape of parse_summary_block)."""
    return {
        "val_logloss": 0.3421,
        "val_acc": 0.851,
        "val_auc": 0.912,
        "train_seconds": 4.7,
        "total_seconds": 12.3,
        "peak_mem_mb": 412.0,
        "model_family": "xgboost",
        "n_params": 4892,
        "task_name": "adult",
    }


@pytest.mark.unit
def test_append_run_row_appends(tmp_path):
    path = tmp_path / "runs.jsonl"
    append_run_row(path, {"trial_id": 1, "a": 1})
    append_run_row(path, {"trial_id": 2, "a": 2})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(ln)["trial_id"] for ln in lines] == [1, 2]


@pytest.mark.unit
def test_append_run_row_atomic(tmp_path, monkeypatch):
    path = tmp_path / "runs.jsonl"
    append_run_row(path, {"trial_id": 1})

    def _boom(src, dst):
        raise RuntimeError("simulated crash at rename")

    # Fail at the atomic rename step of the second append.
    monkeypatch.setattr(logging_lib.os, "replace", _boom)
    with pytest.raises(RuntimeError):
        append_run_row(path, {"trial_id": 2})

    # Original file is intact: exactly one complete line, no partial write.
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["trial_id"] == 1
    # No temp file left behind.
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.unit
def test_record_trial_writes_all(tmp_path):
    logs_dir = tmp_path / "logs"
    results_tsv = tmp_path / "results.tsv"
    src_log = tmp_path / "captured.log"
    src_log.write_text("full stdout here\n", encoding="utf-8")

    row = record_trial(
        commit="abc1234",
        summary=_valid_summary(),
        status="keep",
        description="baseline xgboost",
        hyperparameters={"n_estimators": 20, "max_depth": 4},
        logs_dir=logs_dir,
        run_log_path=str(src_log),
        results_tsv=str(results_tsv),
    )

    # runs.jsonl: one valid row matching the schema.
    runs_jsonl = logs_dir / "runs.jsonl"
    assert runs_jsonl.exists()
    written = json.loads(runs_jsonl.read_text(encoding="utf-8").strip())
    assert written == row
    assert validate_run_row(written) == []
    assert written["task"] == "adult"
    assert written["hyperparameters"]["n_estimators"] == 20

    # results.tsv: header + one data row, with every column populated. The
    # mem_mb / trial_seconds columns map to the row's peak_mem_mb / total_seconds.
    tsv_lines = results_tsv.read_text(encoding="utf-8").splitlines()
    assert len(tsv_lines) == 2
    header = tsv_lines[0].split("\t")
    fields = dict(zip(header, tsv_lines[1].split("\t")))
    assert header[0] == "commit"
    assert fields["commit"] == "abc1234"
    assert fields["mem_mb"] == "412.0"          # <- was blank before the fix
    assert fields["trial_seconds"] == "12.3"    # <- was blank before the fix
    assert fields["model_family"] == "xgboost"
    assert fields["status"] == "keep"

    # Persisted per-commit run log.
    persisted = logs_dir / "runs" / "abc1234.log"
    assert persisted.exists()
    assert persisted.read_text(encoding="utf-8") == "full stdout here\n"


@pytest.mark.unit
def test_start_session_writes_metadata(tmp_path):
    logs_dir = tmp_path / "logs"
    path = start_session(logs_dir, {"task": "adult", "trial_budget": 50})
    meta = json.loads(open(path, encoding="utf-8").read())
    assert meta["task"] == "adult"
    assert meta["schema_version"] == logging_lib.SCHEMA_VERSION


@pytest.mark.unit
def test_extract_hyperparameters_excludes_contract_constants(tmp_path):
    """Only tunable UPPER_CASE literals are captured; contract constants aren't."""
    train_py = tmp_path / "train.py"
    train_py.write_text(
        'MODEL = "xgboost"\n'
        "TIME_BUDGET = 300\n"
        "TIMEOUT_EXIT_CODE = 124\n"
        "N_ESTIMATORS = 300\n"
        "LEARNING_RATE = 0.1\n"
        "some_func = lambda: None\n",
        encoding="utf-8",
    )
    hp = extract_hyperparameters(str(train_py))
    assert hp == {"n_estimators": 300, "learning_rate": 0.1}
    assert "timeout_exit_code" not in hp
    assert "model" not in hp


@pytest.mark.unit
def test_record_trial_rejects_bad_status(tmp_path):
    with pytest.raises(ValueError, match="status"):
        record_trial(
            commit="abc1234",
            summary=_valid_summary(),
            status="bogus",
            description="",
            hyperparameters={},
            logs_dir=tmp_path / "logs",
            results_tsv=str(tmp_path / "results.tsv"),
        )
