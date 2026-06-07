import json

import pytest

from tools.validate_jsonl import main, validate_jsonl, validate_run_row


def _good_row():
    return {
        "schema_version": 1,
        "trial_id": 1,
        "commit": "abc1234",
        "timestamp": "2026-05-30T00:00:00+00:00",
        "task": "adult",
        "model_family": "xgboost",
        "hyperparameters": {"n_estimators": 20},
        "val_logloss": 0.34,
        "val_acc": 0.85,
        "val_auc": 0.91,
        "train_seconds": 4.7,
        "total_seconds": 12.3,
        "peak_mem_mb": 412.0,
        "status": "keep",
        "description": "baseline",
    }


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )


@pytest.mark.unit
def test_jsonl_validator_passes_good(tmp_path):
    path = tmp_path / "runs.jsonl"
    _write_jsonl(path, [_good_row(), {**_good_row(), "trial_id": 2}])
    assert validate_jsonl(path) == []
    assert main([str(path)]) == 0


@pytest.mark.unit
def test_jsonl_validator_rejects_bad(tmp_path):
    missing = _good_row()
    del missing["val_logloss"]
    wrong_type = {**_good_row(), "trial_id": 3, "val_acc": "high"}
    bad_status = {**_good_row(), "trial_id": 4, "status": "maybe"}

    path = tmp_path / "runs.jsonl"
    _write_jsonl(path, [missing, wrong_type, bad_status])

    problems = validate_jsonl(path)
    assert any("val_logloss" in p for p in problems)
    assert any("val_acc" in p for p in problems)
    assert any("status" in p for p in problems)
    assert main([str(path)]) == 1


@pytest.mark.unit
def test_validate_run_row_rejects_bool_for_number():
    # bool is a subclass of int but must not satisfy a numeric field.
    row = {**_good_row(), "val_logloss": True}
    assert any("val_logloss" in e for e in validate_run_row(row))
