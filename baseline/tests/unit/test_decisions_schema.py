import json

import pytest

from logging_lib import (
    build_decision_row,
    record_trial,
    validate_post_trial_reflection,
    validate_pre_trial_plan,
)


def _valid_pre():
    return {
        "family_chosen": "xgboost",
        "locus_of_change": "hyperparameter",
        "intent": "Lower learning rate to 0.05 to reduce overfitting.",
    }


def _valid_post():
    return {
        "keep_or_discard": "keep",
        "reason": "val_logloss improved 0.279 -> 0.271.",
        "surprise": False,
    }


def _summary(model_family="xgboost"):
    return {
        "val_logloss": 0.27, "val_acc": 0.87, "val_auc": 0.93,
        "train_seconds": 0.4, "total_seconds": 0.5, "peak_mem_mb": 250.0,
        "model_family": model_family, "n_params": 1800, "task_name": "adult",
    }


@pytest.mark.unit
def test_decisions_schema_valid():
    assert validate_pre_trial_plan(_valid_pre()) == []
    assert validate_post_trial_reflection(_valid_post()) == []
    row = build_decision_row("abc1234", 1, _valid_pre(), _valid_post(), False)
    assert row["family_chosen"] == "xgboost"
    assert row["locus_of_change"] == "hyperparameter"
    assert row["keep_or_discard"] == "keep"
    assert row["surprise"] is False
    assert row["family_changed_from_prior"] is False
    assert row["trial_id"] == 1 and row["commit"] == "abc1234"


@pytest.mark.unit
def test_decisions_schema_rejects_unknown_locus():
    bad = {**_valid_pre(), "locus_of_change": "vibes"}
    problems = validate_pre_trial_plan(bad)
    assert any("locus_of_change" in p for p in problems)
    with pytest.raises(ValueError, match="locus_of_change"):
        build_decision_row("c", 1, bad, _valid_post(), False)


@pytest.mark.unit
def test_decisions_reflection_requires_bool_surprise():
    bad = {**_valid_post(), "surprise": "yes"}
    assert any("surprise" in p for p in validate_post_trial_reflection(bad))


@pytest.mark.unit
def test_decisions_record_attached_to_trial(tmp_path):
    """A trial recorded with pre+post lands in decisions.jsonl, tied to it."""
    logs_dir = tmp_path / "logs"

    # Trial 1: xgboost (no prior -> family unchanged).
    record_trial(
        commit="aaa1111", summary=_summary("xgboost"), status="keep",
        description="baseline", hyperparameters={"n_estimators": 300},
        logs_dir=logs_dir, results_tsv=str(tmp_path / "results.tsv"),
        pre_trial_plan=_valid_pre(),
        post_trial_reflection=_valid_post(),
    )
    # Trial 2: swap to random_forest -> family_changed_from_prior should be True.
    rf_pre = {**_valid_pre(), "family_chosen": "random_forest",
              "locus_of_change": "model_family"}
    record_trial(
        commit="bbb2222", summary=_summary("random_forest"), status="discard",
        description="try RF", hyperparameters={"n_estimators": 200},
        logs_dir=logs_dir, results_tsv=str(tmp_path / "results.tsv"),
        pre_trial_plan=rf_pre,
        post_trial_reflection={**_valid_post(), "keep_or_discard": "discard",
                               "reason": "worse logloss", "surprise": True},
    )

    decisions = [
        json.loads(line)
        for line in (logs_dir / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(decisions) == 2
    assert decisions[0]["commit"] == "aaa1111" and decisions[0]["trial_id"] == 1
    assert decisions[0]["family_changed_from_prior"] is False
    assert decisions[1]["commit"] == "bbb2222" and decisions[1]["trial_id"] == 2
    assert decisions[1]["family_changed_from_prior"] is True
    assert decisions[1]["keep_or_discard"] == "discard"
    assert decisions[1]["surprise"] is True


@pytest.mark.unit
def test_decisions_not_written_without_both_halves(tmp_path):
    """Pre-plan alone (no reflection) does not emit a decision row."""
    logs_dir = tmp_path / "logs"
    record_trial(
        commit="ccc3333", summary=_summary(), status="keep", description="x",
        hyperparameters={}, logs_dir=logs_dir,
        results_tsv=str(tmp_path / "results.tsv"),
        pre_trial_plan=_valid_pre(), post_trial_reflection=None,
    )
    assert not (logs_dir / "decisions.jsonl").exists()
