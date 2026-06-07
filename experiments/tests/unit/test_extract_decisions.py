import json

import pandas as pd
import pytest

from tools.extract_decisions import COLUMNS, extract_decisions


def _run_row(trial_id, commit, family, logloss, status="keep"):
    return {
        "schema_version": 1, "trial_id": trial_id, "commit": commit,
        "timestamp": f"2026-05-30T00:0{trial_id}:00+00:00", "task": "adult",
        "model_family": family, "hyperparameters": {"n_estimators": 100 + trial_id},
        "val_logloss": logloss, "val_acc": 0.8, "val_auc": 0.9,
        "train_seconds": 1.0, "total_seconds": 1.0, "peak_mem_mb": 100.0,
        "status": status, "description": "x",
    }


def _decision_row(trial_id, commit, family, locus, kod):
    return {
        "schema_version": 1, "trial_id": trial_id, "commit": commit,
        "family_chosen": family, "family_changed_from_prior": False,
        "locus_of_change": locus, "intent": "do a thing",
        "keep_or_discard": kod, "reason": "because", "surprise": False,
    }


def _write_session(tmp_path, runs, decisions):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "runs.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in runs), encoding="utf-8"
    )
    if decisions is not None:
        (logs / "decisions.jsonl").write_text(
            "".join(json.dumps(d) + "\n" for d in decisions), encoding="utf-8"
        )
    return logs


@pytest.mark.unit
def test_extract_decisions_synthetic(tmp_path):
    runs = [_run_row(i, f"c{i}", "xgboost", 0.40 - i * 0.01) for i in range(1, 6)]
    decisions = [_decision_row(i, f"c{i}", "xgboost", "hyperparameter", "keep")
                 for i in range(1, 6)]
    logs = _write_session(tmp_path, runs, decisions)

    df = extract_decisions(logs, repo_dir=None)
    assert len(df) == 5
    assert list(df.columns) == COLUMNS
    # Delta from parent: trial 1 has no parent (NaN), trial 2 is -0.01.
    assert pd.isna(df.iloc[0]["val_logloss_delta_from_parent"])
    assert df.iloc[1]["val_logloss_delta_from_parent"] == pytest.approx(-0.01)
    assert df.iloc[2]["locus_of_change"] == "hyperparameter"
    assert json.loads(df.iloc[0]["hyperparameters_json"])["n_estimators"] == 101


@pytest.mark.unit
def test_family_change_detection(tmp_path):
    runs = [
        _run_row(1, "c1", "xgboost", 0.40),
        _run_row(2, "c2", "random_forest", 0.42),  # swap
        _run_row(3, "c3", "random_forest", 0.39),  # same
    ]
    logs = _write_session(tmp_path, runs, decisions=[])
    df = extract_decisions(logs, repo_dir=None)
    changed = list(df["family_changed_from_prior"])
    assert changed == [False, True, False]


@pytest.mark.unit
def test_extract_handles_missing_decisions(tmp_path):
    """Trials without a decision row get nulls, not dropped."""
    runs = [_run_row(i, f"c{i}", "xgboost", 0.4) for i in range(1, 4)]
    # Only trial 2 has a decision.
    decisions = [_decision_row(2, "c2", "xgboost", "preprocessing", "discard")]
    logs = _write_session(tmp_path, runs, decisions)

    df = extract_decisions(logs, repo_dir=None)
    assert len(df) == 3  # none dropped
    row1 = df[df["trial_id"] == 1].iloc[0]
    # locus is now derived (baseline trial -> "other"); keep_or_discard stays
    # decision-only (None without a decision record / git).
    assert row1["locus_of_change"] == "other"
    assert row1["keep_or_discard"] is None
    row2 = df[df["trial_id"] == 2].iloc[0]
    assert row2["locus_of_change"] == "preprocessing"  # explicit decision wins
    assert row2["keep_or_discard"] == "discard"


@pytest.mark.unit
def test_locus_derived_from_changes(tmp_path):
    runs = [
        _run_row(1, "c1", "xgboost", 0.40),          # baseline -> other
        _run_row(2, "c2", "xgboost", 0.39),          # HP changed -> hyperparameter
        _run_row(3, "c3", "random_forest", 0.42),    # family swap -> model_family
    ]
    logs = _write_session(tmp_path, runs, decisions=[])
    df = extract_decisions(logs, repo_dir=None)
    assert list(df["locus_of_change"]) == ["other", "hyperparameter", "model_family"]


@pytest.mark.unit
def test_kept_on_branch_and_intent_from_git(tmp_path):
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True,
                       capture_output=True)

    def head():
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()

    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (repo / "train.py").write_text("x = 1\n")
    git("add", "train.py")
    git("commit", "-qm", "trial1 baseline")
    c1 = head()
    (repo / "train.py").write_text("x = 2\n")
    git("commit", "-qam", "trial2 lower lr")
    c2 = head()
    (repo / "train.py").write_text("x = 3\n")
    git("commit", "-qam", "trial3 bad idea")
    c3 = head()
    git("reset", "--hard", c2)  # discard trial3 -> orphaned

    runs = [
        _run_row(1, c1, "xgboost", 0.40),
        _run_row(2, c2, "xgboost", 0.39),
        _run_row(3, c3, "xgboost", 0.45),
    ]
    logs = _write_session(tmp_path, runs, decisions=[])
    df = extract_decisions(logs, repo_dir=str(repo))

    kept = dict(zip(df["trial_id"], df["kept_on_branch"]))
    assert kept[1] == True and kept[2] == True   # noqa: E712
    assert kept[3] == False                       # orphaned -> discarded
    intents = dict(zip(df["trial_id"], df["intent"]))
    assert intents[2] == "trial2 lower lr"        # intent from commit message

    # head= lets us measure keep/discard against an arbitrary ref (e.g. an old
    # run's branch) without checking it out.
    df_at_c1 = extract_decisions(logs, repo_dir=str(repo), head=c1)
    kept_at_c1 = dict(zip(df_at_c1["trial_id"], df_at_c1["kept_on_branch"]))
    assert kept_at_c1[1] == True    # noqa: E712  -- c1 is the ref itself
    assert kept_at_c1[2] == False   # noqa: E712  -- c2 is not an ancestor of c1


@pytest.mark.unit
def test_extract_no_decisions_file_at_all(tmp_path):
    runs = [_run_row(1, "c1", "mlp", 0.5)]
    logs = _write_session(tmp_path, runs, decisions=None)
    df = extract_decisions(logs, repo_dir=None)
    assert len(df) == 1
    assert df.iloc[0]["locus_of_change"] == "other"  # derived baseline locus
