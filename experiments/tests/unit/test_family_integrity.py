"""A5/A6 (v1.1): family-integrity validation + ingest fixes.

Uses the committed credit-g C1 pilot ledger (`tests/fixtures/pilot_c1_creditg.jsonl`,
the real v1.0 trajectory enriched with the per-commit estimator class recovered
from git) as the archived-pilot fixture.
"""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from logging_lib import family_violation
from tools.extract_decisions import COLUMNS, extract_decisions
from tools.make_synthetic_session import build_synthetic_c1_entry_voluntary

PILOT = Path(__file__).resolve().parents[1] / "fixtures" / "pilot_c1_creditg.jsonl"


@pytest.mark.unit
def test_family_violation_helper():
    assert family_violation("random_forest", "ExtraTreesClassifier")
    assert family_violation("xgboost", "HistGradientBoostingClassifier")
    assert family_violation("logistic_regression", "RidgeClassifier")
    assert not family_violation("random_forest", "RandomForestClassifier")
    assert not family_violation("xgboost", "XGBClassifier")
    assert not family_violation("logistic_regression", "LogisticRegression")
    # mlp: a torch net is fine; impersonating another family's estimator is not.
    assert not family_violation("mlp", "_TorchMLPClassifier")
    assert family_violation("mlp", "RandomForestClassifier")
    # Missing class (v1.0 archive) or unknown family -> can't judge -> not a violation.
    assert not family_violation("random_forest", None)
    assert not family_violation("unknown", "ExtraTreesClassifier")


def _ingest(tmp_path, fixture):
    logs = tmp_path / "logs"
    logs.mkdir()
    shutil.copy(fixture, logs / "runs.jsonl")
    return extract_decisions(logs, repo_dir=None)


@pytest.mark.unit
def test_ingest_pilot_flags_family_violations(tmp_path):
    df = _ingest(tmp_path, PILOT)
    assert list(df.columns) == COLUMNS  # does not crash; new columns present

    flagged = df[df["family_violation"] == True]  # noqa: E712
    classes = set(flagged["estimator_class"])
    assert "ExtraTreesClassifier" in classes        # logged as random_forest
    assert "HistGradientBoostingClassifier" in classes  # logged as xgboost

    # Genuine canonical-class rows are not flagged.
    rf_ok = df[(df["model_family"] == "random_forest") &
               (df["estimator_class"] == "RandomForestClassifier")]
    assert len(rf_ok) >= 1 and not rf_ok["family_violation"].any()
    # bo rows are never flagged (adapter-built, canonical by construction).
    assert not df[df["source"] == "bo"]["family_violation"].any()


@pytest.mark.unit
def test_ingest_pilot_bo_intent_is_null(tmp_path):
    df = _ingest(tmp_path, PILOT)
    bo = df[df["source"] == "bo"]
    assert len(bo) == 10
    assert bo["intent"].isna().all()  # no parent-commit message inheritance (A6)


@pytest.mark.unit
def test_episode_entry_voluntary_classification(tmp_path):
    logs = tmp_path / "logs"
    build_synthetic_c1_entry_voluntary(logs)
    df = extract_decisions(logs, repo_dir=None)

    kinds = {eid: df[df["bo_episode_id"] == eid]["episode_kind"].iloc[0]
             for eid in ("bo-ep001", "bo-ep002")}
    assert kinds == {"bo-ep001": "entry", "bo-ep002": "voluntary"}

    # The entry episode carries a derivable contrast (episode best vs entry baseline).
    entry = df[df["bo_episode_id"] == "bo-ep001"].iloc[0]
    assert entry["entry_baseline_logloss"] == pytest.approx(0.50)
    assert entry["episode_best_logloss"] == pytest.approx(0.40)
    assert entry["entry_contrast"] == pytest.approx(-0.10)
    # The voluntary episode has no entry contrast.
    vol = df[df["bo_episode_id"] == "bo-ep002"].iloc[0]
    assert pd.isna(vol["entry_contrast"])
