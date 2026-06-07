"""Step 10: v2-aware analysis — BO rows in the table, adoption flag, AUBC."""

import math

import pytest

from tools.extract_decisions import COLUMNS, extract_decisions
from tools.make_synthetic_session import build_synthetic_c1_session
from tools.metrics import aubc, best_so_far


@pytest.mark.unit
def test_extract_with_bo_rows(tmp_path):
    logs = tmp_path / "logs"
    n = build_synthetic_c1_session(logs)
    df = extract_decisions(logs, repo_dir=None)

    assert len(df) == n == 13
    assert list(df.columns) == COLUMNS
    for col in ("source", "bo_episode_id", "adopted_from_episode"):
        assert col in df.columns

    assert (df["source"] == "bo").sum() == 10
    assert (df["source"] == "agent").sum() == 3
    assert set(df["bo_episode_id"].dropna()) == {"bo-ep001", "bo-ep002"}

    # Exactly the adopting agent trial (7, xgboost) is flagged.
    adopted = df[df["adopted_from_episode"]]
    assert list(adopted["trial_id"]) == [7]
    assert adopted.iloc[0]["model_family"] == "xgboost"
    # The from-scratch mlp trial is not an adoption.
    assert not bool(df.loc[df["trial_id"] == 13, "adopted_from_episode"].iloc[0])


@pytest.mark.unit
def test_aubc_computation():
    losses = [0.40, 0.36, 0.30, 0.33, 0.38]
    assert best_so_far(losses) == [0.40, 0.36, 0.30, 0.30, 0.30]
    assert aubc(losses) == pytest.approx((0.40 + 0.36 + 0.30 + 0.30 + 0.30) / 5)


@pytest.mark.unit
def test_best_so_far_ignores_failures():
    # NaN (failed) trials carry the previous best forward, never improving it.
    losses = [0.5, float("nan"), 0.4, float("nan"), 0.45]
    assert best_so_far(losses) == [0.5, 0.5, 0.4, 0.4, 0.4]
    assert aubc(losses) == pytest.approx((0.5 + 0.5 + 0.4 + 0.4 + 0.4) / 5)


@pytest.mark.unit
def test_aubc_empty_is_nan():
    assert math.isnan(aubc([]))
