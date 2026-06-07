"""Step 7: BO-tool constraint enforcement (unit) — refusals consume zero trials."""

import json

import pytest

import prepare
from tools import run_bo


def _session(tmp_path, task="balance-scale", seed=7, n_existing_rows=0):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "session.json").write_text(
        json.dumps({"task": task, "seed": seed, "trial_budget": 50}),
        encoding="utf-8",
    )
    if n_existing_rows:
        (logs / "runs.jsonl").write_text(
            "".join("{}\n" for _ in range(n_existing_rows)), encoding="utf-8"
        )
    return logs


_OK_SPACE = {"max_depth": {"type": "int", "low": 2, "high": 8}}


@pytest.mark.unit
def test_bo_budget_refusal_below_min(tmp_path):
    logs = _session(tmp_path)
    with pytest.raises(run_bo.BORefusal, match=r"\[5, 15\]"):
        run_bo.plan_episode(str(logs), "xgboost", 4, _OK_SPACE)
    assert not (logs / "runs.jsonl").exists()


@pytest.mark.unit
def test_bo_budget_refusal_above_max(tmp_path):
    logs = _session(tmp_path)
    with pytest.raises(run_bo.BORefusal, match=r"\[5, 15\]"):
        run_bo.plan_episode(str(logs), "xgboost", 16, _OK_SPACE)
    assert not (logs / "runs.jsonl").exists()


@pytest.mark.unit
def test_bo_budget_refusal_exceeds_remaining(tmp_path):
    # 48 trials already used -> remaining 2 < requested 5.
    logs = _session(tmp_path, n_existing_rows=48)
    before = (logs / "runs.jsonl").read_text(encoding="utf-8")
    with pytest.raises(run_bo.BORefusal, match="remaining"):
        run_bo.plan_episode(str(logs), "xgboost", 5, _OK_SPACE)
    # Ledger untouched (zero trials consumed).
    assert (logs / "runs.jsonl").read_text(encoding="utf-8") == before


@pytest.mark.unit
def test_bo_invalid_space_refusal(tmp_path):
    logs = _session(tmp_path)
    bad = {"not_a_param": {"type": "int", "low": 1, "high": 2}}
    with pytest.raises(run_bo.BORefusal, match="not_a_param"):
        run_bo.plan_episode(str(logs), "xgboost", 5, bad)
    assert not (logs / "runs.jsonl").exists()


@pytest.mark.unit
def test_bo_unknown_family_refusal(tmp_path):
    logs = _session(tmp_path)
    with pytest.raises(run_bo.BORefusal, match="ALLOWED_FAMILIES"):
        run_bo.plan_episode(str(logs), "not_a_family", 5, _OK_SPACE)


@pytest.mark.unit
def test_bo_refused_without_session(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    with pytest.raises(run_bo.BORefusal, match="session.json"):
        run_bo.plan_episode(str(logs), "xgboost", 5, _OK_SPACE)


@pytest.mark.unit
def test_bo_plan_happy_path(tmp_path):
    logs = _session(tmp_path)
    plan = run_bo.plan_episode(str(logs), "xgboost", 5, _OK_SPACE)
    assert plan["task"] == "balance-scale"
    assert plan["episode_index"] == 1
    assert plan["episode_id"] == "bo-ep001"
    assert plan["remaining"] == prepare.TRIAL_BUDGET
    assert plan["penalty"] == prepare._TASK_REGISTRY["balance-scale"]["penalty_logloss"]
    assert set(plan["space"]) == {"max_depth"}
