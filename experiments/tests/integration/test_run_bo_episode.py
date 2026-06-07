"""Step 7: sealed BO-episode behavior (integration) — determinism, the box,
the failure penalty, and ledger tagging. Uses balance-scale (fast) + xgboost
(bit-deterministic) with the minimum budget."""

import json

import pytest

import prepare
from tools import run_bo
from tools.start_session import start_session
from tools.validate_jsonl import validate_jsonl

_SPACE = {
    "max_depth": {"type": "int", "low": 2, "high": 6},
    "learning_rate": {"type": "float", "low": 0.02, "high": 0.3, "log": True},
}


def _new_session(path, task="balance-scale", seed=7):
    logs = path / "logs"
    logs.mkdir(parents=True)
    (logs / "session.json").write_text(
        json.dumps({"task": task, "seed": seed, "trial_budget": 50}),
        encoding="utf-8",
    )
    return logs


def _run(path, family="xgboost", budget=5, space=None, seed=7):
    logs = _new_session(path, seed=seed)
    plan = run_bo.plan_episode(str(logs), family, budget, space or _SPACE)
    run_bo.execute_episode(plan, family, budget, str(logs),
                           str(path / "results.tsv"), commit="cafe123")
    rows = [json.loads(ln) for ln in
            (logs / "runs.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    return logs, rows


@pytest.mark.integration
def test_bo_episode_deterministic(tmp_path):
    _, rows_a = _run(tmp_path / "a", seed=7)
    _, rows_b = _run(tmp_path / "b", seed=7)
    seq_a = [(r["hyperparameters"], round(r["val_logloss"], 8)) for r in rows_a]
    seq_b = [(r["hyperparameters"], round(r["val_logloss"], 8)) for r in rows_b]
    assert len(seq_a) == 5
    assert seq_a == seq_b  # identical trial sequence given the same seed


@pytest.mark.integration
def test_bo_respects_box(tmp_path):
    _, rows = _run(tmp_path)
    assert len(rows) == 5
    for r in rows:
        hp = r["hyperparameters"]
        assert set(hp) == {"max_depth", "learning_rate"}  # only declared keys vary
        assert 2 <= hp["max_depth"] <= 6
        assert 0.02 <= hp["learning_rate"] <= 0.3


@pytest.mark.integration
def test_bo_refused_in_C0_session(tmp_path):
    """A C0 session enables no capabilities -> run_bo refuses, zero ledger rows."""
    logs = tmp_path / "logs"
    start_session(
        logs_dir=str(logs), task="balance-scale", seed=7, arm="C0",
        create_branch=False, archive=False,
        program_md_path=str(tmp_path / "program.md"),
    )
    with pytest.raises(run_bo.BORefusal, match="does not enable"):
        run_bo.plan_episode(str(logs), "xgboost", 5, _SPACE)
    assert not (logs / "runs.jsonl").exists()


@pytest.mark.integration
def test_bo_allowed_in_C1_session(tmp_path):
    """A C1 session enables bo -> plan_episode proceeds."""
    logs = tmp_path / "logs"
    start_session(
        logs_dir=str(logs), task="balance-scale", seed=7, arm="C1",
        create_branch=False, archive=False,
        program_md_path=str(tmp_path / "program.md"),
    )
    plan = run_bo.plan_episode(str(logs), "xgboost", 5, _SPACE)
    assert plan["episode_id"] == "bo-ep001"


@pytest.mark.integration
def test_bo_failure_penalty(tmp_path):
    # A box whose values all exceed PARAM_SPECS (max_depth high is 12): every
    # build() raises AdapterError -> each trial takes the penalty, episode finishes.
    bad_space = {"max_depth": {"type": "int", "low": 50, "high": 60}}
    logs, rows = _run(tmp_path, space=bad_space)
    penalty = prepare._TASK_REGISTRY["balance-scale"]["penalty_logloss"]
    assert len(rows) == 5
    assert all(r["val_logloss"] == penalty for r in rows)
    assert all(r["status"] == "bo_trial" for r in rows)
    # The episode still emitted its single summary row.
    tsv = (logs.parent / "results.tsv").read_text(encoding="utf-8").splitlines()
    assert sum("bo_episode" in ln for ln in tsv) == 1


@pytest.mark.integration
def test_bo_ledger_tagging(tmp_path):
    logs, rows = _run(tmp_path)
    assert len(rows) == 5
    assert all(r["source"] == "bo" for r in rows)
    assert {r["bo_episode_id"] for r in rows} == {"bo-ep001"}
    assert [r["bo_trial_index"] for r in rows] == [1, 2, 3, 4, 5]

    # results.tsv: exactly one episode-summary row.
    tsv = (logs.parent / "results.tsv").read_text(encoding="utf-8").splitlines()
    assert len(tsv) == 2  # header + one summary
    assert "bo_episode" in tsv[1]

    # A second episode in the same session increments the episode index.
    plan2 = run_bo.plan_episode(str(logs), "xgboost", 5, _SPACE)
    assert plan2["episode_index"] == 2 and plan2["episode_id"] == "bo-ep002"
    run_bo.execute_episode(plan2, "xgboost", 5, str(logs),
                           str(logs.parent / "results.tsv"), commit="cafe123")
    all_rows = [json.loads(ln) for ln in
                (logs / "runs.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(all_rows) == 10
    assert {r["bo_episode_id"] for r in all_rows} == {"bo-ep001", "bo-ep002"}

    # The whole v2 ledger validates.
    assert validate_jsonl(logs / "runs.jsonl") == []
