"""Step 7: BO-tool constraint enforcement (unit) — refusals consume zero trials."""

import json

import pytest

import family_adapters as fa
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


# --- A1: pre-flight value validation against PARAM_SPECS (v1.1) --------------

# The pilot's exact (out-of-spec) space string — the call the v1.0 tool accepted.
_PILOT_SPACE = (
    '{"n_estimators": {"type": "int", "low": 100, "high": 1500, "log": true}, '
    '"max_features": {"type": "categorical", "choices": ["sqrt", "log2", 0.3, 0.5]}, '
    '"min_samples_leaf": {"type": "int", "low": 1, "high": 8, "log": false}}'
)


@pytest.mark.unit
def test_bo_refuses_float_max_features(tmp_path):
    logs = _session(tmp_path)
    space = {"max_features": {"type": "float", "low": 0.1, "high": 0.9}}
    with pytest.raises(run_bo.BORefusal, match="max_features"):
        run_bo.plan_episode(str(logs), "random_forest", 5, space)
    assert not (logs / "runs.jsonl").exists()


@pytest.mark.unit
def test_bo_refuses_n_estimators_above_spec(tmp_path):
    logs = _session(tmp_path)
    space = {"n_estimators": {"type": "int", "low": 100, "high": 1500, "log": True}}
    with pytest.raises(run_bo.BORefusal, match="outside the adapter"):
        run_bo.plan_episode(str(logs), "random_forest", 5, space)
    assert not (logs / "runs.jsonl").exists()


@pytest.mark.unit
def test_bo_refuses_pilot_space_zero_rows(tmp_path):
    logs = _session(tmp_path)
    with pytest.raises(run_bo.BORefusal):
        run_bo.plan_episode(str(logs), "random_forest", 10, json.loads(_PILOT_SPACE))
    assert not (logs / "runs.jsonl").exists()


@pytest.mark.unit
def test_bo_refusal_message_includes_specs(tmp_path):
    logs = _session(tmp_path)
    with pytest.raises(run_bo.BORefusal, match="legal search space"):
        run_bo.plan_episode(str(logs), "random_forest", 5,
                            {"n_estimators": {"type": "int", "low": 1, "high": 5000}})


@pytest.mark.unit
def test_bo_in_spec_box_parses():
    space = run_bo.parse_space("xgboost", {
        "max_depth": {"type": "int", "low": 2, "high": 12},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.4, "log": True},
    })
    assert set(space) == {"max_depth", "learning_rate"}


@pytest.mark.unit
@pytest.mark.parametrize("family", sorted(fa.PARAM_SPECS))
def test_bo_specs_lists_every_param(family):
    out = run_bo.format_specs(family)
    for param in fa.PARAM_SPECS[family]:
        assert param in out, f"{family}/{param} missing from --specs output"
    for param in fa.DEFAULTS[family]:
        assert param in out
    assert "defaults used" in out


# --- A2: episode summary discloses pinned defaults + caveat (v1.1) -----------


@pytest.mark.unit
def test_bo_disclosure_pins_undeclared_defaults():
    # Declare a space omitting max_depth -> the pinned max_depth default is shown.
    lines = run_bo._disclosure_lines("random_forest",
                                     ["n_estimators", "min_samples_leaf"])
    blob = "\n".join(lines)
    assert "max_depth = 16" in blob  # DEFAULTS['random_forest']['max_depth']
    assert any("preprocessing" in line for line in lines)  # the caveat line
