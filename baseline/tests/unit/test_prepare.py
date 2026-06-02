import importlib

import pytest

import prepare


@pytest.mark.unit
def test_constants_present():
    """Every required harness constant exists with the right type."""
    assert isinstance(prepare.RANDOM_SEED, int)
    assert isinstance(prepare.VAL_FRAC, float)
    assert 0.0 < prepare.VAL_FRAC < 1.0
    assert isinstance(prepare.TIME_BUDGET, int)
    assert isinstance(prepare.TRIAL_BUDGET, int)
    assert isinstance(prepare.TASK_NAME, str)
    assert isinstance(prepare.ALLOWED_FAMILIES, list)


@pytest.mark.unit
def test_allowed_families_exact():
    """The allowed-family set is exactly the four-family scope."""
    assert set(prepare.ALLOWED_FAMILIES) == {
        "xgboost",
        "random_forest",
        "logistic_regression",
        "mlp",
    }


@pytest.mark.unit
def test_task_name_env_var(monkeypatch):
    """AUTORESEARCH_TASK overrides the default task at import."""
    monkeypatch.setenv("AUTORESEARCH_TASK", "credit-g")
    importlib.reload(prepare)
    try:
        assert prepare.TASK_NAME == "credit-g"
    finally:
        monkeypatch.delenv("AUTORESEARCH_TASK", raising=False)
        importlib.reload(prepare)
    assert prepare.TASK_NAME == prepare.DEFAULT_TASK == "adult"


@pytest.mark.unit
def test_unknown_task_raises():
    with pytest.raises(ValueError, match="Unknown task"):
        prepare.load_task("not-a-real-task")
