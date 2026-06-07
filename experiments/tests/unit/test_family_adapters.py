"""Step 4: family-adapter config validation (unit)."""

import pytest

import family_adapters as fa


@pytest.mark.unit
def test_adapter_rejects_unknown_param():
    with pytest.raises(fa.AdapterError, match="bogus_param"):
        fa.build("xgboost", {"bogus_param": 5})


@pytest.mark.unit
def test_adapter_rejects_out_of_range():
    # xgboost max_depth spec is [2, 12]; 999 is out of range.
    with pytest.raises(fa.AdapterError, match="max_depth"):
        fa.build("xgboost", {"max_depth": 999})


@pytest.mark.unit
def test_adapter_rejects_bad_categorical_choice():
    with pytest.raises(fa.AdapterError, match="max_features"):
        fa.build("random_forest", {"max_features": "not_a_choice"})


@pytest.mark.unit
def test_adapter_rejects_unknown_family():
    with pytest.raises(fa.AdapterError, match="unknown family"):
        fa.build("not_a_family", {})


@pytest.mark.unit
def test_adapter_rejects_non_integer_for_int_param():
    with pytest.raises(fa.AdapterError, match="n_estimators"):
        fa.build("xgboost", {"n_estimators": 100.5})


@pytest.mark.unit
def test_validate_config_accepts_valid_subset():
    # A valid subset of params validates with no error (the agent may search any
    # subset; unspecified params fall back to DEFAULTS).
    fa.validate_config("xgboost", {"n_estimators": 100, "learning_rate": 0.05})
    fa.validate_config("random_forest", {"max_features": "log2"})
    fa.validate_config("mlp", {"hidden_sizes": "256,128", "batch_size": 128})


@pytest.mark.unit
def test_param_specs_cover_four_families():
    assert set(fa.PARAM_SPECS) == {
        "xgboost", "random_forest", "logistic_regression", "mlp",
    }
    # Every family also has a complete DEFAULTS entry.
    for family, specs in fa.PARAM_SPECS.items():
        assert set(specs).issubset(set(fa.DEFAULTS[family]))
