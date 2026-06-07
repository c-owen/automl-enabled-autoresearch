import textwrap

import pytest

from logging_lib import parse_summary_block

# The canonical print contract (tabular_port_plan section 3.2), wrapped in the
# surrounding noise a real trial emits: training chatter before, a sentinel
# after.
CANONICAL_STDOUT = textwrap.dedent(
    """\
    [trial] loading task adult ...
    [trial] fitting xgboost ...
    [trial] --- intermediate dashes should be ignored ---
    ---
    val_logloss:    0.342100
    val_acc:        0.851000
    val_auc:        0.912000
    train_seconds:  4.7
    total_seconds:  12.3
    peak_mem_mb:    412.0
    model_family:   xgboost
    n_params:       4892
    task_name:      adult
    END_OF_TRIAL
    """
)


@pytest.mark.unit
def test_parse_summary_block_valid():
    summary = parse_summary_block(CANONICAL_STDOUT)
    assert summary["val_logloss"] == pytest.approx(0.3421)
    assert summary["val_acc"] == pytest.approx(0.851)
    assert summary["val_auc"] == pytest.approx(0.912)
    assert summary["train_seconds"] == pytest.approx(4.7)
    assert summary["total_seconds"] == pytest.approx(12.3)
    assert summary["peak_mem_mb"] == pytest.approx(412.0)
    assert summary["model_family"] == "xgboost"
    assert summary["n_params"] == 4892
    assert isinstance(summary["n_params"], int)
    assert summary["task_name"] == "adult"


@pytest.mark.unit
def test_parse_summary_block_missing_key_raises():
    broken = CANONICAL_STDOUT.replace("model_family:   xgboost\n", "")
    with pytest.raises(ValueError, match="model_family"):
        parse_summary_block(broken)


@pytest.mark.unit
def test_parse_summary_block_extra_keys_ok():
    augmented = CANONICAL_STDOUT.replace(
        "task_name:      adult\n",
        "task_name:      adult\nextra_metric:   0.5\n",
    )
    summary = parse_summary_block(augmented)
    # All required keys still parse, and the extra key is preserved verbatim.
    assert summary["model_family"] == "xgboost"
    assert summary["extra_metric"] == "0.5"


@pytest.mark.unit
def test_parse_summary_block_no_block_raises():
    with pytest.raises(ValueError, match="no summary block"):
        parse_summary_block("just some logs\nno marker here\n")
