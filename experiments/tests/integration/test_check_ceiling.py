"""Step 5: family-ceiling check smoke (off by default; opt in with -m smoke).

Runs the real ceiling machinery end-to-end on balance-scale only (the fastest
hard task) and asserts a populated table comes back. The full gate is run by
hand via ``tools/check_ceiling.py``.
"""

import pytest

import family_adapters as fa
from tools.check_ceiling import _format_table, run_ceiling


@pytest.mark.smoke
def test_check_ceiling_runs():
    results = run_ceiling(tasks=["balance-scale"])

    assert "balance-scale" in results
    assert set(results["balance-scale"]) == set(fa.ALLOWED_FAMILIES)
    for family, info in results["balance-scale"].items():
        assert info["best"] > 0  # a finite val_logloss was recorded
        assert "config" in info

    table, wins = _format_table(results)
    assert "balance-scale" in table
    assert 0 <= wins <= 1
