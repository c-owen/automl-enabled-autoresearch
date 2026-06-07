"""Step 9: reference arms R1/R2 — scripted CASH search, comparable ledger."""

import json

import pytest

from tools.run_reference import _ensure_isolated_outdir, run_reference
from tools.validate_jsonl import validate_jsonl


def _best_so_far(rows):
    best = float("inf")
    out = []
    for r in rows:
        best = min(best, r["val_logloss"])
        out.append(best)
    return out


def _read_rows(out_dir):
    path = out_dir / "logs" / "runs.jsonl"
    return [json.loads(ln) for ln in
            path.read_text(encoding="utf-8").splitlines() if ln.strip()]


@pytest.mark.integration
@pytest.mark.parametrize("method", ["tpe", "random"])
def test_reference_smoke(method, tmp_path):
    out = tmp_path / f"{method}-run"
    meta = run_reference(method, "balance-scale", seed=0, trials=5, out_dir=str(out))

    assert meta["method"] == method and meta["trials"] == 5
    rows = _read_rows(out)
    assert len(rows) == 5
    assert all(r["source"] == "reference" for r in rows)
    assert all(r["status"] == "reference" for r in rows)
    assert all(r["method"] == method for r in rows)
    # Each trial declares a real family and a config that includes it.
    for r in rows:
        assert r["model_family"] in {"xgboost", "random_forest",
                                     "logistic_regression", "mlp"}
        assert r["hyperparameters"]["family"] == r["model_family"]

    # The v2 ledger validates, and best-so-far is monotone non-increasing.
    assert validate_jsonl(out / "logs" / "runs.jsonl") == []
    bsf = _best_so_far(rows)
    assert all(b2 <= b1 for b1, b2 in zip(bsf, bsf[1:]))

    # results.tsv was written too (one row per trial + header).
    tsv = (out / "results.tsv").read_text(encoding="utf-8").splitlines()
    assert len(tsv) == 6


@pytest.mark.integration
def test_reference_deterministic(tmp_path):
    a = run_reference("tpe", "balance-scale", seed=1, trials=5,
                      out_dir=str(tmp_path / "a"))
    b = run_reference("tpe", "balance-scale", seed=1, trials=5,
                      out_dir=str(tmp_path / "b"))
    # Same method + seed -> same search outcome.
    assert a["best_value"] == b["best_value"]
    assert a["best_params"] == b["best_params"]


@pytest.mark.unit
def test_reference_isolated_outdir(tmp_path):
    # A directory holding a live session.json must be refused.
    live = tmp_path / "live"
    (live / "logs").mkdir(parents=True)
    (live / "logs" / "session.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="session.json"):
        _ensure_isolated_outdir(str(live))

    # Also when session.json sits at the top level of --out.
    live2 = tmp_path / "live2"
    live2.mkdir()
    (live2 / "session.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="session.json"):
        _ensure_isolated_outdir(str(live2))

    # A fresh dir is fine.
    _ensure_isolated_outdir(str(tmp_path / "fresh"))
