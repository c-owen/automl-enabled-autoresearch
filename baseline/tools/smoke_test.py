"""End-to-end smoke test: a real 3-trial mini-session through every layer.

Runs three trials via run_trial.py (XGB baseline -> XGB HP tweak -> RF swap)
with faked LLM-side decision writes, then runs extract_decisions and executes
analysis.ipynb. This is the documented "is this thing working" check:

    uv run python tools/smoke_test.py

Everything runs in an isolated temp copy of the harness, so the repo's train.py
and logs are never touched.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = Path(__file__).resolve().parents[1]
RF_FIXTURE = REPO / "tests" / "fixtures" / "family_baselines" / "random_forest.py"

# Tiny-budget train.py variants (commit, train_src, pre_plan, post_reflection).
_HARNESS_FILES = ["prepare.py", "logging_lib.py", "run_trial.py", "analysis.ipynb"]


def _xgb_source(extra_replacements=()):
    src = (REPO / "train.py").read_text(encoding="utf-8")
    src = src.replace("N_ESTIMATORS = 300", "N_ESTIMATORS = 20")
    for old, new in extra_replacements:
        assert old in src, f"anchor missing in train.py: {old!r}"
        src = src.replace(old, new)
    return src


def _rf_source():
    src = RF_FIXTURE.read_text(encoding="utf-8")
    src = src.replace("N_ESTIMATORS = 200", "N_ESTIMATORS = 40")
    src = src.replace("MAX_DEPTH = 16", "MAX_DEPTH = 8")
    return src


def _trials():
    return [
        (
            "smoke001", _xgb_source(),
            {"family_chosen": "xgboost", "locus_of_change": "hyperparameter",
             "intent": "Baseline XGBoost with 20 estimators."},
            {"keep_or_discard": "keep", "reason": "baseline established.",
             "surprise": False},
            "xgb baseline",
        ),
        (
            "smoke002", _xgb_source([("MAX_DEPTH = 6", "MAX_DEPTH = 4")]),
            {"family_chosen": "xgboost", "locus_of_change": "hyperparameter",
             "intent": "Shallower trees (max_depth 4) to reduce overfitting."},
            {"keep_or_discard": "keep", "reason": "slightly better logloss.",
             "surprise": False},
            "xgb max_depth=4",
        ),
        (
            "smoke003", _rf_source(),
            {"family_chosen": "random_forest", "locus_of_change": "model_family",
             "intent": "Swap to a random forest to compare families."},
            {"keep_or_discard": "discard", "reason": "no better than XGBoost.",
             "surprise": True},
            "rf swap",
        ),
    ]


def _setup_workdir(work):
    work = Path(work)
    (work / "logs").mkdir(parents=True, exist_ok=True)
    for name in _HARNESS_FILES:
        shutil.copy(REPO / name, work / name)
    shutil.copytree(
        REPO / "tools", work / "tools",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    return work


def _run_trial(work, commit, train_src, pre, post, description):
    work = Path(work)
    (work / "train.py").write_text(train_src, encoding="utf-8")
    (work / "pre.json").write_text(json.dumps(pre), encoding="utf-8")
    (work / "post.json").write_text(json.dumps(post), encoding="utf-8")

    env = {
        **os.environ,
        "LOGS_DIR": str(work / "logs"),
        "TRIAL_COMMIT": commit,
        "TRIAL_DESCRIPTION": description,
        "TRIAL_STATUS": post["keep_or_discard"],  # mirror the LLM's decision
        "PRE_TRIAL_PLAN_PATH": str(work / "pre.json"),
        "POST_TRIAL_REFLECTION_PATH": str(work / "post.json"),
    }
    result = subprocess.run(
        [sys.executable, "run_trial.py"],
        cwd=str(work), env=env, capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"trial {commit} failed (rc={result.returncode}):\n{result.stderr}"
        )
    return result


def _execute_notebook(work):
    import nbformat
    from nbclient import NotebookClient

    os.environ["LOGS_DIR"] = str(Path(work) / "logs")
    nb = nbformat.read(str(Path(work) / "analysis.ipynb"), as_version=4)
    NotebookClient(
        nb, timeout=180, kernel_name="python3",
        resources={"metadata": {"path": str(work)}},
    ).execute()


def run_smoke(work_dir):
    """Run the full 3-trial smoke session under work_dir. Returns a summary."""
    work = _setup_workdir(work_dir)
    logs = work / "logs"

    for commit, src, pre, post, desc in _trials():
        _run_trial(work, commit, src, pre, post, desc)

    runs = [json.loads(l) for l in (logs / "runs.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    decisions = [json.loads(l) for l in (logs / "decisions.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]

    from tools.extract_decisions import extract_decisions
    df = extract_decisions(logs)
    (work / "analysis").mkdir(exist_ok=True)
    df.to_csv(work / "analysis" / "decisions.csv", index=False)

    _execute_notebook(work)

    return {
        "work_dir": str(work),
        "n_runs": len(runs),
        "n_decisions": len(decisions),
        "notebook_ok": True,
        "runs": runs,
    }


def main(argv=None) -> int:
    work = tempfile.mkdtemp(prefix="autoresearch_smoke_")
    result = run_smoke(work)

    print(f"\nSmoke session under: {result['work_dir']}")
    print(f"{'trial':>5}  {'commit':<9} {'family':<20} {'val_logloss':>11}  status")
    for r in result["runs"]:
        ll = r["val_logloss"]
        ll_s = "nan" if ll != ll else f"{ll:.4f}"  # nan check
        print(f"{r['trial_id']:>5}  {r['commit']:<9} {r['model_family']:<20} {ll_s:>11}  {r['status']}")
    print(f"\nruns.jsonl rows: {result['n_runs']}  "
          f"decisions.jsonl rows: {result['n_decisions']}  "
          f"notebook: {'OK' if result['notebook_ok'] else 'FAILED'}")
    ok = result["n_runs"] == 3 and result["n_decisions"] == 3 and result["notebook_ok"]
    print("SMOKE TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
