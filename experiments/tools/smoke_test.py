"""End-to-end smoke test: a mini-grid through every layer of the harness.

Proves the whole stack in one command, in an isolated temp copy of the harness
(the repo's train.py / logs are never touched):

    uv run python tools/smoke_test.py

Three phases on the fast `balance-scale` task:
  (a) C0 mini-session — 3 agent trials via run_trial.py (faked decisions), the
      LLM-only control.
  (b) C1 mini-session — 3 agent trials + one budget-5 BO episode invoked via
      run_bo.py (the tool arm).
  (c) R1 reference   — a 5-trial scripted TPE run via run_reference.py.
Then it ingests the C1 session (extract_decisions, exercising the v2 BO-aware
table) and executes analysis.ipynb against it, and prints a combined summary.
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
SMOKE_TASK = "balance-scale"

# Top-level modules the workdir needs to run trials, episodes and references.
_HARNESS_MODULES = [
    "prepare.py", "logging_lib.py", "run_trial.py", "family_adapters.py",
    "arms.py", "analysis.ipynb",
]


# --- workdir setup ----------------------------------------------------------

def _setup_workdir(work):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    for name in _HARNESS_MODULES:
        shutil.copy(REPO / name, work / name)
    for sub in ("tools", "playbook", "data"):
        shutil.copytree(REPO / sub, work / sub,
                        ignore=shutil.ignore_patterns("__pycache__"))
    return work


# --- train.py variants for the agent trials ---------------------------------

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


def _agent_trials():
    return [
        ("smoke001", _xgb_source(),
         {"family_chosen": "xgboost", "locus_of_change": "hyperparameter",
          "intent": "Baseline XGBoost with 20 estimators."},
         {"keep_or_discard": "keep", "reason": "baseline established.",
          "surprise": False}, "xgb baseline"),
        ("smoke002", _xgb_source([("MAX_DEPTH = 6", "MAX_DEPTH = 4")]),
         {"family_chosen": "xgboost", "locus_of_change": "hyperparameter",
          "intent": "Shallower trees (max_depth 4)."},
         {"keep_or_discard": "keep", "reason": "slightly better logloss.",
          "surprise": False}, "xgb max_depth=4"),
        ("smoke003", _rf_source(),
         {"family_chosen": "random_forest", "locus_of_change": "model_family",
          "intent": "Swap to a random forest to compare families."},
         {"keep_or_discard": "discard", "reason": "no better than XGBoost.",
          "surprise": True}, "rf swap"),
    ]


def _run_trial(work, logs_dir, commit, train_src, pre, post, description):
    work = Path(work)
    (work / "train.py").write_text(train_src, encoding="utf-8")
    (work / "pre.json").write_text(json.dumps(pre), encoding="utf-8")
    (work / "post.json").write_text(json.dumps(post), encoding="utf-8")
    env = {
        **os.environ,
        "LOGS_DIR": str(logs_dir),
        "RESULTS_TSV": str(Path(logs_dir).parent / "results.tsv"),
        "TRIAL_COMMIT": commit,
        "TRIAL_DESCRIPTION": description,
        "TRIAL_STATUS": post["keep_or_discard"],
        "PRE_TRIAL_PLAN_PATH": str(work / "pre.json"),
        "POST_TRIAL_REFLECTION_PATH": str(work / "post.json"),
    }
    result = subprocess.run(
        [sys.executable, "run_trial.py"], cwd=str(work), env=env,
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"trial {commit} failed (rc={result.returncode}):\n"
                           f"{result.stdout}\n{result.stderr}")
    return result


def _start_session(work, logs_dir, arm):
    from tools.start_session import start_session
    start_session(
        logs_dir=str(logs_dir), task=SMOKE_TASK, seed=7, arm=arm,
        create_branch=False, archive=False,
        results_tsv=str(Path(logs_dir).parent / "results.tsv"),
        program_md_path=str(Path(work) / "program.md"),
    )


def _read_runs(logs_dir):
    path = Path(logs_dir) / "runs.jsonl"
    if not path.exists():
        return []
    return [json.loads(ln) for ln in
            path.read_text(encoding="utf-8").splitlines() if ln.strip()]


# --- the three phases -------------------------------------------------------

def _run_c0(work):
    logs = Path(work) / "c0" / "logs"
    logs.mkdir(parents=True)
    _start_session(work, logs, arm="C0")
    for commit, src, pre, post, desc in _agent_trials():
        _run_trial(work, logs, commit, src, pre, post, desc)
    return _read_runs(logs)


def _bo(work, args, logs):
    return subprocess.run(
        [sys.executable, "tools/run_bo.py", *args, "--logs-dir", str(logs),
         "--results-tsv", str(Path(logs).parent / "results.tsv")],
        cwd=str(work), capture_output=True, text=True, timeout=600,
    )


def _run_c1(work):
    logs = Path(work) / "c1" / "logs"
    logs.mkdir(parents=True)
    _start_session(work, logs, arm="C1")
    # Two agent trials, then a budget-5 BO episode invoked by the script.
    for commit, src, pre, post, desc in _agent_trials()[:2]:
        _run_trial(work, logs, commit, src, pre, post, desc)

    # v1.1 (A1): exercise --specs (no session) and a refused out-of-spec call.
    specs = subprocess.run([sys.executable, "tools/run_bo.py", "--specs", "xgboost"],
                           cwd=str(work), capture_output=True, text=True)
    if specs.returncode != 0 or "legal search space" not in specs.stdout:
        raise RuntimeError(f"--specs failed: {specs.stdout}\n{specs.stderr}")
    rows_before = len(_read_runs(logs))
    refused = _bo(work, ["--family", "xgboost", "--budget", "5", "--space",
                         '{"max_depth": {"type": "int", "low": 50, "high": 60}}'], logs)
    if refused.returncode != 2 or "REFUSED" not in refused.stderr:
        raise RuntimeError(f"out-of-spec call not refused: rc={refused.returncode}\n"
                           f"{refused.stdout}\n{refused.stderr}")
    if len(_read_runs(logs)) != rows_before:
        raise RuntimeError("refused call consumed trials (should be zero)")

    space = json.dumps({"max_depth": {"type": "int", "low": 2, "high": 6},
                        "learning_rate": {"type": "float", "low": 0.02,
                                          "high": 0.3, "log": True}})
    result = _bo(work, ["--family", "xgboost", "--budget", "5", "--space", space], logs)
    if result.returncode != 0:
        raise RuntimeError(f"BO episode failed (rc={result.returncode}):\n"
                           f"{result.stdout}\n{result.stderr}")
    return _read_runs(logs)


def _run_r1(work):
    out = Path(work) / "r1"
    result = subprocess.run(
        [sys.executable, "tools/run_reference.py", "--method", "tpe",
         "--task", SMOKE_TASK, "--seed", "0", "--trials", "5", "--out", "r1"],
        cwd=str(work), capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"R1 reference failed (rc={result.returncode}):\n"
                           f"{result.stdout}\n{result.stderr}")
    return _read_runs(out / "logs")


def _ingest_and_notebook(work, logs_dir):
    from tools.extract_decisions import extract_decisions
    df = extract_decisions(logs_dir)
    (Path(work) / "analysis").mkdir(exist_ok=True)
    df.to_csv(Path(work) / "analysis" / "decisions.csv", index=False)

    import nbformat
    from nbclient import NotebookClient
    os.environ["LOGS_DIR"] = str(logs_dir)
    nb = nbformat.read(str(Path(work) / "analysis.ipynb"), as_version=4)
    NotebookClient(nb, timeout=180, kernel_name="python3",
                   resources={"metadata": {"path": str(work)}}).execute()
    return df


def run_smoke(work_dir):
    """Run the full C0 + C1 + R1 mini-grid under work_dir. Returns a summary."""
    work = _setup_workdir(work_dir)

    c0 = _run_c0(work)
    c1 = _run_c1(work)
    r1 = _run_r1(work)
    df = _ingest_and_notebook(work, Path(work) / "c1" / "logs")

    return {
        "work_dir": str(work),
        "c0_runs": c0,
        "c1_runs": c1,
        "r1_runs": r1,
        "c1_bo_trials": sum(1 for r in c1 if r.get("source") == "bo"),
        "c1_table_rows": len(df),
        "notebook_ok": True,
    }


def main(argv=None) -> int:
    work = tempfile.mkdtemp(prefix="autoresearch_smoke_")
    r = run_smoke(work)

    print(f"\nSmoke mini-grid under: {r['work_dir']}\n")
    print(f"  {'arm':<6}{'trials':>8}  detail")
    print(f"  {'C0':<6}{len(r['c0_runs']):>8}  agent-only mini-session")
    print(f"  {'C1':<6}{len(r['c1_runs']):>8}  "
          f"{r['c1_bo_trials']} of them a budget-5 BO episode")
    print(f"  {'R1':<6}{len(r['r1_runs']):>8}  scripted TPE reference")
    print(f"\n  C1 ingest table rows: {r['c1_table_rows']}   "
          f"notebook: {'OK' if r['notebook_ok'] else 'FAILED'}")

    ok = (len(r["c0_runs"]) == 3 and r["c1_bo_trials"] == 5
          and len(r["r1_runs"]) == 5 and r["notebook_ok"])
    print("\nSMOKE TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
