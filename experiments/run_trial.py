"""Cross-platform trial wrapper — the recorded entry point for a trial.

The playbook (program.md) tells the LLM to run trials via this wrapper rather
than calling ``train.py`` directly, because this is where the JSONL/TSV ledger
writes happen. train.py stays purely the experiment; all bookkeeping lives in
the locked logging_lib.

Usage:
    uv run python run_trial.py

Behaviour:
    1. Resolve the current commit (short SHA).
    2. Run ``python train.py``, teeing stdout+stderr to logs/runs/<commit>.log.
    3. Parse the print-contract summary block.
    4. Classify status: a TIMEOUT sentinel or exit code 124 -> crash/"timeout";
       a missing summary block -> crash/"no summary block"; otherwise the
       status/description supplied via env vars (default status "keep").
    5. Record the trial (runs.jsonl + results.tsv + persisted run.log).

The task a trial trains on is resolved from (in order): an explicit
AUTORESEARCH_TASK env var, else the session's recorded task in
<LOGS_DIR>/session.json. The resolved task is injected into train.py's
environment, so `start_session --task <name>` actually drives the run.

Environment variables (all optional):
    LOGS_DIR            logs directory (default "logs")
    RESULTS_TSV         results.tsv path (default "results.tsv")
    AUTORESEARCH_TASK   task override (else taken from session.json)
    TRIAL_STATUS        keep | discard | crash (default "keep" on success)
    TRIAL_DESCRIPTION   free-text trial description
    TRIAL_COMMIT        override the resolved commit (mainly for tests)
"""

import json
import os
import subprocess
import sys

from logging_lib import (
    extract_hyperparameters,
    parse_summary_block,
    record_trial,
    validate_post_trial_reflection,
    validate_pre_trial_plan,
)

TIMEOUT_EXIT_CODE = 124


def _load_decision_file(env_var, validator):
    """Load+validate a decision JSON file named by env_var, or return None."""
    path = os.environ.get(env_var)
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    problems = validator(data)
    if problems:
        sys.stderr.write(
            f"ERROR: {env_var}={path} is invalid: {'; '.join(problems)}\n"
        )
        sys.exit(3)
    return data


def _resolve_commit() -> str:
    override = os.environ.get("TRIAL_COMMIT")
    if override:
        return override
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def _resolve_task(logs_dir: str):
    """Task for this trial: explicit env wins, else the session's recorded task.

    Returns the task name, or None if neither is set (train.py then falls back
    to prepare.DEFAULT_TASK).
    """
    if os.environ.get("AUTORESEARCH_TASK"):
        return os.environ["AUTORESEARCH_TASK"]
    session_path = os.path.join(logs_dir, "session.json")
    if os.path.exists(session_path):
        try:
            with open(session_path, "r", encoding="utf-8") as fh:
                return json.load(fh).get("task")
        except (ValueError, OSError):
            return None
    return None


def _run_train(log_path: str, env=None) -> tuple[str, int]:
    """Run train.py, teeing combined stdout/stderr to log_path. Returns (text, rc)."""
    captured = []
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            [sys.executable, "train.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_file.write(line)
            captured.append(line)
        proc.wait()
    return "".join(captured), proc.returncode


def main() -> int:
    logs_dir = os.environ.get("LOGS_DIR", "logs")
    commit = _resolve_commit()

    runs_log_dir = os.path.join(logs_dir, "runs")
    os.makedirs(runs_log_dir, exist_ok=True)
    log_path = os.path.join(runs_log_dir, f"{commit}.log")

    # Resolve the session's task and inject it so train.py trains on it.
    task = _resolve_task(logs_dir)
    child_env = dict(os.environ)
    if task:
        child_env["AUTORESEARCH_TASK"] = task

    stdout, returncode = _run_train(log_path, env=child_env)

    status = os.environ.get("TRIAL_STATUS", "keep")
    description = os.environ.get("TRIAL_DESCRIPTION", "")

    timed_out = returncode == TIMEOUT_EXIT_CODE or "TIMEOUT" in stdout

    try:
        summary = parse_summary_block(stdout)
    except ValueError:
        summary = None

    if timed_out:
        status, description = "crash", description or "timeout"
    elif summary is None:
        status, description = "crash", description or "no summary block"

    if summary is None:
        # Crash with no parseable summary — record a minimal placeholder row so
        # the failure is still visible in the ledger.
        summary = {
            "task_name": task or "unknown",
            "model_family": "unknown",
            "val_logloss": float("nan"),
            "val_acc": float("nan"),
            "val_auc": float("nan"),
            "train_seconds": float("nan"),
            "total_seconds": float("nan"),
            "peak_mem_mb": float("nan"),
            "n_params": 0,
        }
        hyperparameters = {}
    else:
        hyperparameters = extract_hyperparameters("train.py")

    # Decision capture is optional. If both files are present at run time
    # (e.g. the smoke test, or a one-shot replay), the decision row is written
    # inline. In the live loop the post-reflection is written after the trial,
    # so the LLM finalizes it via tools/record_decision.py instead.
    pre_trial_plan = _load_decision_file(
        "PRE_TRIAL_PLAN_PATH", validate_pre_trial_plan
    )
    post_trial_reflection = _load_decision_file(
        "POST_TRIAL_REFLECTION_PATH", validate_post_trial_reflection
    )

    record_trial(
        commit=commit,
        summary=summary,
        status=status,
        description=description,
        hyperparameters=hyperparameters,
        logs_dir=logs_dir,
        run_log_path=log_path,
        results_tsv=os.environ.get("RESULTS_TSV", "results.tsv"),
        pre_trial_plan=pre_trial_plan,
        post_trial_reflection=post_trial_reflection,
    )

    wrote_decision = pre_trial_plan is not None and post_trial_reflection is not None
    print(
        f"[run_trial] recorded commit={commit} status={status}"
        f" decision={'yes' if wrote_decision else 'no'}"
    )
    return 0 if not timed_out else TIMEOUT_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
