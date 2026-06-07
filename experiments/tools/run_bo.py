"""The BO tool — sealed, time-boxed Bayesian-optimization episodes (protocol §3.2-3.5).

The agent calls this like any other shell command:

    uv run python tools/run_bo.py --family xgboost --budget 10 --space '<json>'

It declares a model family, a bounded search space over that family's
hyperparameters, and a trial sub-budget n (5-15). The tool then runs a SEALED
TPE episode inside that box — the agent is not in the loop — and prints the best
configuration found, its val_logloss, and the per-trial trace. The n trials count
against the session's TRIAL_BUDGET like any other trial.

Mechanics (all decided in the protocol; do not re-litigate here):
  * Sampler: Optuna TPESampler(seed = session_seed + episode_index,
    n_startup_trials = max(3, budget // 3)). Pinned + seeded -> reproducible.
  * Each trial fits in a WORKER SUBPROCESS under the standard TIME_BUDGET
    watchdog (threading.Timer -> os._exit(124)), exactly like run_trial -> train.py.
    A timeout / crash / invalid-config-at-fit scores the task's penalty_logloss so
    TPE avoids the region, and the episode continues (protocol §3.3, §6.4).
  * Measurement is identical to an agent trial by construction: the family adapter
    builds the model with the same fixed preprocessing as the family baselines, and
    prepare.evaluate scores it on the same pinned split.
  * Ledger: one runs.jsonl row per trial (source="bo"), one results.tsv summary row
    per episode; full episode stdout -> logs/runs/<episode_id>.log (protocol §3.5).

Constraint violations are refused with a clear error and ZERO trials consumed.
This module never edits train.py and never changes families.
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import family_adapters as fa  # noqa: E402
import prepare  # noqa: E402
from logging_lib import (  # noqa: E402
    parse_summary_block,
    peak_rss_mb,
    record_bo_episode_summary,
    record_bo_trial,
)

_THIS_FILE = os.path.abspath(__file__)

BUDGET_MIN = 5
BUDGET_MAX = 15


class BORefusal(Exception):
    """A constraint violation: refuse the invocation, consume zero trials."""


# ---------------------------------------------------------------------------
# Session + ledger inspection
# ---------------------------------------------------------------------------


def load_session(logs_dir: str) -> dict:
    path = os.path.join(logs_dir, "session.json")
    if not os.path.exists(path):
        raise BORefusal(
            f"no session.json under {logs_dir!r}; run tools/start_session.py first "
            "(the BO tool needs the session's task and seed)."
        )
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def arm_enables_bo(session: dict) -> bool:
    """Whether this session's arm enables the bo capability.

    Stub until Step 8 stamps `capabilities` into session.json: a session with no
    `capabilities` key (pre-arms) is allowed; once present, `bo` must be in it.
    """
    capabilities = session.get("capabilities")
    if capabilities is None:
        return True
    return "bo" in capabilities


def _count_session_rows(runs_jsonl: str) -> int:
    if not os.path.exists(runs_jsonl):
        return 0
    with open(runs_jsonl, "r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def _count_prior_episodes(runs_jsonl: str) -> int:
    """Distinct bo_episode_id already recorded in this session."""
    if not os.path.exists(runs_jsonl):
        return 0
    seen = set()
    with open(runs_jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("source") == "bo" and row.get("bo_episode_id"):
                seen.add(row["bo_episode_id"])
    return len(seen)


# ---------------------------------------------------------------------------
# Space parsing / validation (keys against PARAM_SPECS; structure well-formed)
# ---------------------------------------------------------------------------


def parse_space(family: str, space: dict) -> dict:
    """Validate and normalize an agent-declared search space (the box).

    Refuses (zero trials) if a key is not a tunable param of the family, or an
    entry is structurally malformed. Bounds are NOT clamped to PARAM_SPECS: an
    out-of-spec value, if sampled, fails at fit time and takes the penalty
    (protocol §3.3) — key validity is the pre-flight contract (§3.2).
    """
    if not isinstance(space, dict) or not space:
        raise BORefusal("space must be a non-empty JSON object of param -> spec")
    specs = fa.PARAM_SPECS[family]
    normalized = {}
    for key, spec in space.items():
        if key not in specs:
            raise BORefusal(
                f"unknown hyperparameter {key!r} for family {family!r}; "
                f"allowed: {sorted(specs)}"
            )
        if not isinstance(spec, dict) or "type" not in spec:
            raise BORefusal(f"space[{key!r}] must be an object with a 'type'")
        kind = spec["type"]
        if kind in ("int", "float"):
            if "low" not in spec or "high" not in spec:
                raise BORefusal(f"space[{key!r}] ({kind}) needs 'low' and 'high'")
            low, high = spec["low"], spec["high"]
            if not isinstance(low, (int, float)) or not isinstance(high, (int, float)) \
                    or isinstance(low, bool) or isinstance(high, bool):
                raise BORefusal(f"space[{key!r}] low/high must be numbers")
            if low > high:
                raise BORefusal(f"space[{key!r}] low {low} > high {high}")
            normalized[key] = {
                "type": kind, "low": low, "high": high,
                "log": bool(spec.get("log", False)),
            }
        elif kind == "categorical":
            choices = spec.get("choices")
            if not isinstance(choices, list) or not choices:
                raise BORefusal(f"space[{key!r}] (categorical) needs non-empty 'choices'")
            normalized[key] = {"type": "categorical", "choices": list(choices)}
        else:
            raise BORefusal(
                f"space[{key!r}] has bad type {kind!r} (int|float|categorical)"
            )
    return normalized


# ---------------------------------------------------------------------------
# Pre-flight: validate the whole invocation, returning an episode plan
# ---------------------------------------------------------------------------


def plan_episode(logs_dir, family, budget, space, task=None) -> dict:
    """Validate every constraint and return the episode plan. Raises BORefusal.

    Writes nothing and fits nothing — a refusal here consumes zero trials.
    """
    session = load_session(logs_dir)
    if not arm_enables_bo(session):
        raise BORefusal(
            "this session's arm does not enable the 'bo' capability "
            f"(capabilities={session.get('capabilities')!r})."
        )

    if family not in prepare.ALLOWED_FAMILIES:
        raise BORefusal(
            f"family {family!r} not in ALLOWED_FAMILIES {prepare.ALLOWED_FAMILIES}"
        )

    if not isinstance(budget, int) or isinstance(budget, bool):
        raise BORefusal(f"budget must be an integer, got {budget!r}")
    if budget < BUDGET_MIN or budget > BUDGET_MAX:
        raise BORefusal(
            f"budget {budget} out of range [{BUDGET_MIN}, {BUDGET_MAX}]"
        )

    runs_jsonl = os.path.join(logs_dir, "runs.jsonl")
    used = _count_session_rows(runs_jsonl)
    remaining = prepare.TRIAL_BUDGET - used
    if budget > remaining:
        raise BORefusal(
            f"budget {budget} exceeds remaining session trials {remaining} "
            f"(TRIAL_BUDGET {prepare.TRIAL_BUDGET} - {used} used)"
        )

    normalized_space = parse_space(family, space)

    resolved_task = (
        os.environ.get("AUTORESEARCH_TASK") or task or session.get("task")
        or prepare.TASK_NAME
    )
    if resolved_task not in prepare._TASK_REGISTRY:
        raise BORefusal(f"unknown task {resolved_task!r}")

    session_seed = session.get("seed")
    if not isinstance(session_seed, int):
        raise BORefusal(f"session seed missing/invalid: {session_seed!r}")

    episode_index = _count_prior_episodes(runs_jsonl) + 1
    episode_id = f"bo-ep{episode_index:03d}"

    return {
        "task": resolved_task,
        "session_seed": session_seed,
        "episode_index": episode_index,
        "episode_id": episode_id,
        "remaining": remaining,
        "space": normalized_space,
        "penalty": prepare._TASK_REGISTRY[resolved_task]["penalty_logloss"],
    }


# ---------------------------------------------------------------------------
# Worker subprocess: fit ONE config under the watchdog, print a summary block
# ---------------------------------------------------------------------------


def _worker_main(family: str, task: str, config: dict) -> int:
    """Fit one config and print the print-contract summary. Run as a subprocess
    so the TIME_BUDGET watchdog can hard-kill a hang without taking the episode
    down (the parent sees exit 124 / no summary and scores the penalty)."""
    import threading

    def _on_timeout():
        print("TIMEOUT", flush=True)
        os._exit(124)

    t_start = time.perf_counter()
    X_train, y_train, X_val, y_val = prepare.load_task(task)
    model = fa.build(family, config)  # AdapterError here -> caught by caller -> exit 1

    watchdog = threading.Timer(prepare.TIME_BUDGET, _on_timeout)
    watchdog.daemon = True
    watchdog.start()
    try:
        t_fit = time.perf_counter()
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - t_fit
    finally:
        watchdog.cancel()

    metrics = prepare.evaluate(model, X_val, y_val)
    total_seconds = time.perf_counter() - t_start

    print("---")
    print(f"val_logloss:    {metrics['val_logloss']:.6f}")
    print(f"val_acc:        {metrics['val_acc']:.6f}")
    print(f"val_auc:        {metrics['val_auc']:.6f}")
    print(f"train_seconds:  {train_seconds:.3f}")
    print(f"total_seconds:  {total_seconds:.3f}")
    print(f"peak_mem_mb:    {peak_rss_mb():.1f}")
    print(f"model_family:   {family}")
    print(f"n_params:       0")
    print(f"task_name:      {task}")
    print("END_OF_TRIAL", flush=True)
    return 0


def _run_one_trial(family, task, config, penalty):
    """Run a single trial in a worker subprocess; return (summary, ok, raw_stdout).

    On timeout/crash/invalid-config the summary carries the task penalty so TPE
    avoids the region; ``ok`` is False.
    """
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, _THIS_FILE, "--worker",
         "--family", family, "--task", task, "--config", json.dumps(config)],
        capture_output=True, text=True,
    )
    elapsed = time.perf_counter() - t0
    stdout = proc.stdout or ""

    timed_out = proc.returncode == 124 or "TIMEOUT" in stdout
    summary = None
    if not timed_out:
        try:
            summary = parse_summary_block(stdout)
        except ValueError:
            summary = None

    if summary is None:
        reason = "timeout" if timed_out else "crash"
        summary = {
            "val_logloss": penalty,
            "val_acc": float("nan"),
            "val_auc": float("nan"),
            "train_seconds": float("nan"),
            "total_seconds": elapsed,
            "peak_mem_mb": float("nan"),
            "model_family": family,
            "n_params": 0,
            "task_name": task,
        }
        raw = stdout + (proc.stderr or "")
        return summary, False, f"[{reason}] {raw.strip()}"

    return summary, True, stdout.strip()


# ---------------------------------------------------------------------------
# The sealed episode
# ---------------------------------------------------------------------------


def _suggest(trial, space: dict) -> dict:
    config = {}
    for key, spec in space.items():
        if spec["type"] == "int":
            config[key] = trial.suggest_int(key, int(spec["low"]), int(spec["high"]),
                                            log=spec["log"])
        elif spec["type"] == "float":
            config[key] = trial.suggest_float(key, float(spec["low"]),
                                              float(spec["high"]), log=spec["log"])
        else:  # categorical
            config[key] = trial.suggest_categorical(key, spec["choices"])
    return config


def execute_episode(plan, family, budget, logs_dir, results_tsv, commit) -> dict:
    """Run the sealed TPE episode: record one bo row per trial + one summary row."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    task = plan["task"]
    penalty = plan["penalty"]
    episode_id = plan["episode_id"]
    space = plan["space"]

    sampler = optuna.samplers.TPESampler(
        seed=plan["session_seed"] + plan["episode_index"],
        n_startup_trials=max(3, budget // 3),
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)

    log_lines = [
        f"=== BO episode {episode_id} ===",
        f"family={family} task={task} budget={budget} "
        f"seed={plan['session_seed'] + plan['episode_index']}",
        f"space={json.dumps(space)}",
        "",
    ]
    trace = []
    best = {"val_logloss": float("inf"), "config": None, "summary": None}

    for i in range(1, budget + 1):
        trial = study.ask()
        config = _suggest(trial, space)
        summary, ok, raw = _run_one_trial(family, task, config, penalty)
        score = summary["val_logloss"]

        record_bo_trial(
            commit=commit, summary=summary, hyperparameters=config,
            bo_episode_id=episode_id, bo_trial_index=i, logs_dir=logs_dir,
        )
        study.tell(trial, score)

        flag = "ok" if ok else "FAIL(penalty)"
        line = f"trial {i:>2}/{budget}  {flag:<13} val_logloss={score:.6f}  {config}"
        trace.append(line)
        log_lines.append(line)
        if ok and score < best["val_logloss"]:
            best = {"val_logloss": score, "config": config, "summary": summary}

    if best["config"] is None:  # every trial failed
        best = {"val_logloss": penalty, "config": None, "summary": None}

    record_bo_episode_summary(
        commit=commit, task=task, model_family=family,
        val_logloss=best["val_logloss"], budget=budget,
        space_keys=list(space), results_tsv=results_tsv,
        best_summary=best["summary"],
    )

    # Persist full episode stdout.
    runs_log_dir = os.path.join(logs_dir, "runs")
    os.makedirs(runs_log_dir, exist_ok=True)
    with open(os.path.join(runs_log_dir, f"{episode_id}.log"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(log_lines) + "\n")

    return {
        "episode_id": episode_id,
        "best_config": best["config"],
        "best_val_logloss": best["val_logloss"],
        "best_summary": best["summary"],
        "trace": trace,
        "n_trials": budget,
    }


def _print_agent_summary(result, family, task):
    best = result["best_summary"]
    print(f"\n=== BO episode {result['episode_id']} complete "
          f"({result['n_trials']} trials, family={family}, task={task}) ===")
    if result["best_config"] is None:
        print("All trials failed (penalty). No configuration to adopt.")
    else:
        print(f"best val_logloss : {result['best_val_logloss']:.6f}")
        if best:
            print(f"best val_acc     : {best['val_acc']:.6f}")
            print(f"best val_auc     : {best['val_auc']:.6f}")
        print(f"best config      : {json.dumps(result['best_config'])}")
    print("\nper-trial trace:")
    for line in result["trace"]:
        print("  " + line)
    print(
        "\nTo adopt: edit train.py to this configuration, commit, and run "
        "run_trial.py as usual (that costs 1 trial and goes through the normal "
        "keep/discard mechanic)."
    )


def _resolve_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip() or "nogit"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family")
    parser.add_argument("--budget", type=int)
    parser.add_argument("--space")
    parser.add_argument("--logs-dir", default=os.environ.get("LOGS_DIR", "logs"))
    parser.add_argument("--results-tsv",
                        default=os.environ.get("RESULTS_TSV", "results.tsv"))
    # Internal worker mode (one config fit); not for direct agent use.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--task", help=argparse.SUPPRESS)
    parser.add_argument("--config", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker:
        try:
            return _worker_main(args.family, args.task, json.loads(args.config))
        except Exception as exc:  # noqa: BLE001 — any failure -> no summary -> penalty
            sys.stderr.write(f"[bo-worker] {type(exc).__name__}: {exc}\n")
            return 1

    if not args.family or args.budget is None or not args.space:
        parser.error("--family, --budget and --space are required")

    try:
        space = json.loads(args.space)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"REFUSED: --space is not valid JSON ({exc}). Zero trials.\n")
        return 2

    try:
        plan = plan_episode(args.logs_dir, args.family, args.budget, space)
    except BORefusal as exc:
        sys.stderr.write(f"REFUSED: {exc} Zero trials consumed.\n")
        return 2

    commit = _resolve_commit()
    result = execute_episode(
        plan, args.family, args.budget, args.logs_dir, args.results_tsv, commit,
    )
    _print_agent_summary(result, args.family, plan["task"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
