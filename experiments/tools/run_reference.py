"""Reference arms R1 (TPE) and R2 (random) — scripted classical CASH search.

No LLM: a single Optuna study searches the joint CASH space (family as a
top-level categorical + conditional per-family hyperparameter subspaces drawn
from family_adapters.PARAM_SPECS, define-by-run) for TRIAL_BUDGET trials. Every
trial builds via the same family adapter and scores via the same locked
prepare.evaluate on the same pinned split as every agent/BO trial — so the
reference ledger is directly comparable. Failures take the task penalty.

    uv run python tools/run_reference.py --method tpe --task credit-g --seed 0 \
        --trials 50 --out reference_runs/tpe-credit-g-seed0

Output is written into a per-run --out dir (its own logs/runs.jsonl + results.tsv
+ reference.json), NEVER a live agent session dir: it refuses to write into a
directory that already contains a session.json.

Fits run in-process (the reference is scripted, supervised, and its configs are
bounded by PARAM_SPECS, so none approach TIME_BUDGET); a crashing/invalid config
takes the penalty and the search continues. The agent-facing BO tool, whose box
is untrusted, keeps the hard subprocess watchdog.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import family_adapters as fa  # noqa: E402
import prepare  # noqa: E402
from logging_lib import peak_rss_mb, record_reference_trial  # noqa: E402

METHODS = ("tpe", "random")


def _ensure_isolated_outdir(out_dir: str) -> None:
    """Refuse to write into a directory that holds a live agent session."""
    for candidate in (os.path.join(out_dir, "logs", "session.json"),
                      os.path.join(out_dir, "session.json")):
        if os.path.exists(candidate):
            raise SystemExit(
                f"refusing to write reference output into {out_dir!r}: it contains "
                f"a live session.json ({candidate}). Use a fresh --out dir."
            )


def _suggest_config(trial, family: str) -> dict:
    """Define-by-run per-family subspace; param names are family-namespaced so
    e.g. xgboost.max_depth and random_forest.max_depth are distinct dimensions."""
    config = {}
    for param, spec in fa.PARAM_SPECS[family].items():
        name = f"{family}__{param}"
        kind = spec["type"]
        if kind == "int":
            config[param] = trial.suggest_int(name, int(spec["low"]),
                                              int(spec["high"]), log=spec.get("log", False))
        elif kind == "float":
            config[param] = trial.suggest_float(name, float(spec["low"]),
                                                float(spec["high"]), log=spec.get("log", False))
        else:
            config[param] = trial.suggest_categorical(name, spec["choices"])
    return config


def _fit_score(family, config, task_data, task, penalty):
    """Build + fit + evaluate in-process. Returns a print-contract-shaped summary;
    any failure yields the penalty summary so the search avoids the region."""
    X_train, y_train, X_val, y_val = task_data
    t0 = time.perf_counter()
    try:
        model = fa.build(family, config)
        t_fit = time.perf_counter()
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - t_fit
        metrics = prepare.evaluate(model, X_val, y_val)
        return {
            "val_logloss": metrics["val_logloss"],
            "val_acc": metrics["val_acc"],
            "val_auc": metrics["val_auc"],
            "train_seconds": train_seconds,
            "total_seconds": time.perf_counter() - t0,
            "peak_mem_mb": peak_rss_mb(),
            "model_family": family,
            "n_params": 0,
            "task_name": task,
        }
    except Exception as exc:  # noqa: BLE001 — a bad config simply takes the penalty
        sys.stderr.write(f"[reference] {family} {config} failed: {exc!r}\n")
        return {
            "val_logloss": penalty,
            "val_acc": float("nan"),
            "val_auc": float("nan"),
            "train_seconds": float("nan"),
            "total_seconds": time.perf_counter() - t0,
            "peak_mem_mb": float("nan"),
            "model_family": family,
            "n_params": 0,
            "task_name": task,
        }


def run_reference(method, task, seed, trials, out_dir) -> dict:
    """Run one reference search and write its ledger into ``out_dir``."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if method not in METHODS:
        raise SystemExit(f"--method must be one of {METHODS}")
    if task not in prepare._TASK_REGISTRY:
        raise SystemExit(f"unknown task {task!r}")

    _ensure_isolated_outdir(out_dir)
    logs_dir = os.path.join(out_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    results_tsv = os.path.join(out_dir, "results.tsv")

    penalty = prepare._TASK_REGISTRY[task]["penalty_logloss"]
    task_data = prepare.load_task(task)

    sampler = (optuna.samplers.TPESampler(seed=seed) if method == "tpe"
               else optuna.samplers.RandomSampler(seed=seed))
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial):
        family = trial.suggest_categorical("family", sorted(fa.ALLOWED_FAMILIES))
        config = _suggest_config(trial, family)
        summary = _fit_score(family, config, task_data, task, penalty)
        record_reference_trial(
            summary=summary, hyperparameters={"family": family, **config},
            method=method, logs_dir=logs_dir, results_tsv=results_tsv,
            trial_id=trial.number + 1,
        )
        return summary["val_logloss"]

    study.optimize(objective, n_trials=trials)

    meta = {
        "method": method, "task": task, "seed": seed, "trials": trials,
        "best_value": study.best_value, "best_params": study.best_params,
    }
    with open(os.path.join(out_dir, "reference.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--task", required=True, choices=sorted(prepare._TASK_REGISTRY))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=prepare.TRIAL_BUDGET)
    parser.add_argument("--out", required=True, help="per-run output directory")
    args = parser.parse_args(argv)

    meta = run_reference(args.method, args.task, args.seed, args.trials, args.out)
    print(f"reference {meta['method']} task={meta['task']} seed={meta['seed']} "
          f"trials={meta['trials']} -> best val_logloss={meta['best_value']:.6f}")
    print(f"  best: {json.dumps(meta['best_params'])}")
    print(f"  ledger: {os.path.join(args.out, 'logs', 'runs.jsonl')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
