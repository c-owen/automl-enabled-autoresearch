"""Generate a synthetic multi-trial session (runs + decisions + results.tsv).

Used by the analysis-notebook test and as a deterministic stand-in for a real
session. Writes through the real logging_lib recorder so the artifacts match the
production schema exactly.

    uv run python tools/make_synthetic_session.py [logs_dir] [results_tsv]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging_lib import (
    record_bo_episode_summary,
    record_bo_trial,
    record_trial,
    start_session as write_session_json,
)

# (commit, family, status, val_logloss, locus, keep_or_discard, surprise)
_SCRIPT = [
    ("c0000001", "xgboost", "keep", 0.322, "hyperparameter", "keep", False),
    ("c0000002", "xgboost", "keep", 0.301, "hyperparameter", "keep", False),
    ("c0000003", "random_forest", "discard", 0.345, "model_family", "discard", True),
    ("c0000004", "logistic_regression", "discard", 0.358, "model_family", "discard", False),
    ("c0000005", "mlp", "keep", 0.294, "model_family", "keep", True),
    ("c0000006", "xgboost", "crash", float("nan"), "architecture", "crash", False),
]


def build_synthetic_session(logs_dir, results_tsv=None):
    """Write a 6-trial session under logs_dir; return the number of trials."""
    logs_dir = str(logs_dir)
    if results_tsv is None:
        results_tsv = os.path.join(logs_dir, "results.tsv")

    for i, (commit, family, status, logloss, locus, kod, surprise) in enumerate(
        _SCRIPT, start=1
    ):
        summary = {
            "val_logloss": logloss, "val_acc": 0.85, "val_auc": 0.91,
            "train_seconds": 1.0 + i, "total_seconds": 2.0 + i,
            "peak_mem_mb": 250.0, "model_family": family,
            "n_params": 1000 * i, "task_name": "adult",
        }
        pre = {
            "family_chosen": family,
            "locus_of_change": locus,
            "intent": f"Trial {i}: try {family} via {locus}.",
        }
        post = {
            "keep_or_discard": kod,
            "reason": f"Trial {i} outcome: {kod}.",
            "surprise": surprise,
        }
        record_trial(
            commit=commit, summary=summary, status=status,
            description=f"synthetic trial {i} ({family})",
            hyperparameters={"n_estimators": 100 + i * 10},
            logs_dir=logs_dir, trial_id=i,
            timestamp=f"2026-05-30T00:{i:02d}:00+00:00",
            results_tsv=results_tsv,
            pre_trial_plan=pre, post_trial_reflection=post,
        )
    return len(_SCRIPT)


# --- C1 (LLM+BO) synthetic session: agent trials + two sealed BO episodes ----

_TASK_C1 = "credit-g"

# Each episode is a list of (config, val_logloss); the lowest-logloss config is
# the episode best. Episode 1 is xgboost, episode 2 is random_forest.
_EPISODE_1 = [
    ({"n_estimators": 100, "max_depth": 3}, 0.40),
    ({"n_estimators": 150, "max_depth": 5}, 0.36),
    ({"n_estimators": 200, "max_depth": 4}, 0.30),  # best
    ({"n_estimators": 300, "max_depth": 6}, 0.33),
    ({"n_estimators": 120, "max_depth": 2}, 0.38),
]
_EPISODE_2 = [
    ({"n_estimators": 200, "max_depth": 12}, 0.42),
    ({"n_estimators": 400, "max_depth": 16}, 0.37),
    ({"n_estimators": 300, "max_depth": 8}, 0.35),   # best
    ({"n_estimators": 500, "max_depth": 20}, 0.39),
    ({"n_estimators": 250, "max_depth": 10}, 0.41),
]


def _summary(family, logloss, i, task=_TASK_C1):
    return {
        "val_logloss": logloss, "val_acc": 0.85, "val_auc": 0.91,
        "train_seconds": 1.0 + i, "total_seconds": 2.0 + i, "peak_mem_mb": 250.0,
        "model_family": family, "n_params": 1000 * i, "task_name": task,
    }


def build_synthetic_c1_session(logs_dir, results_tsv=None):
    """Write a 13-trial C1 session: agent trials + two BO episodes, one of which
    is adopted into a kept agent commit. Returns the number of trials."""
    logs_dir = str(logs_dir)
    if results_tsv is None:
        results_tsv = os.path.join(logs_dir, "results.tsv")

    write_session_json(logs_dir, {
        "run_id": "synthC1", "branch": "autoresearch/synthC1", "task": _TASK_C1,
        "arm": "C1", "capabilities": ["bo"], "initial_model": "xgboost",
        "model_source": "random", "seed": 7, "family_locked": False,
        "trial_budget": 50, "started_at": "2026-06-07T00:00:00",
    })

    def ts(i):
        return f"2026-06-07T00:{i:02d}:00+00:00"

    tid = 0

    # Trial 1: agent xgboost baseline (kept).
    tid += 1
    record_trial(
        commit="agent001", summary=_summary("xgboost", 0.40, tid), status="keep",
        description="baseline xgboost", hyperparameters={"n_estimators": 120, "max_depth": 3},
        logs_dir=logs_dir, trial_id=tid, timestamp=ts(tid), results_tsv=results_tsv,
    )

    # Episode 1 (xgboost), trials 2-6.
    for idx, (cfg, ll) in enumerate(_EPISODE_1, start=1):
        tid += 1
        record_bo_trial(
            commit="agent001", summary=_summary("xgboost", ll, tid),
            hyperparameters=cfg, bo_episode_id="bo-ep001", bo_trial_index=idx,
            logs_dir=logs_dir, trial_id=tid, timestamp=ts(tid),
        )
    best1_cfg, best1_ll = min(_EPISODE_1, key=lambda x: x[1])
    record_bo_episode_summary(
        commit="agent001", task=_TASK_C1, model_family="xgboost",
        val_logloss=best1_ll, budget=5, space_keys=["n_estimators", "max_depth"],
        results_tsv=results_tsv, best_summary=_summary("xgboost", best1_ll, 4),
    )

    # Trial 7: agent xgboost ADOPTS episode-1 best (kept).
    tid += 1
    record_trial(
        commit="agent007", summary=_summary("xgboost", best1_ll, tid), status="keep",
        description="adopt bo-ep001 best", hyperparameters=dict(best1_cfg),
        logs_dir=logs_dir, trial_id=tid, timestamp=ts(tid), results_tsv=results_tsv,
    )

    # Episode 2 (random_forest), trials 8-12.
    for idx, (cfg, ll) in enumerate(_EPISODE_2, start=1):
        tid += 1
        record_bo_trial(
            commit="agent007", summary=_summary("random_forest", ll, tid),
            hyperparameters=cfg, bo_episode_id="bo-ep002", bo_trial_index=idx,
            logs_dir=logs_dir, trial_id=tid, timestamp=ts(tid),
        )
    best2_cfg, best2_ll = min(_EPISODE_2, key=lambda x: x[1])
    record_bo_episode_summary(
        commit="agent007", task=_TASK_C1, model_family="random_forest",
        val_logloss=best2_ll, budget=5, space_keys=["n_estimators", "max_depth"],
        results_tsv=results_tsv, best_summary=_summary("random_forest", best2_ll, 3),
    )

    # Trial 13: agent mlp from scratch (not adopted; best overall, kept).
    tid += 1
    record_trial(
        commit="agent013", summary=_summary("mlp", 0.28, tid), status="keep",
        description="mlp from scratch", hyperparameters={"hidden_sizes": "128,64", "dropout": 0.1},
        logs_dir=logs_dir, trial_id=tid, timestamp=ts(tid), results_tsv=results_tsv,
    )
    return tid


def build_synthetic_c1_entry_voluntary(logs_dir, results_tsv=None):
    """C1 session with one ENTRY episode and one VOLUNTARY episode (protocol §6.3).

    Episode 1 (xgboost) follows a single baseline -> entry. Episode 2 (xgboost,
    after further hand-tuning) is a second episode in the same family -> voluntary.
    Includes an estimator_class on agent rows so family-integrity ingest is also
    exercised (all compliant here). Returns the trial count (9)."""
    logs_dir = str(logs_dir)
    if results_tsv is None:
        results_tsv = os.path.join(logs_dir, "results.tsv")

    write_session_json(logs_dir, {
        "run_id": "synthEV", "branch": "autoresearch/synthEV", "task": _TASK_C1,
        "arm": "C1", "capabilities": ["bo"], "initial_model": "xgboost",
        "model_source": "random", "seed": 1, "family_locked": False,
        "trial_budget": 50, "started_at": "2026-06-07T00:00:00",
    })

    def ts(i):
        return f"2026-06-07T00:{i:02d}:00+00:00"

    def agent(tid, ll, desc):
        s = _summary("xgboost", ll, tid)
        s["estimator_class"] = "XGBClassifier"
        record_trial(commit=f"ev{tid:03d}", summary=s, status="keep",
                     description=desc, hyperparameters={"max_depth": 4 + tid},
                     logs_dir=logs_dir, trial_id=tid, timestamp=ts(tid),
                     results_tsv=results_tsv)

    def episode(eid, first_tid, configs):
        for idx, (cfg, ll) in enumerate(configs, start=1):
            record_bo_trial(commit=f"ev{first_tid - 1:03d}",
                            summary=_summary("xgboost", ll, first_tid + idx - 1),
                            hyperparameters=cfg, bo_episode_id=eid,
                            bo_trial_index=idx, logs_dir=logs_dir,
                            trial_id=first_tid + idx - 1, timestamp=ts(first_tid + idx - 1))
        best_cfg, best_ll = min(configs, key=lambda x: x[1])
        record_bo_episode_summary(commit=f"ev{first_tid - 1:03d}", task=_TASK_C1,
                                  model_family="xgboost", val_logloss=best_ll,
                                  budget=len(configs), space_keys=["max_depth"],
                                  results_tsv=results_tsv,
                                  best_summary=_summary("xgboost", best_ll, first_tid))

    agent(1, 0.50, "baseline xgboost")                       # entry baseline
    episode("bo-ep001", 2, [({"max_depth": 3}, 0.46), ({"max_depth": 4}, 0.40),
                            ({"max_depth": 5}, 0.43)])        # ENTRY (best 0.40)
    agent(5, 0.42, "hand-tune xgboost")
    agent(6, 0.41, "hand-tune xgboost")
    episode("bo-ep002", 7, [({"max_depth": 4}, 0.44), ({"max_depth": 6}, 0.43),
                            ({"max_depth": 3}, 0.45)])        # VOLUNTARY
    return 9


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    logs_dir = argv[0] if len(argv) >= 1 else "logs"
    results_tsv = argv[1] if len(argv) >= 2 else None
    n = build_synthetic_session(logs_dir, results_tsv)
    print(f"wrote synthetic session: {n} trials under {logs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
