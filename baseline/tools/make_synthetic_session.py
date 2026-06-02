"""Generate a synthetic multi-trial session (runs + decisions + results.tsv).

Used by the analysis-notebook test and as a deterministic stand-in for a real
session. Writes through the real logging_lib recorder so the artifacts match the
production schema exactly.

    uv run python tools/make_synthetic_session.py [logs_dir] [results_tsv]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging_lib import record_trial

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


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    logs_dir = argv[0] if len(argv) >= 1 else "logs"
    results_tsv = argv[1] if len(argv) >= 2 else None
    n = build_synthetic_session(logs_dir, results_tsv)
    print(f"wrote synthetic session: {n} trials under {logs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
