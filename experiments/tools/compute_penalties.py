"""Compute the pre-registered failure-penalty constant for each task.

Protocol §6.4: any failed trial (crash, timeout, invalid config) records
``val_logloss = 2 × prior-classifier logloss`` — where the *prior classifier*
predicts the training-set class frequencies for every validation row. The value
is **pre-registered**: it is computed once here and pasted as a literal
``penalty_logloss`` into ``prepare._TASK_REGISTRY``. It is never recomputed at
runtime (so a data change can't silently move the penalty).

Run it to (re)derive the constants and the numbers behind them:

    uv run python tools/compute_penalties.py
    uv run python tools/compute_penalties.py --task credit-g

The printed ``penalty_logloss`` is what belongs in the registry.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import prepare  # noqa: E402


def prior_logloss(task: str) -> float:
    """Logloss of the train-prior classifier on the pinned validation split.

    The prior classifier outputs the *training* class frequencies as a constant
    probability vector for every validation row — the trivial baseline a real
    model must beat. Returns its ``log_loss`` against the true val labels.
    """
    _X_train, y_train, _X_val, y_val = prepare.load_task(task)
    # Class order spanning both splits; priors come from TRAIN only.
    classes = sorted(set(map(_key, y_train)) | set(map(_key, y_val)))
    freqs = pd.Series(list(map(_key, y_train))).value_counts(normalize=True)
    priors = np.array([freqs.get(c, 0.0) for c in classes], dtype=float)
    proba = np.tile(priors, (len(y_val), 1))
    y_val_keys = list(map(_key, y_val))
    return float(log_loss(y_val_keys, proba, labels=classes))


def _key(label):
    """Stable, hashable, sortable key for a class label (handles int/str/np)."""
    return str(label)


def penalty_for(task: str) -> float:
    """The pre-registered penalty: 2 × prior-classifier logloss."""
    return 2.0 * prior_logloss(task)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=sorted(prepare._TASK_REGISTRY))
    args = parser.parse_args(argv)

    tasks = [args.task] if args.task else sorted(prepare._TASK_REGISTRY)
    print(f"{'task':<16} {'prior_logloss':>14} {'penalty (2x)':>14}")
    for task in tasks:
        pll = prior_logloss(task)
        print(f"{task:<16} {pll:>14.6f} {2.0 * pll:>14.6f}")
    print(
        "\nPaste each `penalty (2x)` as the task's `penalty_logloss` literal in "
        "prepare._TASK_REGISTRY."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
