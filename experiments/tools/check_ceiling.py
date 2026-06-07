"""Family-ceiling check — protocol gate 2 (experiments_fork_execution_plan Step 5).

For each HARD-for-GBDT task, fit every family at its baseline default plus a small,
fixed, seeded "light manual tune" grid (<=10 configs/family, equal effort across
families — this is honest evidence, not a search tuned to make a non-GBDT win) and
report the best val_logloss per (task, family).

GATE: a non-GBDT family must beat xgboost on >= 2 of the 3 hard tasks. Exit 0 if
so, else exit 1 — in which case the dataset suite is revisited per protocol §8
BEFORE any further build effort.

    uv run python tools/check_ceiling.py            # the 3 hard tasks (the gate)
    uv run python tools/check_ceiling.py --task balance-scale   # one task (fast)
"""

import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import family_adapters as fa  # noqa: E402
import prepare  # noqa: E402

HARD_TASKS = ["credit-g", "balance-scale", "cnae-9"]
GBDT_FAMILY = "xgboost"

# Light manual tune per family: the baseline default ({}) plus a handful of
# sensible variations spanning each family's main knobs. Equal effort across
# families; all deterministic (RF random_state=0, MLP seeded, xgb/lr exact).
GRIDS = {
    "xgboost": [
        {},
        {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 500},
        {"max_depth": 4, "learning_rate": 0.1, "n_estimators": 400},
        {"max_depth": 8, "learning_rate": 0.05, "n_estimators": 500},
        {"max_depth": 6, "learning_rate": 0.3, "n_estimators": 200},
        {"max_depth": 2, "learning_rate": 0.1, "n_estimators": 800, "subsample": 0.8},
        {"max_depth": 10, "learning_rate": 0.03, "n_estimators": 600,
         "colsample_bytree": 0.7},
        {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 600,
         "reg_lambda": 5.0},
    ],
    "random_forest": [
        {},
        {"max_depth": 8},
        {"max_depth": 4},
        {"max_depth": 32, "min_samples_leaf": 1},
        {"n_estimators": 500, "max_features": "log2"},
        {"n_estimators": 500, "min_samples_leaf": 5},
        {"max_depth": 24, "min_samples_leaf": 1, "n_estimators": 600},
        {"max_depth": 12, "min_samples_leaf": 3, "max_features": "log2"},
    ],
    "logistic_regression": [
        {},
        {"C": 0.001},
        {"C": 0.01},
        {"C": 0.1},
        {"C": 10.0},
        {"C": 100.0},
        {"C": 0.03, "max_iter": 2000},
        {"C": 3.0, "max_iter": 2000},
    ],
    "mlp": [
        {},
        {"hidden_sizes": "256,128"},
        {"hidden_sizes": "64,32"},
        {"hidden_sizes": "128"},
        {"dropout": 0.0},
        {"dropout": 0.3, "max_epochs": 60},
        {"learning_rate": 0.003, "max_epochs": 60},
        {"hidden_sizes": "256,128", "dropout": 0.2, "weight_decay": 1e-3,
         "max_epochs": 60},
    ],
}


def _best_for_family(task, family, X_train, y_train, X_val, y_val, penalty):
    """Best (lowest) val_logloss over this family's grid; failures take penalty."""
    best = float("inf")
    best_cfg = None
    for cfg in GRIDS[family]:
        try:
            model = fa.build(family, cfg)
            model.fit(X_train, y_train)
            ll = prepare.evaluate(model, X_val, y_val)["val_logloss"]
        except Exception:  # noqa: BLE001 — a crashed config simply can't win
            sys.stderr.write(
                f"[warn] {task}/{family} cfg={cfg} failed:\n"
                f"{traceback.format_exc()}\n"
            )
            ll = penalty
        if ll < best:
            best, best_cfg = ll, cfg
    return best, best_cfg


def run_ceiling(tasks=None, families=None) -> dict:
    """Return ``{task: {family: {"best": logloss, "config": cfg}}}``."""
    tasks = tasks or HARD_TASKS
    families = families or fa.ALLOWED_FAMILIES
    results = {}
    for task in tasks:
        X_train, y_train, X_val, y_val = prepare.load_task(task)
        penalty = prepare._TASK_REGISTRY[task]["penalty_logloss"]
        results[task] = {}
        for family in families:
            best, cfg = _best_for_family(
                task, family, X_train, y_train, X_val, y_val, penalty
            )
            results[task][family] = {"best": best, "config": cfg}
    return results


def _format_table(results) -> str:
    families = fa.ALLOWED_FAMILIES
    header = f"{'task':<16}" + "".join(f"{fam:>22}" for fam in families) + \
        f"{'non-GBDT beats xgb?':>22}"
    lines = [header, "-" * len(header)]
    wins = 0
    for task, by_family in results.items():
        xgb = by_family[GBDT_FAMILY]["best"]
        best_non_gbdt = min(
            by_family[f]["best"] for f in families if f != GBDT_FAMILY
        )
        beat = best_non_gbdt < xgb
        wins += int(beat)
        cells = ""
        for fam in families:
            val = by_family[fam]["best"]
            mark = " *" if (fam != GBDT_FAMILY and val == best_non_gbdt and beat) else ""
            cells += f"{val:>20.5f}{mark:<2}"
        verdict = "YES" if beat else "no"
        lines.append(f"{task:<16}{cells}{verdict:>22}")
    lines.append("-" * len(header))
    lines.append(
        f"non-GBDT beats xgboost on {wins}/{len(results)} hard tasks "
        f"(gate needs >= 2). * = winning non-GBDT family."
    )
    return "\n".join(lines), wins


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--task", action="append", choices=HARD_TASKS,
        help="restrict to one or more hard tasks (default: all three = the gate)",
    )
    args = parser.parse_args(argv)
    tasks = args.task or HARD_TASKS

    print(f"Family-ceiling check on: {tasks}\n(this fits every family's grid; "
          "give it a few minutes)\n", flush=True)
    results = run_ceiling(tasks)
    table, wins = _format_table(results)
    print(table, flush=True)

    # The gate is defined over all three hard tasks; a partial run is informational.
    if set(tasks) != set(HARD_TASKS):
        print("\n[partial run — not the full gate]", flush=True)
        return 0

    passed = wins >= 2
    print(f"\nGATE {'PASS (exit 0)' if passed else 'FAIL (exit 1)'}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
