"""Sweep a reference arm across seeds (protocol §5: 20 seeds per dataset).

Runs ``run_reference`` once per seed, sequentially (cheap, no LLM), into a nested
``<out_root>/<method>/<task>/seed<s>/`` tree — so the full grid is two top-level
folders (tpe, random), not a couple hundred sibling dirs.

    uv run python tools/run_reference_batch.py --method tpe --task credit-g \
        --seeds 0-19 --out-root reference_runs
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import prepare  # noqa: E402
from tools.run_reference import METHODS, run_reference  # noqa: E402


def _parse_seeds(spec: str) -> list:
    """Parse "0-19" (inclusive range) or "0,3,7" (explicit list)."""
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",") if s.strip()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--task", required=True, choices=sorted(prepare._TASK_REGISTRY))
    parser.add_argument("--seeds", default="0-19", help='e.g. "0-19" or "0,3,7"')
    parser.add_argument("--trials", type=int, default=prepare.TRIAL_BUDGET)
    parser.add_argument("--out-root", default="reference_runs")
    args = parser.parse_args(argv)

    seeds = _parse_seeds(args.seeds)
    print(f"sweeping {args.method} on {args.task}: {len(seeds)} seeds "
          f"x {args.trials} trials")
    for seed in seeds:
        # Nested layout: <out_root>/<method>/<task>/seed<N>/ (created recursively
        # by run_reference). Keeps the grid to a couple of top-level folders.
        out_dir = os.path.join(args.out_root, args.method, args.task, f"seed{seed}")
        meta = run_reference(args.method, args.task, seed, args.trials, out_dir)
        print(f"  seed {seed:>2}: best val_logloss={meta['best_value']:.6f} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
