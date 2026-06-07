"""Attach a pre-plan + post-reflection to the most recent trial.

In the live loop the post-trial reflection is written *after* the trial runs,
so it can't be bundled at run_trial time. After writing both JSON files, run:

    uv run python tools/record_decision.py <pre_plan.json> <post_reflection.json>

This reads the most recent trial from logs/runs.jsonl (for its commit, trial_id
and model_family), derives family_changed_from_prior, and appends the row to
logs/decisions.jsonl.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging_lib import write_decision_record


def _read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) not in (2, 3):
        print(
            "usage: record_decision.py <pre_plan.json> <post_reflection.json> [logs_dir]",
            file=sys.stderr,
        )
        return 2
    pre_path, post_path = argv[0], argv[1]
    logs_dir = argv[2] if len(argv) == 3 else "logs"

    with open(pre_path, "r", encoding="utf-8") as fh:
        pre = json.load(fh)
    with open(post_path, "r", encoding="utf-8") as fh:
        post = json.load(fh)

    runs_path = os.path.join(logs_dir, "runs.jsonl")
    rows = _read_jsonl(runs_path)
    if not rows:
        print(f"no trials recorded in {runs_path}", file=sys.stderr)
        return 1

    latest = rows[-1]
    family_changed = (
        len(rows) >= 2 and rows[-2].get("model_family") != latest.get("model_family")
    )

    row = write_decision_record(
        logs_dir=logs_dir,
        commit=latest["commit"],
        trial_id=latest["trial_id"],
        pre_trial_plan=pre,
        post_trial_reflection=post,
        family_changed_from_prior=family_changed,
    )
    print(
        f"recorded decision for trial {row['trial_id']} (commit {row['commit']}): "
        f"{row['keep_or_discard']}, family_changed={row['family_changed_from_prior']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
