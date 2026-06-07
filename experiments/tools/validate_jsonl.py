"""Schema validator for the runs.jsonl ledger.

Run:
    uv run python tools/validate_jsonl.py logs/runs.jsonl

Exits 0 if every line is a well-formed run row, 1 otherwise (printing the
offending line numbers and reasons). Importable as ``validate_jsonl`` /
``validate_run_row`` for tests.
"""

import json
import os
import sys

# Allow running as a standalone script (`python tools/validate_jsonl.py`): the
# script's own dir (tools/) lands on sys.path, not the repo root, so add it.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging_lib import ALL_STATUSES, KNOWN_SCHEMA_VERSIONS, VALID_SOURCES

_NUMBER = (int, float)

# Required key -> accepted python type(s). bool is excluded from numbers.
_REQUIRED_FIELDS = {
    "schema_version": int,
    "trial_id": int,
    "commit": str,
    "timestamp": str,
    "task": str,
    "model_family": str,
    "hyperparameters": dict,
    "val_logloss": _NUMBER,
    "val_acc": _NUMBER,
    "val_auc": _NUMBER,
    "train_seconds": _NUMBER,
    "total_seconds": _NUMBER,
    "peak_mem_mb": _NUMBER,
    "status": str,
    "description": str,
}


def validate_run_row(row) -> list:
    """Return a list of human-readable problems with ``row`` (empty == valid)."""
    errors = []
    if not isinstance(row, dict):
        return [f"row is not a JSON object (got {type(row).__name__})"]

    for key, types in _REQUIRED_FIELDS.items():
        if key not in row:
            errors.append(f"missing required key {key!r}")
            continue
        value = row[key]
        # bool is a subclass of int but never a valid value for these fields.
        if isinstance(value, bool) or not isinstance(value, types):
            type_names = getattr(types, "__name__", None) or "/".join(
                t.__name__ for t in types
            )
            errors.append(
                f"key {key!r} has type {type(value).__name__}, expected {type_names}"
            )

    if row.get("schema_version") not in (None,) + KNOWN_SCHEMA_VERSIONS:
        errors.append(
            f"schema_version {row.get('schema_version')!r} not in "
            f"{KNOWN_SCHEMA_VERSIONS}"
        )
    if isinstance(row.get("status"), str) and row["status"] not in ALL_STATUSES:
        errors.append(f"status {row['status']!r} not in {ALL_STATUSES}")

    # v2 provenance fields are optional (absent on v1 rows); validate when present.
    if "source" in row and row["source"] not in VALID_SOURCES:
        errors.append(f"source {row['source']!r} not in {VALID_SOURCES}")
    episode_id = row.get("bo_episode_id")
    if "bo_episode_id" in row and episode_id is not None and not isinstance(episode_id, str):
        errors.append(
            f"bo_episode_id has type {type(episode_id).__name__}, expected str/null"
        )
    trial_index = row.get("bo_trial_index")
    if "bo_trial_index" in row and trial_index is not None and (
        isinstance(trial_index, bool) or not isinstance(trial_index, int)
    ):
        errors.append(
            f"bo_trial_index has type {type(trial_index).__name__}, expected int/null"
        )
    return errors


def validate_jsonl(path) -> list:
    """Validate every non-blank line in ``path``. Returns a list of problems."""
    problems = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                problems.append(f"line {lineno}: invalid JSON ({exc})")
                continue
            for err in validate_run_row(row):
                problems.append(f"line {lineno}: {err}")
    return problems


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: validate_jsonl.py <runs.jsonl>", file=sys.stderr)
        return 2
    problems = validate_jsonl(argv[0])
    if problems:
        print(f"INVALID: {len(problems)} problem(s) in {argv[0]}")
        for p in problems:
            print(f"  {p}")
        return 1
    print(f"OK: {argv[0]} is a valid runs.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
