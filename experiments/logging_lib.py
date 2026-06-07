"""Locked logging layer for the tabular-port harness.

This module is *not* edited by the LLM during a search session — it sits next
to prepare.py as a fixed piece. It owns the JSONL/TSV ledger contract so that a
free rewrite of train.py can never silently corrupt the run record: train.py
only has to emit the print-contract summary block to stdout; everything that
touches disk lives here (invoked by run_trial.py).

Artifacts produced (under <logs_dir>, default "logs/"):
    runs.jsonl          one structured row per trial
    runs/<commit>.log   full stdout per trial
    session.json        session-level metadata
and the Karpathy-style results.tsv ledger at the repo root.
"""

import ast
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone

# v2 (Step 6): runs.jsonl rows gain source / bo_episode_id / bo_trial_index to
# distinguish agent trials from BO-episode trials (protocol §3.5). v1 ledgers
# from archived sessions remain valid (forward-compat) — see KNOWN_SCHEMA_VERSIONS.
SCHEMA_VERSION = 2
KNOWN_SCHEMA_VERSIONS = (1, 2)

# The print contract emitted at the end of every trial (see tabular_port_plan
# section 3.2). Maps each required key to a coercion callable.
_SUMMARY_FIELDS = {
    "val_logloss": float,
    "val_acc": float,
    "val_auc": float,
    "train_seconds": float,
    "total_seconds": float,
    "peak_mem_mb": float,
    "model_family": str,
    "n_params": lambda v: int(float(v)),
    "task_name": str,
}

# Agent-trial statuses (what an agent trial row may carry; build_run_row guards
# this strictly). BO rows use their own statuses: each episode trial row is
# "bo_trial" in runs.jsonl, and the single results.tsv episode summary is
# "bo_episode". ALL_STATUSES is what the ledger validator accepts.
VALID_STATUSES = ("keep", "discard", "crash")
BO_STATUSES = ("bo_trial", "bo_episode")
ALL_STATUSES = VALID_STATUSES + BO_STATUSES

# Source of a runs.jsonl row (protocol §3.5). Absent on v1 rows (treated as
# "agent" by readers).
SOURCE_AGENT = "agent"
SOURCE_BO = "bo"
VALID_SOURCES = (SOURCE_AGENT, SOURCE_BO)

# Decision-record vocabulary (program.md "Decision capture").
LOCUS_VALUES = (
    "hyperparameter", "preprocessing", "model_family", "architecture",
    "validation_strategy", "other",
)
KEEP_DISCARD_VALUES = ("keep", "discard", "crash")

# results.tsv columns (tabular_port_plan section 4) mapped to their source key
# in the runs.jsonl row. mem_mb/trial_seconds are the ledger's names for the
# row's peak_mem_mb/total_seconds.
_TSV_COLUMN_SOURCES = {
    "commit": "commit",
    "task": "task",
    "model_family": "model_family",
    "val_logloss": "val_logloss",
    "val_acc": "val_acc",
    "val_auc": "val_auc",
    "mem_mb": "peak_mem_mb",
    "trial_seconds": "total_seconds",
    "status": "status",
    "description": "description",
}
_TSV_COLUMNS = list(_TSV_COLUMN_SOURCES)


# ---------------------------------------------------------------------------
# Print-contract parsing
# ---------------------------------------------------------------------------

def parse_summary_block(stdout: str) -> dict:
    """Parse the trailing ``---`` summary block from a trial's stdout.

    Returns a typed dict with every required print-contract field coerced to
    its type; any extra ``key: value`` lines in the block are preserved as raw
    strings (forward-compat). Raises ``ValueError`` if the block is missing or
    a required key is absent/unparseable.
    """
    lines = stdout.splitlines()

    # Use the LAST '---' marker so stray dashes in training logs don't fool us.
    start = None
    for i, line in enumerate(lines):
        if line.strip() == "---":
            start = i
    if start is None:
        raise ValueError("no summary block ('---' marker) found in stdout")

    raw = {}
    for line in lines[start + 1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if ":" not in stripped:
            # A non key:value line (e.g. the END_OF_TRIAL sentinel) ends the block.
            break
        key, _, value = stripped.partition(":")
        raw[key.strip()] = value.strip()

    result = {}
    for key, coerce in _SUMMARY_FIELDS.items():
        if key not in raw:
            raise ValueError(f"summary block missing required key: {key!r}")
        try:
            result[key] = coerce(raw[key])
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"could not parse {key}={raw[key]!r} as {coerce}"
            ) from exc

    # Keep unknown keys as-is for forward compatibility.
    for key, value in raw.items():
        result.setdefault(key, value)
    return result


# ---------------------------------------------------------------------------
# Atomic JSONL append
# ---------------------------------------------------------------------------

def append_run_row(jsonl_path, row: dict) -> None:
    """Append one JSON object as a line to ``jsonl_path``, atomically.

    Implemented as read-existing + write-all-to-tempfile + ``os.replace``, so a
    crash mid-write leaves the original file untouched — the ledger never holds
    a partial line.
    """
    jsonl_path = str(jsonl_path)
    directory = os.path.dirname(jsonl_path) or "."
    os.makedirs(directory, exist_ok=True)

    existing = ""
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            existing = fh.read()

    new_line = json.dumps(row, ensure_ascii=False) + "\n"

    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".jsonl.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(existing)
            if existing and not existing.endswith("\n"):
                fh.write("\n")
            fh.write(new_line)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, jsonl_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _append_tsv_row(tsv_path: str, row: dict) -> None:
    """Append one row to the Karpathy-style results.tsv (header on creation)."""
    directory = os.path.dirname(tsv_path) or "."
    os.makedirs(directory, exist_ok=True)
    write_header = not os.path.exists(tsv_path) or os.path.getsize(tsv_path) == 0
    with open(tsv_path, "a", encoding="utf-8", newline="") as fh:
        if write_header:
            fh.write("\t".join(_TSV_COLUMNS) + "\n")
        fh.write(
            "\t".join(
                str(row.get(source, "")) for source in _TSV_COLUMN_SOURCES.values()
            )
            + "\n"
        )


# ---------------------------------------------------------------------------
# Session + trial recording
# ---------------------------------------------------------------------------

def start_session(logs_dir, session_meta: dict) -> str:
    """Create ``<logs_dir>/session.json`` with session-level metadata."""
    logs_dir = str(logs_dir)
    os.makedirs(logs_dir, exist_ok=True)
    path = os.path.join(logs_dir, "session.json")
    payload = {"schema_version": SCHEMA_VERSION, **session_meta}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return path


def _next_trial_id(runs_jsonl_path: str) -> int:
    if not os.path.exists(runs_jsonl_path):
        return 1
    with open(runs_jsonl_path, "r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip()) + 1


def build_run_row(commit, summary, status, description, hyperparameters,
                  trial_id, timestamp) -> dict:
    """Assemble a runs.jsonl row from a parsed summary plus trial metadata."""
    if status not in VALID_STATUSES:
        raise ValueError(
            f"status {status!r} not in {VALID_STATUSES}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "trial_id": trial_id,
        "commit": commit,
        "timestamp": timestamp,
        "task": summary["task_name"],
        "model_family": summary["model_family"],
        "hyperparameters": dict(hyperparameters or {}),
        "val_logloss": summary["val_logloss"],
        "val_acc": summary["val_acc"],
        "val_auc": summary["val_auc"],
        "train_seconds": summary["train_seconds"],
        "total_seconds": summary["total_seconds"],
        "peak_mem_mb": summary["peak_mem_mb"],
        "n_params": summary.get("n_params"),
        "status": status,
        "description": description,
        # v2 provenance: an ordinary agent trial.
        "source": SOURCE_AGENT,
        "bo_episode_id": None,
        "bo_trial_index": None,
    }


def build_bo_run_row(commit, summary, hyperparameters, bo_episode_id,
                     bo_trial_index, trial_id, timestamp) -> dict:
    """Assemble a runs.jsonl row for a single BO-episode trial (protocol §3.5).

    Same measured fields as an agent row (``summary`` is the adapter trial's
    metrics, identical in shape to ``parse_summary_block``'s output), but tagged
    ``source="bo"`` / ``status="bo_trial"`` and carrying the episode id and the
    trial's index within the episode. ``commit`` is HEAD at the invocation; there
    is no per-trial commit association beyond that.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "trial_id": trial_id,
        "commit": commit,
        "timestamp": timestamp,
        "task": summary["task_name"],
        "model_family": summary["model_family"],
        "hyperparameters": dict(hyperparameters or {}),
        "val_logloss": summary["val_logloss"],
        "val_acc": summary["val_acc"],
        "val_auc": summary["val_auc"],
        "train_seconds": summary["train_seconds"],
        "total_seconds": summary["total_seconds"],
        "peak_mem_mb": summary["peak_mem_mb"],
        "n_params": summary.get("n_params"),
        "status": "bo_trial",
        "description": f"bo episode {bo_episode_id} trial {bo_trial_index}",
        "source": SOURCE_BO,
        "bo_episode_id": bo_episode_id,
        "bo_trial_index": bo_trial_index,
    }


def record_trial(commit, summary, status, description, hyperparameters,
                 logs_dir, run_log_path=None, trial_id=None, timestamp=None,
                 results_tsv="results.tsv", pre_trial_plan=None,
                 post_trial_reflection=None) -> dict:
    """Record one trial: append to runs.jsonl + results.tsv, persist run.log.

    If both ``pre_trial_plan`` and ``post_trial_reflection`` are supplied, also
    write a logs/decisions.jsonl row tied to this trial's commit/trial_id, with
    ``family_changed_from_prior`` derived from the previous trial's family.

    Returns the runs.jsonl row that was written.
    """
    logs_dir = str(logs_dir)
    runs_jsonl = os.path.join(logs_dir, "runs.jsonl")

    if trial_id is None:
        trial_id = _next_trial_id(runs_jsonl)
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    # Capture the prior family before this trial's row lands in the ledger.
    prior_family = _prior_model_family(runs_jsonl)

    row = build_run_row(
        commit, summary, status, description, hyperparameters,
        trial_id, timestamp,
    )

    append_run_row(runs_jsonl, row)
    _append_tsv_row(results_tsv, row)

    if pre_trial_plan is not None and post_trial_reflection is not None:
        family_changed = (
            prior_family is not None and prior_family != summary["model_family"]
        )
        write_decision_record(
            logs_dir, commit, trial_id, pre_trial_plan,
            post_trial_reflection, family_changed,
        )

    # Persist the full stdout under logs/runs/<commit>.log.
    runs_log_dir = os.path.join(logs_dir, "runs")
    os.makedirs(runs_log_dir, exist_ok=True)
    dest_log = os.path.join(runs_log_dir, f"{commit}.log")
    if run_log_path is not None and os.path.abspath(run_log_path) != os.path.abspath(dest_log):
        shutil.copyfile(run_log_path, dest_log)
    elif run_log_path is None and not os.path.exists(dest_log):
        # Guarantee the file exists even if the wrapper didn't tee stdout.
        open(dest_log, "a", encoding="utf-8").close()

    return row


# ---------------------------------------------------------------------------
# BO-episode recording (protocol §3.5)
# ---------------------------------------------------------------------------

def record_bo_trial(commit, summary, hyperparameters, bo_episode_id,
                    bo_trial_index, logs_dir, trial_id=None, timestamp=None) -> dict:
    """Append ONE BO-episode trial row to runs.jsonl (and nothing else).

    Unlike ``record_trial``, a BO trial writes no results.tsv row and no
    per-trial run log: results.tsv gets a single episode-summary row (see
    ``record_bo_episode_summary``) and the full episode stdout goes to
    ``logs/runs/<episode_id>.log`` (written by the BO tool). Episode trials draw
    1:1 from TRIAL_BUDGET, so they take a sequential ``trial_id`` like any trial.

    Returns the runs.jsonl row that was written.
    """
    logs_dir = str(logs_dir)
    runs_jsonl = os.path.join(logs_dir, "runs.jsonl")

    if trial_id is None:
        trial_id = _next_trial_id(runs_jsonl)
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    row = build_bo_run_row(
        commit, summary, hyperparameters, bo_episode_id, bo_trial_index,
        trial_id, timestamp,
    )
    append_run_row(runs_jsonl, row)
    return row


def record_bo_episode_summary(commit, task, model_family, val_logloss, budget,
                              space_keys, results_tsv="results.tsv",
                              best_summary=None) -> dict:
    """Append the SINGLE human-scannable results.tsv row for a BO episode.

    One row per episode (not one per trial): ``status="bo_episode"``,
    ``model_family`` = the episode's family, ``val_logloss`` = the episode best,
    auto description ``bo n=<budget> space=<declared space keys>``. The acc/auc/
    mem/seconds columns are filled from ``best_summary`` (the episode-best trial's
    metrics) when provided, else left blank.
    """
    best = best_summary or {}
    description = f"bo n={budget} space={','.join(space_keys)}"
    row = {
        "commit": commit,
        "task": task,
        "model_family": model_family,
        "val_logloss": val_logloss,
        "val_acc": best.get("val_acc", ""),
        "val_auc": best.get("val_auc", ""),
        "peak_mem_mb": best.get("peak_mem_mb", ""),
        "total_seconds": best.get("total_seconds", ""),
        "status": "bo_episode",
        "description": description,
    }
    _append_tsv_row(results_tsv, row)
    return row


# ---------------------------------------------------------------------------
# Decision records (logs/decisions.jsonl)
# ---------------------------------------------------------------------------

def validate_pre_trial_plan(plan: dict) -> list:
    """Return a list of problems with a pre-trial plan (empty == valid)."""
    errors = []
    if not isinstance(plan, dict):
        return [f"pre-trial plan is not an object (got {type(plan).__name__})"]
    if not isinstance(plan.get("family_chosen"), str) or not plan.get("family_chosen"):
        errors.append("family_chosen must be a non-empty string")
    locus = plan.get("locus_of_change")
    if locus not in LOCUS_VALUES:
        errors.append(f"locus_of_change {locus!r} not in {LOCUS_VALUES}")
    if not isinstance(plan.get("intent"), str) or not plan.get("intent"):
        errors.append("intent must be a non-empty string")
    return errors


def validate_post_trial_reflection(reflection: dict) -> list:
    """Return a list of problems with a post-trial reflection (empty == valid)."""
    errors = []
    if not isinstance(reflection, dict):
        return [f"reflection is not an object (got {type(reflection).__name__})"]
    kod = reflection.get("keep_or_discard")
    if kod not in KEEP_DISCARD_VALUES:
        errors.append(f"keep_or_discard {kod!r} not in {KEEP_DISCARD_VALUES}")
    if not isinstance(reflection.get("reason"), str) or not reflection.get("reason"):
        errors.append("reason must be a non-empty string")
    if not isinstance(reflection.get("surprise"), bool):
        errors.append("surprise must be a boolean")
    return errors


def build_decision_row(commit, trial_id, pre_trial_plan, post_trial_reflection,
                       family_changed_from_prior) -> dict:
    """Assemble one logs/decisions.jsonl row from a pre-plan + post-reflection.

    Validates both halves and raises ValueError on any problem.
    """
    pre_errors = validate_pre_trial_plan(pre_trial_plan)
    post_errors = validate_post_trial_reflection(post_trial_reflection)
    if pre_errors or post_errors:
        raise ValueError(
            "invalid decision record: " + "; ".join(pre_errors + post_errors)
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "trial_id": trial_id,
        "commit": commit,
        "family_chosen": pre_trial_plan["family_chosen"],
        "family_changed_from_prior": bool(family_changed_from_prior),
        "locus_of_change": pre_trial_plan["locus_of_change"],
        "intent": pre_trial_plan["intent"],
        "keep_or_discard": post_trial_reflection["keep_or_discard"],
        "reason": post_trial_reflection["reason"],
        "surprise": bool(post_trial_reflection["surprise"]),
    }


def write_decision_record(logs_dir, commit, trial_id, pre_trial_plan,
                          post_trial_reflection, family_changed_from_prior) -> dict:
    """Append one validated decision row to <logs_dir>/decisions.jsonl."""
    row = build_decision_row(
        commit, trial_id, pre_trial_plan, post_trial_reflection,
        family_changed_from_prior,
    )
    append_run_row(os.path.join(str(logs_dir), "decisions.jsonl"), row)
    return row


def _prior_model_family(runs_jsonl_path: str):
    """Model family of the most recently recorded trial, or None."""
    if not os.path.exists(runs_jsonl_path):
        return None
    last = None
    with open(runs_jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                last = line
    if last is None:
        return None
    try:
        return json.loads(last).get("model_family")
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Cross-platform peak memory (no third-party deps)
# ---------------------------------------------------------------------------

def peak_rss_mb() -> float:
    """Peak working-set memory of this process, in MB (Windows).

    Fills the ``peak_mem_mb`` print-contract field via the Windows psapi
    ``GetProcessMemoryInfo`` (PeakWorkingSetSize) — no third-party dependency.
    The harness targets Windows + Python; returns 0.0 if the query fails.
    """
    try:
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        # Signatures matter on 64-bit: without these the pseudo-handle is
        # truncated to 32 bits and the call fails.
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

        counters = _PMC()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return 0.0
        return counters.PeakWorkingSetSize / (1024 * 1024)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Hyperparameter extraction (AST parse of train.py's constant block)
# ---------------------------------------------------------------------------

# Uppercase module-level names that are harness contract, not tunable HPs.
_NON_HYPERPARAMETER_NAMES = {"MODEL", "TASK_NAME", "RANDOM_SEED", "VAL_FRAC",
                             "TIME_BUDGET", "TRIAL_BUDGET", "ALLOWED_FAMILIES",
                             "TIMEOUT_EXIT_CODE"}


def extract_hyperparameters(train_path: str) -> dict:
    """Best-effort: read module-level UPPER_CASE literal assignments from train.py.

    Used by run_trial.py to record the hyperparameters that produced a trial.
    Only literal values are captured (numbers, strings, bools, simple
    collections); anything non-literal is skipped. Harness-contract constants
    (MODEL, TASK_NAME, ...) are excluded.
    """
    with open(train_path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=train_path)

    hyperparameters = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            name = target.id
            if not name.isupper() or name in _NON_HYPERPARAMETER_NAMES:
                continue
            try:
                hyperparameters[name.lower()] = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                continue  # non-literal RHS — skip
    return hyperparameters
