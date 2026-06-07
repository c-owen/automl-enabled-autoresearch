"""Start an autoresearch session: assign a random initial model + run branch.

Run this once, by hand, to begin a session:

    uv run python tools/start_session.py                  # default task, random model
    uv run python tools/start_session.py --task credit-g --seed 7
    uv run python tools/start_session.py --model mlp      # force the starting family
    uv run python tools/start_session.py --model mlp --lock  # locked mlp-only run

It picks a starting model family at random (seeded, reproducible) — a control
for the first-mover bias where the LLM over-optimizes whichever family it tries
first — creates the run branch, writes logs/session.json, and prints the
one-liner to hand to the agent.

The branch is named with a local-time timestamp:  autoresearch/<YYYYMMDD-HHMMSS>-<model>
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prepare import ALLOWED_FAMILIES, TASK_NAME, TRIAL_BUDGET
from logging_lib import start_session as write_session_json
from arms import ARMS, capabilities_for, generate_playbook

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def choose_initial_model(seed: int, task: str = None) -> str:
    """Deterministically pick a starting family from ALLOWED_FAMILIES.

    Derives only from ``(seed, task)`` — never from the arm — so C0 and C1 with
    the same seed and task start from the same family (protocol §5 quasi-pairing).
    ``task=None`` keeps the legacy seed-only pick. (seed, task) is hashed with
    sha256 for a stable cross-process integer seed; Python's ``hash`` is salted.
    """
    if task is None:
        rng = random.Random(seed)
    else:
        digest = hashlib.sha256(f"{seed}:{task}".encode("utf-8")).hexdigest()
        rng = random.Random(int(digest, 16))
    return rng.choice(sorted(ALLOWED_FAMILIES))


def build_branch_name(when: datetime, model: str) -> str:
    return f"autoresearch/{when.strftime('%Y%m%d-%H%M%S')}-{model}"


def _seed_from(when: datetime) -> int:
    return int(when.strftime("%Y%m%d%H%M%S"))


def _git_head() -> str:
    """Short HEAD SHA of the harness, for per-run reproducibility (protocol §5)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip() or "nogit"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def archive_previous_session(logs_dir="logs", results_tsv="results.tsv",
                             archive_root="archive", when=None):
    """Move a prior session's artifacts out of the way so a new run starts clean.

    Archives logs/ + results.tsv, plus the rendered figures/ and analysis/ dirs
    (co-located with the logs, as the notebook writes them), so a finished run's
    plots are preserved and never need reproducing. Returns the archive path, or
    None if there was nothing to archive. Nothing is deleted — only moved.
    """
    # figures/ and analysis/ live next to the logs dir (see analysis.ipynb).
    base = os.path.dirname(os.path.normpath(logs_dir)) or "."
    extra = [os.path.join(base, name) for name in ("figures", "analysis")]

    has_logs = os.path.isdir(logs_dir) and bool(os.listdir(logs_dir))
    has_tsv = os.path.exists(results_tsv) and os.path.getsize(results_tsv) > 0
    present_extra = [d for d in extra if os.path.isdir(d) and os.listdir(d)]
    if not has_logs and not has_tsv and not present_extra:
        return None

    # Name the archive after the previous session if we can read it.
    name = None
    session_json = os.path.join(logs_dir, "session.json")
    if os.path.exists(session_json):
        try:
            with open(session_json, encoding="utf-8") as fh:
                name = json.load(fh).get("run_id")
        except (ValueError, OSError):
            name = None
    if not name:
        when = when or datetime.now()
        name = "session-" + when.strftime("%Y%m%d-%H%M%S")

    dest = os.path.join(archive_root, name)
    os.makedirs(dest, exist_ok=True)
    if has_logs:
        shutil.move(logs_dir, os.path.join(dest, "logs"))
    if has_tsv:
        shutil.move(results_tsv, os.path.join(dest, "results.tsv"))
    for d in present_extra:
        shutil.move(d, os.path.join(dest, os.path.basename(os.path.normpath(d))))
    return dest


def start_session(logs_dir="logs", task=None, seed=None, when=None,
                  locked=False, create_branch=True, archive=True,
                  results_tsv="results.tsv", archive_root="archive", model=None,
                  arm=None, program_md_path=None, model_id=None):
    """Assign the run's initial model + branch and write session.json.

    ``model`` forces a specific starting family (must be in ALLOWED_FAMILIES);
    if omitted, one is picked deterministically from ``(seed, task)`` — never the
    arm — the first-mover-bias control and the quasi-pairing key (protocol §5).
    ``arm`` (one of arms.ARMS) selects which capabilities are enabled: it stamps
    ``arm`` + ``capabilities`` into session.json and regenerates ``program.md`` =
    base + the arm's enabled playbook sections. Returns the session-metadata dict.
    """
    when = when or datetime.now()
    task = task or TASK_NAME
    if seed is None:
        seed = _seed_from(when)
    if model is not None:
        if model not in ALLOWED_FAMILIES:
            raise ValueError(
                f"model {model!r} not in ALLOWED_FAMILIES {ALLOWED_FAMILIES}"
            )
        chosen_model, model_source = model, "explicit"
    else:
        chosen_model, model_source = choose_initial_model(seed, task), "random"
    branch = build_branch_name(when, chosen_model)

    capabilities = None
    if arm is not None:
        if arm not in ARMS:
            raise ValueError(f"arm {arm!r} not in {sorted(ARMS)}")
        capabilities = capabilities_for(arm)

    archived = None
    if archive:
        archived = archive_previous_session(logs_dir, results_tsv, archive_root, when)

    if create_branch:
        subprocess.run(["git", "checkout", "-b", branch], check=True)

    # Generate the arm's playbook (program.md = base + enabled sections).
    if arm is not None:
        path = program_md_path or os.path.join(_REPO_ROOT, "program.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(generate_playbook(arm))

    meta = {
        "run_id": when.strftime("%Y%m%d-%H%M%S"),
        "branch": branch,
        "task": task,
        "arm": arm,
        "capabilities": capabilities,
        "initial_model": chosen_model,
        "model_source": model_source,   # "explicit" or "random"
        "seed": seed,
        "family_locked": bool(locked),
        "trial_budget": TRIAL_BUDGET,
        "harness_commit": _git_head(),   # pin the harness version (protocol §5)
        "agent_model": model_id,         # the LLM driving this run (operator-supplied)
        "started_at": when.isoformat(timespec="seconds"),
        "archived_previous": archived,
    }
    write_session_json(logs_dir, meta)
    return meta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS),
                        help="grid arm: C0 (LLM-only) or C1 (LLM+BO)")
    parser.add_argument("--task", default=None, help="task name (default: TASK_NAME)")
    parser.add_argument("--model", default=None, choices=sorted(ALLOWED_FAMILIES),
                        help="force the starting family (default: random, seeded)")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for the random model pick (default: from timestamp)")
    parser.add_argument("--model-id", default=None,
                        help="identity of the LLM driving this run (e.g. 'claude-opus-4-8'); "
                             "recorded in session.json — keep it fixed across the grid")
    parser.add_argument("--lock", action="store_true",
                        help="family-locked run (agent may only use the assigned family)")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--no-branch", action="store_true",
                        help="do not create the git branch (just write session.json)")
    parser.add_argument("--no-archive", action="store_true",
                        help="do not archive a previous session's ledger first")
    args = parser.parse_args(argv)

    meta = start_session(
        logs_dir=args.logs_dir, task=args.task, seed=args.seed, model=args.model,
        locked=args.lock, create_branch=not args.no_branch,
        archive=not args.no_archive, arm=args.arm, model_id=args.model_id,
    )
    if meta["archived_previous"]:
        print(f"(archived previous session -> {meta['archived_previous']})")

    lock_note = " (FAMILY-LOCKED — no swaps)" if meta["family_locked"] else ""
    print("\n=== autoresearch session started ===")
    print(f"arm           : {meta['arm']} (capabilities={meta['capabilities']})")
    print(f"branch        : {meta['branch']}")
    print(f"task          : {meta['task']}")
    print(f"initial model : {meta['initial_model']} ({meta['model_source']}){lock_note}")
    print(f"trial budget  : {meta['trial_budget']}")
    print(f"session.json  : {os.path.join(args.logs_dir, 'session.json')}")
    print("\nPaste this to the agent:")
    print("-" * 60)
    print(
        f"Read program.md and run a tabular autoresearch session. You are "
        f"already on the run branch. Your assigned starting family (in "
        f"logs/session.json) is '{meta['initial_model']}' on task "
        f"'{meta['task']}' - begin with a baseline of that family"
        + (", and use only that family for the whole run." if meta["family_locked"]
           else ", then search freely.")
        + " Run each trial via run_trial.py and commit every trial with a clear "
        "message. Stop at the trial budget."
    )
    print("-" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
