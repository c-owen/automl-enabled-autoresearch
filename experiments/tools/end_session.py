"""Wrap up a session: ingest the results, archive this run, return to the base.

The symmetric counterpart to start_session.py. Run it once when a run is done:

    uv run python tools/end_session.py            # ingest + archive + checkout main
    uv run python tools/end_session.py --stay     # ... but stay on the run branch
    uv run python tools/end_session.py --base X    # return to branch X instead of main

What it does, on the current run branch:
1. Ingests the trial ledger -> a decisions.csv (bundled into the archive).
2. Moves this run's logs/ + results.tsv to archive/<run-id>/ (nothing deleted).
3. Switches back to the base branch (default `main` — the experiments harness
   line), ready for the next start_session — unless the working tree is dirty
   (finish/commit your last trial first) or --stay is set.

(start_session also auto-archives, so forgetting this is harmless — but this
makes wrap-up explicit and returns you to a clean base branch.)
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.ingest_session import ingest, summarize
from tools.start_session import archive_previous_session


def _current_branch(repo_dir="."):
    try:
        return subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "--abbrev-ref", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _is_clean(repo_dir="."):
    # Only uncommitted *tracked* changes block a safe checkout; untracked files
    # (new tools, gitignored logs) are carried along harmlessly.
    try:
        out = subprocess.check_output(
            ["git", "-C", repo_dir, "status", "--porcelain", "--untracked-files=no"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return out.strip() == ""


def end_session(logs_dir="logs", results_tsv="results.tsv", archive_root="archive",
                repo_dir=".", to_master=True, base_branch="main"):
    """Ingest + archive the current run; optionally return to the base branch.

    ``base_branch`` defaults to ``main`` (the experiments harness line). Returns a
    result dict. The decisions table is written into the logs dir so it is
    preserved inside the archive.
    """
    branch = _current_branch(repo_dir)

    df = None
    if os.path.exists(os.path.join(logs_dir, "runs.jsonl")):
        df = ingest(logs_dir, repo_dir=repo_dir,
                    out=os.path.join(logs_dir, "decisions.csv"))

    archived = archive_previous_session(logs_dir, results_tsv, archive_root)

    switched = False
    skipped_reason = None
    if to_master:
        if branch == base_branch:
            skipped_reason = f"already on {base_branch}"
        elif not _is_clean(repo_dir):
            skipped_reason = "working tree not clean (commit/reset your last trial first)"
        else:
            subprocess.run(["git", "-C", repo_dir, "checkout", base_branch], check=True)
            switched = True

    return {
        "branch": branch,
        "base_branch": base_branch,
        "archived": archived,
        "n_trials": (0 if df is None else len(df)),
        "switched_to_master": switched,
        "skipped_reason": skipped_reason,
        "summary": (None if df is None else summarize(df)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--base", default="main",
                        help="base branch to return to (default: main)")
    parser.add_argument("--stay", action="store_true",
                        help="do not switch back to the base branch")
    args = parser.parse_args(argv)

    result = end_session(logs_dir=args.logs_dir, to_master=not args.stay,
                         base_branch=args.base)

    print(f"\n=== session wrapped up: {result['branch']} ===")
    if result["summary"]:
        print(result["summary"])
    if result["archived"]:
        print(f"archived    : {result['archived']}")
    if result["switched_to_master"]:
        print(f"now on      : {result['base_branch']} (ready for the next start_session)")
    elif result["skipped_reason"]:
        print(f"stayed on   : {result['branch']} ({result['skipped_reason']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
