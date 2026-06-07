"""Join the trial ledger + decision records (+ git) into one structured table.

This is the analysis-time tool the characterization study consumes. It reads
logs/runs.jsonl and logs/decisions.jsonl (and, optionally, git diff stats) and
emits one row per trial.

    uv run python tools/extract_decisions.py logs --repo . --out analysis/decisions.csv

Importable as ``extract_decisions(logs_dir, repo_dir=None) -> pandas.DataFrame``.
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

COLUMNS = [
    "trial_id", "commit", "timestamp", "task", "model_family",
    "family_changed_from_prior", "locus_of_change", "intent",
    "val_logloss", "val_logloss_delta_from_parent", "status",
    "keep_or_discard", "reason", "surprise", "hyperparameters_json",
    "diff_size_lines", "kept_on_branch",
]


def _read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _git_diff_size(repo_dir, commit):
    """Lines changed in train.py between <commit>^ and <commit>, or None."""
    try:
        out = subprocess.check_output(
            ["git", "-C", repo_dir, "diff", "--numstat",
             f"{commit}^", commit, "--", "train.py"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    total = 0
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            total += int(parts[0]) + int(parts[1])
    return total


def _is_kept_on_branch(repo_dir, commit, head="HEAD"):
    """True if <commit> is an ancestor of HEAD (kept), False if orphaned
    (discarded via git reset), None if unresolvable.

    run_trial.py records every trial to runs.jsonl *before* the agent decides,
    and logs/ is gitignored, so a discarded trial's row survives while its
    commit is reset away. This recovers the real keep/discard outcome.
    """
    if not commit:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", repo_dir, "merge-base", "--is-ancestor", commit, head],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return None
    return {0: True, 1: False}.get(result.returncode)


def _git_commit_subject(repo_dir, commit):
    """First line of <commit>'s message (the agent's intent), or None."""
    if not commit:
        return None
    try:
        out = subprocess.check_output(
            ["git", "-C", repo_dir, "show", "-s", "--format=%s", commit],
            text=True, stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    subject = out.strip()
    return subject or None


def _derive_locus(family_changed, hp_changed):
    """Approximate locus_of_change from what moved between trials."""
    if family_changed:
        return "model_family"
    if hp_changed:
        return "hyperparameter"
    return "other"


def extract_decisions(logs_dir, repo_dir=None, head="HEAD"):
    """Return a per-trial DataFrame joining runs + decisions (+ git).

    For baseline runs (no decisions.jsonl), the subjective fields are derived:
    keep/discard from git ancestry (``kept_on_branch``), intent from the commit
    message, and locus from family/HP changes. Explicit decision records, when
    present, take precedence. ``head`` is the ref keep/discard is measured
    against — pass a run's branch to analyze it without checking it out.
    """
    runs = _read_jsonl(os.path.join(str(logs_dir), "runs.jsonl"))
    runs.sort(key=lambda r: r.get("trial_id", 0))
    decisions = {d.get("trial_id"): d for d in
                 _read_jsonl(os.path.join(str(logs_dir), "decisions.jsonl"))}

    records = []
    prev_family = None
    prev_logloss = None
    prev_hp = None
    for run in runs:
        decision = decisions.get(run.get("trial_id"), {})
        family = run.get("model_family")
        hp = run.get("hyperparameters", {})

        # Derive family change from the run order (always available); fall back
        # to the decision's own flag only when there is no prior trial.
        if prev_family is None:
            family_changed = bool(decision.get("family_changed_from_prior", False))
        else:
            family_changed = family != prev_family
        hp_changed = prev_hp is not None and hp != prev_hp

        logloss = run.get("val_logloss")
        if prev_logloss is None or logloss is None:
            delta = None
        else:
            delta = logloss - prev_logloss

        # Subjective fields: explicit decision record wins, else derive.
        locus = decision.get("locus_of_change") or _derive_locus(family_changed, hp_changed)
        intent = decision.get("intent")
        if intent is None and repo_dir:
            intent = _git_commit_subject(repo_dir, run.get("commit"))

        records.append({
            "trial_id": run.get("trial_id"),
            "commit": run.get("commit"),
            "timestamp": run.get("timestamp"),
            "task": run.get("task"),
            "model_family": family,
            "family_changed_from_prior": family_changed,
            "locus_of_change": locus,
            "intent": intent,
            "val_logloss": logloss,
            "val_logloss_delta_from_parent": delta,
            "status": run.get("status"),
            "keep_or_discard": decision.get("keep_or_discard"),
            "reason": decision.get("reason"),
            "surprise": decision.get("surprise"),
            "hyperparameters_json": json.dumps(hp),
            "diff_size_lines": (
                _git_diff_size(repo_dir, run.get("commit")) if repo_dir else None
            ),
            "kept_on_branch": (
                _is_kept_on_branch(repo_dir, run.get("commit"), head) if repo_dir else None
            ),
        })
        prev_family = family
        # Track the parent score/HP from the previous trial that wasn't a crash.
        if run.get("status") != "crash" and logloss is not None:
            prev_logloss = logloss
            prev_hp = hp

    return pd.DataFrame(records, columns=COLUMNS)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs_dir", nargs="?", default="logs")
    parser.add_argument("--repo", default=None,
                        help="repo dir for git diff sizes (optional)")
    parser.add_argument("--head", default="HEAD",
                        help="ref to measure keep/discard against (default HEAD)")
    parser.add_argument("--out", default="analysis/decisions.csv")
    args = parser.parse_args(argv)

    df = extract_decisions(args.logs_dir, repo_dir=args.repo, head=args.head)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(df)} trials, {len(df.columns)} columns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
