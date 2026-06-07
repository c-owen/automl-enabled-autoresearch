"""End-of-session batch ingest — the one command you run after a session.

    uv run python tools/ingest_session.py            # logs/, repo=., -> analysis/decisions.csv

Reads the trial ledger (logs/runs.jsonl) and git, and writes a single per-trial
decision table. Keep/discard comes from git ancestry, intent from each trial's
commit message, locus from what changed — so the agent never has to log
anything beyond a clear commit message. Idempotent: re-run anytime.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from tools.extract_decisions import extract_decisions


def ingest(logs_dir="logs", repo_dir=".", out="analysis/decisions.csv", head="HEAD"):
    """Build the decision table and write it to ``out``. Returns the DataFrame."""
    df = extract_decisions(logs_dir, repo_dir=repo_dir, head=head)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out, index=False)
    return df


def summarize(df) -> str:
    n = len(df)
    kept = int((df["kept_on_branch"] == True).sum())      # noqa: E712
    discarded = int((df["kept_on_branch"] == False).sum())  # noqa: E712
    crashes = int((df["status"] == "crash").sum())
    lines = [
        f"trials      : {n}",
        f"kept        : {kept}   discarded: {discarded}   crashes: {crashes}",
        "by family   : " + ", ".join(
            f"{fam}={cnt}" for fam, cnt in df["model_family"].value_counts().items()
        ),
    ]
    # BO engagement (C1 sessions only): episodes, episode-trials, adoptions.
    if "source" in df.columns:
        bo = df[df["source"] == "bo"]
        if len(bo):
            n_episodes = bo["bo_episode_id"].nunique()
            n_adopt = int(df.get("adopted_from_episode", pd.Series(dtype=bool)).sum())
            lines.append(
                f"BO          : {n_episodes} episode(s), {len(bo)} episode-trials, "
                f"{n_adopt} adopted"
            )
    scored = df[df["val_logloss"].notna()]
    if len(scored):
        best = scored.loc[scored["val_logloss"].idxmin()]
        lines.append(
            f"best        : val_logloss={best['val_logloss']:.6f} "
            f"({best['model_family']}, trial {best['trial_id']})"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs_dir", nargs="?", default="logs")
    parser.add_argument("--repo", default=".", help="repo dir for git derivation")
    parser.add_argument("--head", default="HEAD",
                        help="ref to measure keep/discard against (e.g. a run's branch)")
    parser.add_argument("--out", default="analysis/decisions.csv")
    args = parser.parse_args(argv)

    df = ingest(args.logs_dir, repo_dir=args.repo, out=args.out, head=args.head)
    print(summarize(df))
    print(f"\nwrote {args.out} ({len(df)} trials). Open analysis.ipynb to visualize.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
