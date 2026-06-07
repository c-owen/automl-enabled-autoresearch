# Attribution

The code in this `baseline/` folder is a **derivative work** of Andrej Karpathy's
`autoresearch` project:

- Upstream: https://github.com/karpathy/autoresearch

## What was kept

The core structure of Karpathy's loop is preserved:

- a **locked harness** + a **mutable workpiece** the LLM edits,
- a **natural-language playbook** that drives the research loop,
- **git-branch hill-climbing** (commit to keep, `git reset --hard` to discard),
  with a `results.tsv` ledger of trials.

"The agent is the LLM, not code."

## What was changed

This is a port from Karpathy's original domain (LLM pretraining) to **supervised
tabular classification**:

- `prepare.py` — a locked harness that loads pinned, host-agnostic tabular tasks
  (UCI direct URLs, cached locally) and an authoritative `evaluate()` scoring
  function (`val_logloss`, binary + multiclass).
- `train.py` — the mutable workpiece, shipping as a scikit-learn / XGBoost
  classification baseline across four model families.
- `run_trial.py`, `logging_lib.py`, and the `tools/` operator workflow
  (start/ingest/end session, decision extraction) — the surrounding harness for
  running and analyzing baseline search sessions.

The purpose is to produce **LLM-only baseline search trajectories** for a
master's characterization study; a later AutoML decision layer (the actual
research contribution) is compared against these baselines.

## License note

As of this writing, the upstream repository specifies **no license file**. This
folder is shared for academic research and credits the original author above. If
you intend to reuse this code beyond that context, consult the upstream
repository and its author regarding licensing.
