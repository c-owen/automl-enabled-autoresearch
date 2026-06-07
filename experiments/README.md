# autoresearch — tabular classification

> **Phase-2 grid harness.** This `experiments/` tree runs every grid arm (C0, C1,
> R1, R2) of the BO tool study; the sibling `baseline/` is frozen as the historical
> pilot artifact and is never modified. The pre-registered design lives in
> `../../experimental_protocol_bo.md` (the protocol), executed via
> `../../experiments_fork_execution_plan.md`.

An experiment in letting an LLM run its own research, ported from Karpathy's
[autoresearch](https://github.com/karpathy/autoresearch) (LLM pretraining) to
**supervised tabular classification**. The agent edits one file, runs a trial,
checks whether validation loss improved, keeps or discards, and repeats — a
git-branch hill-climb over model family and hyperparameters.

This is the harness that produces the LLM-only baseline trajectories for the
autoresearch characterization study.

## How it works

Three files matter, mirroring the original's locked-harness / mutable-workpiece
/ NL-playbook structure:

- **`prepare.py`** — locked harness. Task loading (`load_task`), the pinned
  train/val split, and the authoritative metric (`evaluate` →
  `val_logloss` / `val_acc` / `val_auc`). **Not modified by the agent.**
- **`train.py`** — the single file the agent edits: preprocessing, model
  family, hyperparameters, fit loop. Everything here is fair game.
- **`program.md`** — the natural-language playbook the agent follows.

Supporting the loop: **`logging_lib.py`** (locked JSONL/TSV ledger + decision
records) and **`run_trial.py`** (the wrapper that runs a trial and records it).

The selection metric is **`val_logloss`** (lower is better). Each trial is
capped at a 5-minute wall-clock budget; a session runs for `TRIAL_BUDGET`
trials.

### Model families

The agent searches across four families (`MODEL` must be one of these, enforced
by an assertion in `train.py`):

```
xgboost · random_forest · logistic_regression · mlp
```

### Tasks

Data is fetched from pinned, **host-agnostic** direct URLs and cached locally
(no dependence on any single dataset platform). Switch tasks via the
`AUTORESEARCH_TASK` env var:

| task             | rows   | features      | positive rate |
|------------------|--------|---------------|---------------|
| `adult` (default)| ~32.5k | 8 cat / 6 num | ~24%          |
| `credit-g`       | 1k     | 13 cat / 7 num| 30%           |
| `bank-marketing` | 4.5k   | 10 cat / 6 num| ~11.5%        |

## Quick start

**Requirements:** Windows, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies
uv sync

# 2. Verify the harness (first call fetches + caches the task)
uv run python -c "from prepare import load_task; print([a.shape for a in load_task()])"

# 3. Run a single trial end-to-end (records it to the ledger)
uv run python run_trial.py
```

## Running the agent

Point your coding agent at `program.md` and let it run the trial loop. Each
trial: write a pre-trial plan, edit `train.py`, commit, run
`uv run python run_trial.py`, write a post-trial reflection, then keep (advance
the branch) or `git reset --hard` (discard).

## What gets logged

```
logs/
├── runs.jsonl          # one structured row per trial
├── decisions.jsonl     # the LLM's pre-trial plan + post-trial reflection
├── runs/<commit>.log   # full stdout per trial
└── session.json        # session metadata
results.tsv             # Karpathy-style ledger
```

`logs/` and `results.tsv` are gitignored; the per-commit `train.py` is recovered
from git history.

## Analysis

```bash
# Join runs + decisions (+ git diff stats) into one table
uv run python tools/extract_decisions.py logs --repo . --out analysis/decisions.csv

# Render the session (progress curve, family/locus distributions, decisions)
#   open analysis.ipynb (reads LOGS_DIR, default "logs")

# Full end-to-end "is this working?" check (3-trial mini-session)
uv run python tools/smoke_test.py
```

## Tests

```bash
uv run pytest                 # all unit + integration (smoke excluded)
uv run pytest -m unit         # fast, no subprocess/network
uv run pytest -m integration  # runs train.py subprocesses
uv run pytest -m smoke        # the end-to-end mini-session
```

## Project structure

```
prepare.py        — locked harness: load_task, evaluate, task registry (do not modify)
logging_lib.py    — locked ledger: print-contract parse, JSONL/TSV writes, decisions (do not modify)
train.py          — the mutable workpiece (agent edits this)
run_trial.py      — trial wrapper that records to the ledger
program.md        — the agent playbook
tools/            — validate_jsonl, extract_decisions, record_decision, smoke_test, ...
tests/            — unit + integration; family baselines under tests/fixtures/
analysis.ipynb    — session visualization
```

## License

MIT
