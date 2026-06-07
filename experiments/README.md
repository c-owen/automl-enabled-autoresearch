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

Data is **committed under `data/`** and served fully offline — `load_task` never
touches the network (a socket-blocking test enforces this). Each file is pinned
by SHA256; `tools/fetch_datasets.py` is the one-time populater. Switch tasks via
`start_session --task` (or the `AUTORESEARCH_TASK` env var).

| task             | rows   | features | classes | role |
|------------------|--------|----------|---------|------|
| `electricity`    | 45,312 | 8        | 2       | GBDT-friendly anchor |
| `adult` (default)| ~32.5k | 14       | 2       | imbalanced binary |
| `credit-g`       | 1,000  | 20       | 2       | small, hard-for-GBDT |
| `balance-scale`  | 625    | 4        | 3       | tiny, multiclass |
| `cnae-9`         | 1,080  | 856      | 9       | high-dim, multiclass |

## Quick start

**Requirements:** Windows, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies
uv sync

# 2. Verify the committed data layer (offline; checksums every task)
uv run python tools/fetch_datasets.py --verify-only

# 3. Start a session with an arm, then run a single trial
uv run python tools/start_session.py --arm C0 --task credit-g --no-branch
uv run python run_trial.py
```

## Running the grid

This harness runs four arms (see `OPERATING.md`): **C0** (LLM-only), **C1**
(LLM + the `run_bo` BO tool), and the scripted references **R1** (TPE) / **R2**
(random). A human starts an LLM session with `tools/start_session.py --arm
C0|C1`, which generates `program.md` for that arm and stamps the arm into
`session.json`. The agent reads `program.md` and runs the trial loop: edit
`train.py`, commit, `uv run python run_trial.py`, then keep (advance the branch)
or `git reset --hard` (discard). In C1 it may also call
`tools/run_bo.py` for a sealed Bayesian-optimization episode.

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
prepare.py        — locked harness: load_task (offline), evaluate, task registry (do not modify)
logging_lib.py    — locked ledger: print-contract parse, JSONL/TSV writes, BO/reference rows (do not modify)
family_adapters.py— locked: typed config -> fitted model per family (shared measurement plumbing)
arms.py           — arm/capability registry + playbook generation (C0/C1)
train.py          — the mutable workpiece (agent edits this)
run_trial.py      — trial wrapper that records to the ledger
program.md        — the agent playbook (generated per session by start_session)
playbook/         — base.md + sections/ (the source program.md is generated from)
data/             — committed, checksummed task data (offline source of truth)
tools/            — start_session, run_bo, run_reference[_batch], fetch_datasets,
                    compute_penalties, check_ceiling, ingest/extract, metrics, smoke_test, ...
tests/            — unit + integration (+ smoke); family baselines under tests/fixtures/
analysis.ipynb    — session visualization (progress, AUBC, BO engagement)
```

## License

MIT
