# OPERATING.md — how to run a session (for the human)

`program.md` is the **agent's** instructions. This file is **yours**. The split:

- **The agent** does research only: edits `train.py`, commits each trial with a
  clear message, runs `run_trial.py`, keeps or `git reset`s. It logs nothing
  extra.
- **You** run two commands — one to start a session, one to ingest it at the
  end — and open the notebook. Everything else is derived from the trial ledger
  and git.

## One-time setup

```bash
uv sync
git checkout master      # the harness lives on master
```

## 1. Start a session

```bash
uv run python tools/start_session.py
```

This picks a **random starting model family** (the control for first-mover
bias), creates the run branch `autoresearch/<YYYYMMDD-HHMMSS>-<model>` (local
time), writes `logs/session.json`, and prints a one-liner to paste to the agent.

It also **auto-archives the previous session** — any existing `logs/` and
`results.tsv` are moved to `archive/<prev-run-id>/` first, so the new run starts
with a clean ledger and never appends onto an earlier session. (Use
`--no-archive` to skip.)

### Run modes (the key choice)

Two conditions, picked at start:

- **Free choice (default)** — the agent starts from the assigned family, then
  may switch families freely. Studies the LLM's own family-selection behavior.
- **Family-locked** (`--lock`) — the agent may use **only** the assigned family
  for the whole run. Gives each family a full, fair optimization budget — use
  this to measure per-family ceilings without the within-run drift toward a
  favorite (e.g. run #2 was assigned LR but spent 43/50 trials on xgboost).

The starting family is **random by default** (the bias control). To **choose**
it, pass `--model <family>`. For a fair per-family ceiling run, combine them:

```bash
uv run python tools/start_session.py --model mlp --lock     # locked, mlp only
```

### Choosing the task

Pass `--task` to `start_session` (default `adult`):

```bash
uv run python tools/start_session.py --task credit-g     # or bank-marketing
```

The task is recorded in `session.json`, and `run_trial.py` injects it into every
trial — so the whole run trains on it, no env var needed. (Setting the
`AUTORESEARCH_TASK` env var still works and overrides the session's task.)

All flags are in the **Command reference** at the bottom.

## 2. Let the agent work

Point a fresh agent (Claude Code / Codex / …) at the repo and paste the
one-liner `start_session` printed. The agent reads `program.md`, starts from the
assigned family, and runs the trial loop until `TRIAL_BUDGET` is reached.

Peek anytime (optional): `results.tsv` and `logs/runs.jsonl` update per trial.

## 3. Ingest + visualize (end of session)

Run **one** command from the run branch:

```bash
uv run python tools/ingest_session.py
```

This builds `analysis/decisions.csv` (one row per trial) by deriving:
- **keep / discard** from git ancestry (kept commits stayed on the branch;
  discarded ones were reset away),
- **intent** from each trial's commit message,
- **locus / family-change / deltas** from the ledger.

Then open **`analysis.ipynb`** (it reads `LOGS_DIR`, default `logs`) for the
progress curve, family/locus distributions, and the kept-vs-discarded scatter.
Figures are also saved next to the logs under `figures/`.

> Re-runnable: `ingest_session.py` is idempotent — run it mid-session for a peek
> or at the end; it regenerates the table from scratch.

## 4. Wrap up the session

When you're done with a run, close it out with one command (from the run
branch):

```bash
uv run python tools/end_session.py
```

This ingests the run, **archives** its `logs/` + `results.tsv` + `figures/` +
`analysis/` (plus the generated `decisions.csv`) to `archive/<run-id>/`, and
**switches you back to master** — clean and ready for the next `start_session`.
Use `--stay` to remain on the run branch. (So the rendered plots are preserved;
no need to reproduce them. To re-render anyway, open `analysis.ipynb` with
`LOGS_DIR=archive/<run-id>/logs` and `HEAD_REF=autoresearch/<run>`.)

If your working tree is dirty (an unfinished trial), it ingests and archives but
won't switch — commit or `git reset` your last trial first, then re-run it or
`git checkout master` yourself.

> The run branch itself (with its kept trial commits) stays put — that's the
> permanent record of the search. `end_session` only tidies the working-dir
> artifacts and returns you to master.

### The full cycle

```
start_session.py   ->   [agent runs trials]   ->   ingest_session.py + notebook   ->   end_session.py
   (new branch,                                       (analyze)                          (archive + back
    random model,                                                                         to master)
    auto-archive prev)
```

## Command reference

### `tools/start_session.py` — begin a run
| flag | default | meaning |
|---|---|---|
| `--model FAMILY` | random | force the starting family (xgboost / random_forest / logistic_regression / mlp); default is a seeded random pick |
| `--lock` | off | family-locked run (agent may use only the assigned family) |
| `--seed N` | from timestamp | RNG seed for the *random* model pick (reproducible assignment) |
| `--task NAME` | `adult` | dataset for the run (adult / credit-g / bank-marketing); recorded in session.json and used by every trial |
| `--logs-dir DIR` | `logs` | logs directory |
| `--no-archive` | off | do not archive the previous session first |
| `--no-branch` | off | do not create the git branch (just write session.json) |

### `tools/ingest_session.py` — build the decision table
| flag | default | meaning |
|---|---|---|
| `logs_dir` (positional) | `logs` | the session's logs dir |
| `--repo DIR` | `.` | repo dir for git derivation (keep/discard, intent, diff size) |
| `--head REF` | `HEAD` | ref to measure keep/discard against — pass a run's branch to analyze it without checking it out |
| `--out PATH` | `analysis/decisions.csv` | output CSV path |

### `tools/end_session.py` — wrap up a run
| flag | default | meaning |
|---|---|---|
| `--stay` | off | ingest + archive but do **not** switch back to master |
| `--logs-dir DIR` | `logs` | logs directory |

### `AUTORESEARCH_TASK` env var
Optional override of the session's task: `adult` (default), `credit-g`, or
`bank-marketing`. Normally you just use `start_session --task`; this env var
takes precedence over `session.json` if both are set.

## Notes

- `logs/`, `results.tsv`, `analysis/`, `figures/`, and `archive/` are gitignored
  — per-run artifacts, not committed. The per-commit `train.py` is recovered
  from git history.
- An **archived** run already has its `decisions.csv` bundled in
  `archive/<run-id>/logs/`. To regenerate it, point ingest at that logs dir and
  the run's branch: `ingest_session.py archive/<run-id>/logs --head autoresearch/<run>`.
- The agent never writes the ledger or any decision files; if you ever see it
  trying to, that's a sign `program.md` drifted.
