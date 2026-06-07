# OPERATING.md — running the Phase-2 grid (for the human)

This is the **experiments/** harness — it runs every arm of the BO tool study.
`program.md` is the **agent's** instructions (generated per session); this file is
**yours**. The pre-registered design lives in `../../experimental_protocol_bo.md`.

The split is unchanged from the baseline:
- **The agent** does research only: edits `train.py`, commits each trial with a
  clear message, runs `run_trial.py`, keeps or `git reset`s. In C1 it may also
  call the BO tool. It logs nothing extra.
- **You** start a session (now with an **arm**), let the agent run, then ingest
  and open the notebook. Everything else is derived from the ledger and git.

## The arms

| Arm | Driver | What it is |
|---|---|---|
| **C0** | agent + `program.md` | LLM-only control (no capabilities). |
| **C1** | agent + `program.md` + bo section | Identical to C0 **plus** the `run_bo` tool. |
| **R1** | `run_reference.py --method tpe` | Scripted TPE over the joint CASH space. No LLM. |
| **R2** | `run_reference.py --method random` | Scripted random search. No LLM. |

C0 and C1 are LLM sessions; R1/R2 are scripted. The **only** difference between
the C0 and C1 playbooks is the bo section (enforced by a test — protocol gate 4).

## One-time setup

```bash
uv sync
uv run python tools/fetch_datasets.py --verify-only   # confirm data/ checksums
```

`data/` is the committed source of truth (no network at run time). You only need
`tools/fetch_datasets.py` (the one network-touching script) if `data/` is missing
or you add a task.

## 1. Start an LLM session (C0 or C1)

```bash
uv run python tools/start_session.py --arm C1 --task credit-g --seed 3
```

`--arm` is **required**. This:
- picks the starting family deterministically from **(seed, task)** — never the
  arm, so C0 and C1 with the same seed+task start identically (quasi-pairing),
- creates the run branch `autoresearch/<YYYYMMDD-HHMMSS>-<model>`,
- writes `logs/session.json` (stamping `arm` + `capabilities`),
- **generates `program.md`** = base playbook + the arm's enabled sections,
- auto-archives any previous session's `logs/` + `results.tsv`,
- prints a one-liner to paste to the agent.

Run modes (`--lock`, `--model`) work as before; see the command reference.

## 2. Let the agent work

Paste the printed one-liner to a fresh agent. It reads `program.md`, starts from
the assigned family, and runs the trial loop to `TRIAL_BUDGET`. In **C1** it may
call the BO tool whenever it likes:

```bash
uv run python tools/run_bo.py --family xgboost --budget 10 --space '<json>'
```

The tool runs a sealed TPE episode inside the declared box (5–15 trials, drawn
from the session budget), prints the best config + trace, and tags the trials
`source="bo"` in the ledger. Constraint violations are refused with zero trials.
In a **C0** session the tool refuses (the arm doesn't enable it).

## 3. Reference runs (R1 / R2)

Scripted, no LLM — run them whenever:

```bash
# one run
uv run python tools/run_reference.py --method tpe --task credit-g --seed 0 \
    --trials 50 --out reference_runs/tpe-credit-g-seed0
# the 20-seed sweep for a (method, task)
uv run python tools/run_reference_batch.py --method random --task credit-g \
    --seeds 0-19 --out-root reference_runs
```

Each run writes its own `logs/runs.jsonl` + `results.tsv` + `reference.json` into
`--out`; it refuses to write into a directory holding a live `session.json`.

## 4. Ingest + visualize

```bash
uv run python tools/ingest_session.py            # -> analysis/decisions.csv
```

The table is **v2-aware**: BO-episode trials join tagged with `source` /
`bo_episode_id`, plus an `adopted_from_episode` flag (a kept agent commit that
matches an episode best). Then open **`analysis.ipynb`** (reads `LOGS_DIR`): it
adds AUBC (§6.1), a best-so-far curve (BO trials shaded), and a tool-engagement
summary (invocations, budgets, trial share, adoptions, incumbent share) for C1.

## 5. Wrap up

```bash
uv run python tools/end_session.py     # ingest + archive + back to the base branch
```

## The grid (protocol §5)

- **LLM arms:** floor of **3 seeds** per (arm × dataset); target **5 seeds** on
  the headline cells (`credit-g`, `cnae-9`, `electricity`). Same seed→family map
  across C0/C1.
- **Reference arms:** **20 seeds** per dataset.
- `TRIAL_BUDGET = 50`, `TIME_BUDGET = 300`s — unchanged.
- Datasets: `electricity`, `adult`, `credit-g`, `balance-scale`, `cnae-9`.

## The full cycle

```
start_session --arm   ->   [agent runs trials (+ run_bo in C1)]   ->   ingest + notebook   ->   end_session
run_reference[_batch] (R1/R2, independent, scripted)
```

## Command reference

### `tools/start_session.py`
| flag | default | meaning |
|---|---|---|
| `--arm ARM` | **required** | `C0` (LLM-only) or `C1` (LLM+BO) |
| `--task NAME` | `adult` | electricity / adult / credit-g / balance-scale / cnae-9 |
| `--model FAMILY` | from (seed, task) | force the starting family |
| `--seed N` | from timestamp | seed for the family pick + quasi-pairing |
| `--lock` | off | family-locked run |
| `--logs-dir DIR` | `logs` | logs directory |
| `--no-archive` / `--no-branch` | off | skip archiving / branch creation |

### `tools/run_bo.py` (C1 only)
`--family`, `--budget` (5–15), `--space '<json>'`, `--logs-dir`, `--results-tsv`.

### `tools/run_reference.py` / `run_reference_batch.py`
`--method tpe|random`, `--task`, `--seed`/`--seeds`, `--trials` (default 50),
`--out` / `--out-root`.

### `tools/fetch_datasets.py`
`--task NAME` (one task), `--verify-only` (no network; checksum committed files).

### Quick proof
```bash
uv run python tools/smoke_test.py     # C0 + C1 (with a BO episode) + R1, end to end
uv run pytest                          # unit + integration (smoke excluded)
uv run pytest -m smoke                 # the end-to-end mini-grid + ceiling smoke
```

## Notes

- `logs/`, `results.tsv`, `analysis/`, `figures/`, `archive/`, `reference_runs/`,
  and the generated `program.md` are gitignored. `data/` **is** committed.
- The agent never writes the ledger; the wrapper / tools own those writes.
- Don't change the agent model or harness git hash mid-grid (both recorded in
  `session.json`); a forced change restarts the grid or is reported as a limit.
