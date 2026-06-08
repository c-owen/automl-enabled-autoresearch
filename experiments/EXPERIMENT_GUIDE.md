# The BO Tool Study — Build Report & Run Guide

This document has two parts:

- **Part A — What has been built.** A short context section, then each piece of
  the harness explained in a bit more depth.
- **Part B — How to run the experiments.** A hand-holding, step-by-step runbook
  for every arm of the study, plus a complete grid checklist.

Everything here lives in `experiments/` and is run from inside that folder. The
pre-registered design is `../../experimental_protocol_bo.md` (the *protocol*);
the build plan is `../../experiments_fork_execution_plan.md`. `OPERATING.md` is
the terse command reference; this guide is the narrated version.

---

# Part A — What has been built

## At a glance

- **The question.** Does giving the autoresearch LLM agent a callable,
  time-boxed Bayesian-optimization (BO) tool improve its search at a matched
  trial budget, versus the LLM working alone? And does any benefit concentrate
  on tasks where boosted trees (GBDTs) are *not* the right answer?
- **Four arms.** **C0** = LLM only (the control). **C1** = the *same* LLM setup
  *plus* a `run_bo` tool. **R1** = a scripted TPE optimizer (no LLM). **R2** = a
  scripted random search (no LLM). C0 vs C1 isolates "what does the tool add?";
  R1/R2 are the classical yardsticks.
- **Five datasets**, deliberately split into GBDT-friendly (`electricity`,
  `adult`) and hard-for-GBDT (`credit-g`, `balance-scale`, `cnae-9`).
- **One codebase, modular by arm.** No copies of the harness — an "arm" is just a
  named set of enabled capabilities. C0 and C1 differ by exactly one playbook
  section (machine-checked).
- **Built and verified.** Steps 1–11 of the plan are done and merged to `main`;
  **155 automated tests pass**; the end-to-end smoke test passes; all six
  protocol setup gates are green — including the key result that a non-GBDT model
  beats XGBoost on **all three** hard datasets, so there is real room for the
  tool to help.
- **Not yet done (this is the next phase, and the subject of Part B):** actually
  *running* the grid — the LLM sessions and the reference sweeps — and analyzing
  the results.

## A.1 The study and its hypotheses

The agent does ML research on itself: it edits a training script (`train.py`),
runs one trial, keeps the change if validation loss improved or discards it
otherwise, and repeats — hill-climbing a git branch for a fixed budget of 50
trials. The selection metric is **`val_logloss`** (lower is better) on a pinned
validation split.

The study asks whether a BO tool helps. Three pre-registered hypotheses:

- **H1 (efficacy):** on hard-for-GBDT tasks, C1 reaches a lower best
  `val_logloss` and a lower **AUBC** (area under the best-so-far curve) than C0.
- **H2 (interaction):** the C1−C0 improvement is *larger* on hard tasks than on
  GBDT-friendly tasks (where the agent's habit of drifting to XGBoost is already
  the right call, so there's little to gain).
- **H3 (mechanism):** C1 explores more model families and stagnates less,
  because a cheap BO episode lowers the cost of seriously probing a neglected
  family before giving up on it.

A null result is still a result: "the agent never used the tool" (no uptake) and
"the agent used it but it didn't help" (no utility) are *different* findings, so
the tool is described to the agent neutrally — never recommended.

## A.2 The four arms

| Arm | Driver | One-line description |
|---|---|---|
| **C0** | LLM + `program.md` | The LLM-only baseline. No capabilities. |
| **C1** | LLM + `program.md` + bo section | Identical to C0 in every respect except the `run_bo` tool exists and is documented. |
| **R1** | `run_reference.py --method tpe` | Optuna TPE over the joint "which family + which hyperparameters" space. The classical reference. |
| **R2** | `run_reference.py --method random` | Random search over the same space. The floor. |

C0 and C1 are LLM sessions (you drive an agent). R1 and R2 are scripts you launch
and walk away from. The protocol's central control is that **the only difference
between C0 and C1 is whether the tool exists** — enforced by a test that the two
playbooks differ by exactly the BO section and nothing else.

## A.3 The five datasets

| Task | Rows | Features | Classes | Role |
|---|---|---|---|---|
| `electricity` | 45,312 | 8 | 2 | GBDT-friendly anchor ("drift is correct") |
| `adult` | ~32,500 | 14 | 2 | imbalanced binary; continuity with the pilot |
| `credit-g` | 1,000 | 20 | 2 | small; **hard for GBDTs** |
| `balance-scale` | 625 | 4 | 3 | tiny, multiclass; **hard for GBDTs** |
| `cnae-9` | 1,080 | 856 | 9 | high-dimensional, multiclass; **hard for GBDTs** |

The hard tasks are where a neural net or a different family tends to win in the
literature (TabZilla). They are the cells where the tool has the most to prove.

## A.4 The offline data layer

All task data is **committed** under `data/`, byte-pinned by SHA256, and served
fully offline — `load_task()` never touches the network (a test disables sockets
and confirms loading still works). The only code that ever reaches the network is
the one-time `tools/fetch_datasets.py`, which downloads each file from its pinned
source and verifies the checksum. The data files are marked binary in
`.gitattributes` so Windows line-ending conversion can't silently change their
bytes (which would break the checksums on a fresh checkout). Net effect: the
exact same data, reproducibly, on any machine, with no live dependency on UCI or
OpenML.

## A.5 Family adapters (the measurement layer)

`family_adapters.py` is the shared "build a model from a config" layer used by
the BO tool, the reference arms, and the ceiling check. For each of the four
families (`xgboost`, `random_forest`, `logistic_regression`, `mlp`) it knows:

- `PARAM_SPECS[family]` — the typed, bounded set of hyperparameters that may be
  searched (so an agent-declared search box can be validated);
- `build(family, config)` — produces an unfitted model using **exactly the same
  preprocessing** as the frozen baseline reference models (imputation, scaling,
  one-hot encoding, category dtypes for XGBoost, the same MLP training loop).

That "exactly the same" claim is verified by parity tests to within numerical
noise, which is what makes a trial measured through the tool directly comparable
to a trial the agent ran by hand. The adapters are locked plumbing — the agent
never edits them.

## A.6 The BO tool (`run_bo.py`)

This is the heart of the C1 arm. The agent calls it like any shell command:

```
uv run python tools/run_bo.py --family xgboost --budget 10 --space '<json>'
```

The agent declares **one model family**, a **bounded box** of hyperparameters to
search, and a **sub-budget** of 5–15 trials. The tool then runs a *sealed* TPE
optimization episode inside that box — the agent is not in the loop — and prints
the best configuration it found, its `val_logloss`, the full per-trial trace, the
**pinned defaults** for any parameter it didn't declare, and a caveat that episode
scores use the harness's fixed preprocessing (re-measure on adoption). Key design
points:

- **The agent makes the macro decisions** (which family, which knobs, how long to
  commit); the BO does the micro-refinement it's good at. There is no automatic
  optimizer-led control — BO only ever runs inside a box the agent explicitly
  opened.
- **The episode's trials count against the same 50-trial budget** as the agent's
  own trials. How much budget the agent spends on the tool is itself a measurement.
- **Measurement is identical to a hand trial** — same data split, same metric,
  same preprocessing (via the adapters).
- **It is robust.** Each trial fits in a worker subprocess under the standard
  5-minute watchdog, so a hang or crash penalizes only that one trial and the
  episode keeps going.
- **The box is validated pre-flight (v1.1).** A bad family, out-of-range budget,
  unknown hyperparameter, malformed JSON, **or a declared bound/choice outside the
  adapter's legal range** is refused with a clear error and **zero trials**. The
  refusal prints the legal specs, and `run_bo.py --specs <family>` prints them on
  demand — so an invalid call teaches the agent the legal box for free. (This fixes
  the v1.0 pilot, where an out-of-spec box silently burned 5 of 10 trials.)
- **It's arm-gated** — in a C0 session the tool refuses to run.

What the agent does with the result is its own call: adopt the configuration
(edit `train.py`, commit, run a normal trial) or reject it and move on.

## A.7 Arms & playbook generation

`arms.py` is the registry: `ARMS = {C0: no capabilities, C1: [bo]}` and a
`CAPABILITIES` map. `start_session --arm C0|C1` generates that run's `program.md`
(the agent's instructions) by taking a shared `playbook/base.md` and splicing in
the enabled capability sections. For C0 nothing is spliced; for C1 the BO section
is added. A test asserts the two are identical except for that one section — this
is the "single difference" control, written as executable code. (Because
`program.md` is generated, it is gitignored; the committed source is
`playbook/`.)

The **BO section is prescriptive at family entry (v1.1)**: it mandates the
baseline-trial-then-one-episode routine whenever the agent enters a family it
hasn't tuned this session (including its assigned start), and is neutral about use
beyond that. So C1 measures *utility at cold-start entry* (where engagement is
guaranteed); voluntary use beyond the mandate is the residual uptake signal. The
**base** playbook (both arms, not part of the C0/C1 delta) also states the
family-integrity rule — each family means its canonical estimator class.

The starting model family for each run is chosen deterministically from
`(seed, task)` — **never** the arm — so C0 and C1 launched with the same seed and
task begin from the same family. That keeps the two arms paired at the starting
line.

## A.8 The reference arms (`run_reference.py`)

R1 (TPE) and R2 (random) are scripted searches with no LLM. A single Optuna study
searches the joint space — model family as a top-level choice, plus that family's
hyperparameters — for 50 trials, building and scoring through the same adapters
and metric as everything else. Each run writes its own self-contained ledger into
an output directory, and refuses to clobber a live agent session. A batch script
sweeps the 20 seeds per dataset that the protocol asks for. These give you the
"what would a classical optimizer get?" baseline on every figure.

## A.9 The ledger (what gets recorded)

Every trial appends one row to `logs/runs.jsonl`, tagged with its `source`:
`agent`, `bo` (a BO-episode trial), or `reference`. BO episodes also write a
single human-readable summary row to `results.tsv` (one per episode, not one per
internal trial). Failed trials record a fixed, task-specific **penalty** score
(twice the trivial "always predict the class frequencies" loss) so they're
clearly worse than any real model but don't corrupt averages. Keep/discard isn't
logged by the agent at all — it's recovered after the fact from git history
(kept commits stay on the branch; discarded ones are reset away but their trial
rows survive in the gitignored ledger). The commit message *is* the agent's
record of intent.

## A.10 The analysis pipeline

`tools/ingest_session.py` (and the `analysis.ipynb` notebook) turn a session's
ledger + git history into one table and a set of figures:

- **AUBC** — the mean best-so-far `val_logloss` over the 50 trials (the primary
  early-progress metric).
- **Best-so-far curve**, with BO-episode trials shaded so you can see where the
  agent spent budget on the tool.
- **Adoption detection** — flags a kept agent trial whose configuration matches a
  BO episode's best (same family, right after the episode): "the agent took the
  tool's suggestion."
- **Tool-engagement summary (C1)** — how many episodes, how much budget went to
  them, the adoption rate, and what share of the session's best-score
  improvements came from the tool versus the agent's own trials. This is the
  uptake-vs-utility picture H3 needs.

## A.11 Setup gates — all green

The protocol requires six "don't start the grid until these pass" gates. All
pass. The most important is the **family-ceiling check** (`tools/check_ceiling.py`):
it confirms the hard datasets are actually winnable by a non-GBDT model within
our four families — otherwise the whole premise (that the tool has room to help
where GBDTs are wrong) would be empty. Result:

```
task            xgboost   random_forest   logistic_regression   mlp     non-GBDT beats xgb?
credit-g        0.569     0.485 *         0.511                 0.504   YES  (random forest)
balance-scale   0.244     0.509           0.295                 0.129 * YES  (mlp)
cnae-9          0.274     0.494           0.182                 0.160 * YES  (mlp)
```

A non-GBDT wins on **3 of 3** hard tasks. The premise holds.

## A.12 Test suite & git state

- `uv run pytest` → **155 passed** (fast unit + integration; two slow "smoke"
  tests are off by default).
- `uv run pytest -m smoke` → 2 passed (a ceiling-check smoke and the full
  end-to-end mini-grid).
- `uv run python tools/smoke_test.py` exits 0 — proves C0 + a C1 BO episode + an
  R1 reference + ingest + notebook, all in one isolated run.
- The work is merged to `main` (local; not pushed). The frozen `baseline/` folder
  was never modified.

---

# Part B — How to run the experiments

This part assumes no prior familiarity. Do the steps in order the first time.

## B.0 One-time setup

You need: Windows, Python 3.10+, and [uv](https://docs.astral.sh/uv/). Open a
terminal **inside the `experiments/` folder** — every command below is run from
there.

```bash
cd "C:\Users\charl\Documents\njit\masters project\autoresearch\experiments"

uv sync                                          # install the environment (once)
uv run python tools/fetch_datasets.py --verify-only   # confirm all data checksums OK
uv run pytest -q                                 # sanity check: expect "155 passed"
```

If `--verify-only` reports OK for all six files and pytest is green, you're ready.
(You should **not** need `fetch_datasets.py` without `--verify-only` — the data is
already committed. You'd only re-fetch if a file went missing.)

### What you'll need for the LLM arms (C0, C1)

C0 and C1 require an actual coding agent to do the research — a fresh
**Claude Code** session (or equivalent) that can read files, run shell commands,
and make git commits, pointed at this `experiments/` folder. You are the
operator: you start each session, hand a one-line prompt to a *fresh* agent, let
it run to the trial budget, then wrap up. The reference arms (R1/R2) need no
agent — they're scripts.

## B.1 The mental model: the grid you're filling

Think of the study as a table you're filling in, cell by cell.

**LLM arms (you drive an agent — 42 sessions total):**

| Dataset | C0 seeds | C1 seeds |
|---|---|---|
| `electricity` (headline) | 0,1,2,3,4 | 0,1,2,3,4 |
| `credit-g` (headline) | 0,1,2,3,4 | 0,1,2,3,4 |
| `cnae-9` (headline) | 0,1,2,3,4 | 0,1,2,3,4 |
| `adult` | 0,1,2 | 0,1,2 |
| `balance-scale` | 0,1,2 | 0,1,2 |

That's 21 C0 + 21 C1 = **42 agent sessions**. Use the **same seed numbers** for
C0 and C1 — that's what pairs them (same seed + task ⇒ same starting family).

**Reference arms (scripts — 200 runs, but only 10 commands):**

20 seeds (0–19) × 5 datasets × 2 methods (tpe, random) = 200 reference runs,
launched as 10 batch commands.

Each LLM session runs 50 trials and takes a while (agent thinking + model fits;
the big `electricity`/`cnae-9` tasks are slower). The references are cheap. A
realistic plan: do the references first (cheap, unattended), then work through
the 42 agent sessions a few at a time. **Pick your LLM model now and keep it the
same for all 42 sessions** (the protocol forbids changing it mid-grid).

## B.2 A 10-minute confidence dry run (do this first)

Before the real grid, run the smoke test and one of each arm on the smallest
dataset (`balance-scale`) so you've seen every piece work.

```bash
# 1. The all-in-one smoke (C0 + a C1 BO episode + R1 + analysis), isolated:
uv run python tools/smoke_test.py          # expect "SMOKE TEST: PASS", exit 0

# 2. One reference run (≈ 15 seconds):
uv run python tools/run_reference.py --method tpe --task balance-scale --seed 0 \
    --trials 50 --out reference_runs/dryrun-tpe-balance-scale-seed0
```

If both succeed, the machinery is healthy. Now do a real C0 session (next
section) on `balance-scale` as a rehearsal before committing to the full grid.

## B.3 Running one LLM session (C0 or C1) — the full ritual

This is the loop you'll repeat 42 times. Example: **C1, credit-g, seed 0.**

**Step 1 — start the session.** From `experiments/`:

```bash
uv run python tools/start_session.py --arm C1 --task credit-g --seed 0 \
    --model-id claude-opus-4-8
```

- `--arm` is **required** (`C0` or `C1`).
- `--task` is the dataset; `--seed` sets the starting family (and the pairing).
- `--model-id` records *which LLM* will drive the run (use the same string every
  time). The harness git commit is recorded automatically.

This creates a git branch like `autoresearch/20260607-141500-xgboost`, writes
`logs/session.json`, generates `program.md` for the C1 arm, auto-archives any
previous run, and prints a block ending with:

```
Paste this to the agent:
------------------------------------------------------------
Read program.md and run a tabular autoresearch session. You are already on the
run branch. Your assigned starting family (in logs/session.json) is 'xgboost' on
task 'credit-g' - begin with a baseline of that family, then search freely. Run
each trial via run_trial.py and commit every trial with a clear message. Stop at
the trial budget.
------------------------------------------------------------
```

**Step 2 — hand it to a fresh agent.** Open a **new** Claude Code session in the
`experiments/` folder (fresh context — do not reuse a previous session's agent,
to avoid contamination between grid cells). Paste the printed prompt. The agent
will read `program.md`, write a baseline `train.py` for the assigned family, and
begin the trial loop on its own: edit → commit → `uv run python run_trial.py` →
keep or `git reset --hard`. In a **C1** session the playbook (v1.1) directs it to
run one BO episode whenever it enters a new family (after a baseline trial), and
it may call `tools/run_bo.py` freely beyond that; it can check a family's legal
search box with `run_bo.py --specs <family>`. You don't intervene — just let it
run. (The agent drives the tool; you only start and end the session.)

**Step 3 — let it run to budget.** The agent stops itself at 50 trials. You can
peek anytime: `results.tsv` and `logs/runs.jsonl` update per trial. (If the agent
stalls or stops early, you can nudge it: "continue until the trial budget is
reached.")

**Step 4 — wrap up.** When the agent has stopped, run:

```bash
uv run python tools/end_session.py
```

This ingests the run into a decision table, archives the run's `logs/` +
`results.tsv` into `archive/<run-id>/` (nothing is deleted), and returns you to
the `main` branch — ready for the next `start_session`. (If it says the working
tree isn't clean, the agent left an uncommitted trial; commit or `git reset` it,
then re-run `end_session`.)

**A C0 session is identical**, just with `--arm C0` (and the agent has no BO tool;
if it somehow tries `run_bo.py` it will be politely refused).

That's one cell. Repeat for every (arm, dataset, seed) in the table in B.1.

> **Tip:** the run branch (with its kept trial commits) is the permanent record
> of that search — it stays put. `end_session` only tidies the working-dir
> artifacts. To re-analyze an archived run later, see B.5.

## B.4 Running the reference arms (R1, R2)

No agent needed. For each dataset, sweep all 20 seeds with one command. R1 is
`--method tpe`, R2 is `--method random`.

```bash
# R1 (TPE) — one command per dataset:
uv run python tools/run_reference_batch.py --method tpe --task electricity   --seeds 0-19
uv run python tools/run_reference_batch.py --method tpe --task adult         --seeds 0-19
uv run python tools/run_reference_batch.py --method tpe --task credit-g      --seeds 0-19
uv run python tools/run_reference_batch.py --method tpe --task balance-scale --seeds 0-19
uv run python tools/run_reference_batch.py --method tpe --task cnae-9        --seeds 0-19

# R2 (random) — same five, with --method random:
uv run python tools/run_reference_batch.py --method random --task electricity   --seeds 0-19
uv run python tools/run_reference_batch.py --method random --task adult         --seeds 0-19
uv run python tools/run_reference_batch.py --method random --task credit-g      --seeds 0-19
uv run python tools/run_reference_batch.py --method random --task balance-scale --seeds 0-19
uv run python tools/run_reference_batch.py --method random --task cnae-9        --seeds 0-19
```

Each command writes 20 self-contained run folders under
`reference_runs/<method>-<task>-seed<N>/`. These are gitignored (results
artifacts, not code). `credit-g`/`balance-scale`/`cnae-9` are quick;
`electricity` (45k rows) and `cnae-9` (856 features) are slower but unattended —
start them and let them run. You can validate any run's ledger with
`uv run python tools/validate_jsonl.py reference_runs/<...>/logs/runs.jsonl`.

## B.5 After a session: analyze

`end_session` already builds the per-run decision table. To look at a run:

```bash
# Re-build the table for the *current* run (idempotent) and open the notebook:
uv run python tools/ingest_session.py
#   then open analysis.ipynb in Jupyter/VS Code (it reads LOGS_DIR, default "logs")
```

To analyze a **finished, archived** run without checking it out, point the tools
at its archived logs and its branch:

```bash
uv run python tools/ingest_session.py archive/<run-id>/logs \
    --head autoresearch/<run-id>-<model>
#   then open analysis.ipynb with the env var LOGS_DIR=archive/<run-id>/logs
#   (and HEAD_REF=autoresearch/<run-id>-<model>) so keep/discard is correct.
```

The notebook renders the progress curve, AUBC, the best-so-far curve, family/
locus distributions, and — for C1 runs — the BO tool-engagement summary.

The cross-arm comparison the *study* wants (C0 vs C1 vs R1 vs R2 per dataset, the
H1/H2/H3 analysis of protocol §7) is the final analysis step. The per-run tables
and reference ledgers produced above are its inputs; that aggregate analysis is
work still to be designed once real runs exist.

## B.6 The full grid checklist

Tick these off as you go.

**References (10 commands, ~unattended):** the 10 `run_reference_batch.py` lines
in B.4 (R1×5 datasets, R2×5 datasets).

**LLM sessions (42 — each is: `start_session` → fresh agent → `end_session`):**

```
C0  electricity     seeds 0 1 2 3 4
C0  credit-g        seeds 0 1 2 3 4
C0  cnae-9          seeds 0 1 2 3 4
C0  adult           seeds 0 1 2
C0  balance-scale   seeds 0 1 2
C1  electricity     seeds 0 1 2 3 4
C1  credit-g        seeds 0 1 2 3 4
C1  cnae-9          seeds 0 1 2 3 4
C1  adult           seeds 0 1 2
C1  balance-scale   seeds 0 1 2
```

For each line, run (substituting arm/task/seed):

```bash
uv run python tools/start_session.py --arm <C0|C1> --task <task> --seed <s> \
    --model-id <your-llm>
#   -> paste the printed prompt to a FRESH agent, let it run to 50 trials
uv run python tools/end_session.py
```

## B.7 Operator rules (don't break these)

- **One fixed LLM model for all 42 sessions.** Pass it as `--model-id` every time;
  don't switch mid-grid. (If you must switch, the protocol says restart the grid
  or report it as a limitation.)
- **A fresh agent per session.** Don't reuse one agent's context across cells.
- **Same seeds for C0 and C1** (the pairing). The reference seeds are 0–19.
- **Run everything from inside `experiments/`.**
- **Never edit** `prepare.py`, `logging_lib.py`, `family_adapters.py`,
  `train.py`-the-shipped-file, or anything in `../baseline/`. The agent edits its
  *own* working `train.py`; that's expected.
- **Don't change the harness mid-grid.** It's pinned by commit in `session.json`.

## B.8 Troubleshooting / FAQ

- **`fetch_datasets.py --verify-only` reports a mismatch / missing file.** A data
  file changed or is gone. Re-fetch: `uv run python tools/fetch_datasets.py
  --task <task>` (this is the one command allowed to use the network), then
  verify again.
- **`start_session` says a branch already exists / working tree dirty.** You're
  mid-run. Finish or `end_session` the current run first; commit or `git reset`
  any half-done trial.
- **The agent tries to hand-edit `results.tsv` or files under `logs/`.** It
  shouldn't — those are owned by the harness. That's a sign the playbook drifted;
  stop and check `program.md`.
- **A BO episode (C1) printed "REFUSED".** Expected when a constraint is violated
  (bad family, budget outside 5–15 or over the remaining budget, unknown
  hyperparameter, bad JSON). No trials were consumed; the agent can correct and
  retry.
- **`run_reference` says it refuses to write into the output dir.** That dir
  contains a live `session.json`. Point `--out` somewhere fresh under
  `reference_runs/`.
- **A reference run on `electricity`/`cnae-9` is slow.** Expected (large data).
  It's unattended; let it finish. Reduce `--trials` only for a quick smoke, never
  for a real grid cell (the budget is fixed at 50).
- **I want to re-look at an old run.** See B.5 (point ingest + the notebook at
  `archive/<run-id>/logs` with the run's branch as `--head`).
