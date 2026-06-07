# autoresearch — tabular classification

This is an experiment to have an LLM run its own research: search for the best
supervised tabular **classification** model, autonomously, by hill-climbing a
git branch. You are the researcher. `train.py` is your workpiece. `prepare.py`
and `logging_lib.py` are the locked harness — you do not edit them.

## Setup

A human starts each session by running `tools/start_session.py`, which creates
this run's branch and writes `logs/session.json`. You begin **already on that
branch** — do not create a new one.

1. Read the in-scope files: `README.md`, `prepare.py` (locked harness, do not
   modify), `train.py` (the file you edit).
2. Read `logs/session.json`. It names this run's **assigned starting model
   family** and the task. Your **first trial must use the assigned family**.
3. Verify the harness works:
   `uv run python -c "from prepare import load_task; print([a.shape for a in load_task()])"`

## The task

Supervised tabular classification. **The goal is the lowest `val_logloss`** on
the pinned validation split. `val_acc` and `val_auc` ride along in the summary
but are *not* the selection metric.

Data is supplied clean by `prepare.py` via `load_task()` →
`(X_train, y_train, X_val, y_val)`. You do **not** touch the raw dataset or the
train/val split. Preprocessing *inside* `train.py` (impute, scale, encode) is
fair game.

## Model families

The family your `train.py` uses **must** be one of:

```
ALLOWED_FAMILIES = ["xgboost", "random_forest", "logistic_regression", "mlp"]
```

`train.py` sets `MODEL = "<family>"` and asserts it is allowed (it exits
non-zero otherwise). **Your assigned starting family is in `logs/session.json`.**
If it differs from the shipped `train.py`, rewrite `train.py` into a baseline of
the assigned family — written from your own knowledge — *before* your first
trial. There is no reference/`examples` directory to copy from. Switching
families later is a normal move in the search.

## What is fair game

- **Hyperparameters** — all of them.
- **Preprocessing** — imputation, scaling, encoding, feature handling.
- **Model family** — within `ALLOWED_FAMILIES`.
- **Validation strategy inside `train.py`** — internal CV / early-stopping
  holdout is fine. The authoritative score is always the printed `val_logloss`
  from `prepare.evaluate(model, X_val, y_val)`.

## What you cannot do

- Modify `prepare.py` or `logging_lib.py` (the locked harness).
- Touch the raw dataset or the pinned train/val split.
- Add dependencies beyond `pyproject.toml`.
- Change `prepare.evaluate` — it is the ground-truth metric.

## Output format (print contract)

`train.py` must end by printing this exact `---` block, then `END_OF_TRIAL`:

```
---
val_logloss:    0.278900
val_acc:        0.872300
val_auc:        0.928600
train_seconds:  0.4
total_seconds:  0.5
peak_mem_mb:    249.1
model_family:   xgboost
n_params:       1800
task_name:      adult
END_OF_TRIAL
```

The wrapper parses this block to record the trial — keep it intact.

## Budget

- **Trial budget**: a session runs for `TRIAL_BUDGET` trials total (see
  `prepare.py`). When the trial budget is exhausted, **stop** — a finite,
  principled stop condition, not an indefinite run.
- **Per-trial cap**: 5 minutes wall-clock (`TIME_BUDGET`), enforced inside
  `train.py` by a watchdog. A trial that exceeds it prints `TIMEOUT` and exits
  124, recorded as `status=crash`.

## The trial loop

Run each trial through the wrapper — **not** `train.py` directly — because the
wrapper records the trial to the ledger:

1. Edit `train.py` with one experimental idea (tune HPs, change preprocessing,
   swap family, alter the net).
2. **`git commit`** the change with a clear message saying *what you changed and
   why*. **The commit message is your trial's intent record** — it is how your
   reasoning is captured, so write it well. No separate plan/reflection/decision
   files are needed.
3. **`uv run python run_trial.py`** — runs `train.py`, captures stdout to
   `logs/runs/<commit>.log`, and appends one row to `logs/runs.jsonl` and
   `results.tsv`.
4. Read `val_logloss` from the printed summary.
5. **Keep or discard**: if `val_logloss` improved on the parent commit, keep the
   commit (advance the branch). Otherwise `git reset --hard HEAD~1` to discard
   and return to the parent.

Do **not** hand-edit `results.tsv` or anything under `logs/` — the wrapper owns
those writes, and they are gitignored (don't commit them). The keep/discard
outcome is recovered post-hoc from git history; you don't log it explicitly.

## Crashes

If a trial crashes (a bug, an OOM, a bad config): if it's trivially fixable (a
typo, a missing import), fix it and re-run. If the idea itself is broken, move
on — the wrapper records crashes (including timeouts) automatically.

{{CAPABILITY_SECTIONS}}
