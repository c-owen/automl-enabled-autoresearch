## Optional tool: bounded hyperparameter-search episodes

The harness provides a search tool you may call at any point:

`uv run python tools/run_bo.py --family <family> --budget <n> --space '<json>'`

You declare a model family (one of ALLOWED_FAMILIES), a bounded search space over
that family's hyperparameters, and a number of trials n (5–15). The tool runs n
trials of Bayesian optimization (TPE) inside the box you declared, against the
same train/val split and metric as your own trials, and prints the best
configuration found, its val_logloss, and the full trial trace. These n trials
count against the session's TRIAL_BUDGET like any other trial, and may not exceed
the trials you have remaining.

The tool only searches inside the box you give it. It never edits train.py and
never changes families. What you do with the result is up to you: to adopt a
configuration, edit train.py accordingly, commit, and run a trial as usual.

Space format: `{"<param>": {"type": "int|float|categorical", "low": .., "high": .., "log": true|false, "choices": [..]}}`
