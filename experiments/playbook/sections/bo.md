## Hyperparameter-search tool: bounded BO episodes

The harness provides a search tool:

`uv run python tools/run_bo.py --family <family> --budget <n> --space '<json>'`

You declare a model family (one of ALLOWED_FAMILIES), a bounded search space over
that family's hyperparameters, and a number of trials n (5–15). The tool runs n
trials of Bayesian optimization (TPE) inside the box you declared, against the
same train/val split and metric as your own trials, and prints the best
configuration found, its val_logloss, the full trial trace, and the default
values it used for any parameters you did not declare. These n trials count
against the session's TRIAL_BUDGET like any other trial, and may not exceed the
trials you have remaining.

Run `uv run python tools/run_bo.py --specs <family>` to see the legal parameters,
types, and ranges for a family before declaring a space. Calls outside these
specs are refused without consuming trials.

**Rule — entering a family:** when you start working in a model family you have
not yet tuned this session (including your assigned starting family), first run
one baseline trial of that family, then one BO episode on it (budget 5–10) before
further hand-tuning. This rule governs *how* you enter a family, not *whether* or
*when* to switch families — switching remains entirely your decision. Outside
family entry, use of the tool is optional.

The tool only searches inside the box you give it. It never edits train.py and
never changes families. Episode scores are measured with the harness's standard
preprocessing; what you do with the result is up to you — to adopt a
configuration, edit train.py accordingly, commit, and run a trial as usual.

Space format: `{"<param>": {"type": "int|float|categorical", "low": .., "high": .., "log": true|false, "choices": [..]}}`
