"""A5 (v1.1): train.py asserts the fitted estimator is the family's canonical
class — substitutes (ExtraTrees as random_forest, ...) exit non-zero."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# A minimal, self-contained train.py whose estimator class is parameterized, run
# on the fast all-numeric balance-scale task. The end-of-script integrity check is
# the SHIPPED check verbatim, so a compliant class passes and a substitute fails.
_MINIMAL = '''import sys
from prepare import load_task, evaluate, TASK_NAME
from logging_lib import family_violation, peak_rss_mb
from sklearn.ensemble import {cls}
MODEL = "random_forest"
Xtr, ytr, Xv, yv = load_task()
model = {cls}(n_estimators=20, random_state=0)
model.fit(Xtr, ytr)
m = evaluate(model, Xv, yv)
estimator_class = type(model).__name__
if family_violation(MODEL, estimator_class):
    sys.stderr.write(f"ERROR: family violation: {{estimator_class}} for {{MODEL}}\\n")
    sys.exit(2)
print("---")
print(f"val_logloss:    {{m['val_logloss']:.6f}}")
print(f"val_acc:        {{m['val_acc']:.6f}}")
print(f"val_auc:        {{m['val_auc']:.6f}}")
print("train_seconds:  0.1")
print("total_seconds:  0.1")
print(f"peak_mem_mb:    {{peak_rss_mb():.1f}}")
print(f"model_family:   {{MODEL}}")
print("n_params:       20")
print(f"task_name:      {{TASK_NAME}}")
print(f"estimator_class: {{estimator_class}}")
print("END_OF_TRIAL")
'''


def _run(script_path):
    env = {**os.environ, "AUTORESEARCH_TASK": "balance-scale",
           "PYTHONPATH": str(REPO)}
    return subprocess.run([sys.executable, str(script_path)], cwd=str(REPO),
                          env=env, capture_output=True, text=True, timeout=300)


@pytest.mark.integration
def test_shipped_train_passes_integrity():
    r = _run(REPO / "train.py")
    assert r.returncode == 0, r.stderr
    assert "estimator_class: XGBClassifier" in r.stdout


@pytest.mark.integration
def test_canonical_random_forest_passes(tmp_path):
    f = tmp_path / "train_rf.py"
    f.write_text(_MINIMAL.format(cls="RandomForestClassifier"), encoding="utf-8")
    r = _run(f)
    assert r.returncode == 0, r.stderr
    assert "END_OF_TRIAL" in r.stdout
    assert "estimator_class: RandomForestClassifier" in r.stdout


@pytest.mark.integration
def test_extratrees_as_random_forest_fails(tmp_path):
    f = tmp_path / "train_et.py"
    f.write_text(_MINIMAL.format(cls="ExtraTreesClassifier"), encoding="utf-8")
    r = _run(f)
    assert r.returncode != 0
    assert "family violation" in r.stderr
    assert "END_OF_TRIAL" not in r.stdout
