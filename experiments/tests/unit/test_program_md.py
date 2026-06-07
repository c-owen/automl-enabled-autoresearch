"""Cheap insurance that the C0 playbook keeps its key invariants.

These are presence checks, not prose review — they guard against silent removal
of the playbook's load-bearing facts. program.md is now GENERATED per session
(arms.generate_playbook); the C0 render is the LLM-only playbook an agent reads,
so it is what we check here.
"""

import pytest

from arms import generate_playbook

PROGRAM_MD = generate_playbook("C0")


@pytest.mark.unit
def test_program_md_lists_all_families():
    for family in ("xgboost", "random_forest", "logistic_regression", "mlp"):
        assert family in PROGRAM_MD, f"program.md missing family {family!r}"
    assert "ALLOWED_FAMILIES" in PROGRAM_MD


@pytest.mark.unit
def test_program_md_mentions_selection_metric():
    assert "val_logloss" in PROGRAM_MD


@pytest.mark.unit
def test_program_md_mentions_trial_budget():
    assert "TRIAL_BUDGET" in PROGRAM_MD
    assert "loop forever" not in PROGRAM_MD.lower()


@pytest.mark.unit
def test_program_md_references_print_contract():
    for key in ("val_logloss:", "model_family:", "task_name:", "END_OF_TRIAL"):
        assert key in PROGRAM_MD, f"program.md missing print-contract token {key!r}"


@pytest.mark.unit
def test_program_md_workflow_uses_wrapper():
    assert "uv run python run_trial.py" in PROGRAM_MD


@pytest.mark.unit
def test_program_md_intent_is_commit_message():
    # Karpathy-faithful: intent is carried by the commit message, not a
    # separate per-trial decision/plan/reflection file.
    assert "commit message" in PROGRAM_MD.lower()
    assert "decisions.jsonl" not in PROGRAM_MD
    assert "PRE_TRIAL_PLAN_PATH" not in PROGRAM_MD


@pytest.mark.unit
def test_program_md_assigned_model_and_session():
    # Each run starts from a session-assigned family (first-mover-bias control).
    assert "session.json" in PROGRAM_MD
    assert "assigned" in PROGRAM_MD.lower()


@pytest.mark.unit
def test_program_md_keep_discard_via_git():
    assert "git reset --hard" in PROGRAM_MD


@pytest.mark.unit
def test_program_md_no_examples_directory_reference():
    assert "examples/" not in PROGRAM_MD
