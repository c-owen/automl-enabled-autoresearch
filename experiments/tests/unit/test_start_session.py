import json
from datetime import datetime

import pytest

from prepare import ALLOWED_FAMILIES
from tools.start_session import (
    archive_previous_session,
    build_branch_name,
    choose_initial_model,
    start_session,
)


@pytest.mark.unit
def test_choose_initial_model_deterministic():
    assert choose_initial_model(7) == choose_initial_model(7)


@pytest.mark.unit
def test_choose_initial_model_in_allowed():
    for seed in range(50):
        assert choose_initial_model(seed) in ALLOWED_FAMILIES


@pytest.mark.unit
def test_choose_initial_model_covers_families():
    # Across many seeds the pick is not constant (the whole point of the control).
    picks = {choose_initial_model(s) for s in range(200)}
    assert len(picks) >= 3


@pytest.mark.unit
def test_build_branch_name_format():
    when = datetime(2026, 6, 1, 14, 30, 22)
    assert build_branch_name(when, "mlp") == "autoresearch/20260601-143022-mlp"


@pytest.mark.unit
def test_start_session_writes_session_json(tmp_path):
    when = datetime(2026, 6, 1, 14, 30, 22)
    meta = start_session(
        logs_dir=str(tmp_path), task="adult", seed=7, when=when,
        locked=False, create_branch=False, archive=False,
    )
    expected_model = choose_initial_model(7)
    assert meta["initial_model"] == expected_model
    assert meta["branch"] == f"autoresearch/20260601-143022-{expected_model}"
    assert meta["seed"] == 7
    assert meta["family_locked"] is False

    written = json.loads((tmp_path / "session.json").read_text(encoding="utf-8"))
    assert written["initial_model"] == expected_model
    assert written["task"] == "adult"
    assert written["trial_budget"] == meta["trial_budget"]


@pytest.mark.unit
def test_start_session_lock_flag(tmp_path):
    meta = start_session(
        logs_dir=str(tmp_path), seed=1, when=datetime(2026, 6, 1, 9, 0, 0),
        locked=True, create_branch=False, archive=False,
    )
    assert meta["family_locked"] is True


@pytest.mark.unit
def test_start_session_explicit_model(tmp_path):
    meta = start_session(
        logs_dir=str(tmp_path), seed=7, when=datetime(2026, 6, 1, 9, 0, 0),
        model="mlp", create_branch=False, archive=False,
    )
    assert meta["initial_model"] == "mlp"
    assert meta["model_source"] == "explicit"
    assert meta["branch"].endswith("-mlp")


@pytest.mark.unit
def test_start_session_random_model_source(tmp_path):
    meta = start_session(
        logs_dir=str(tmp_path), seed=7, when=datetime(2026, 6, 1, 9, 0, 0),
        create_branch=False, archive=False,
    )
    assert meta["initial_model"] == choose_initial_model(7)
    assert meta["model_source"] == "random"


@pytest.mark.unit
def test_start_session_rejects_unknown_model(tmp_path):
    with pytest.raises(ValueError, match="ALLOWED_FAMILIES"):
        start_session(
            logs_dir=str(tmp_path), model="lightgbm",
            create_branch=False, archive=False,
        )


@pytest.mark.unit
def test_archive_previous_session_moves_ledger(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "runs.jsonl").write_text('{"trial_id": 1}\n', encoding="utf-8")
    (logs / "session.json").write_text('{"run_id": "oldrun"}', encoding="utf-8")
    results = tmp_path / "results.tsv"
    results.write_text("commit\tstatus\n", encoding="utf-8")
    figures = tmp_path / "figures"  # co-located with logs (dirname == tmp_path)
    figures.mkdir()
    (figures / "progress.png").write_bytes(b"x")
    archive_root = tmp_path / "archive"

    dest = archive_previous_session(
        logs_dir=str(logs), results_tsv=str(results),
        archive_root=str(archive_root),
    )

    # Archived under the previous session's run_id; originals moved away.
    assert dest == str(archive_root / "oldrun")
    assert (archive_root / "oldrun" / "logs" / "runs.jsonl").exists()
    assert (archive_root / "oldrun" / "results.tsv").exists()
    assert (archive_root / "oldrun" / "figures" / "progress.png").exists()
    assert not logs.exists()
    assert not results.exists()
    assert not figures.exists()


@pytest.mark.unit
def test_archive_previous_session_noop_when_empty(tmp_path):
    # Nothing to archive -> returns None, creates nothing.
    dest = archive_previous_session(
        logs_dir=str(tmp_path / "logs"),
        results_tsv=str(tmp_path / "results.tsv"),
        archive_root=str(tmp_path / "archive"),
    )
    assert dest is None
    assert not (tmp_path / "archive").exists()
