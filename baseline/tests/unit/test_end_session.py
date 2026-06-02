import pytest

from tools.end_session import end_session
from tools.make_synthetic_session import build_synthetic_session


@pytest.mark.unit
def test_end_session_ingests_and_archives(tmp_path):
    logs = tmp_path / "logs"
    results = tmp_path / "results.tsv"
    build_synthetic_session(logs, results_tsv=str(results))
    (logs / "session.json").write_text('{"run_id": "testrun"}', encoding="utf-8")

    result = end_session(
        logs_dir=str(logs), results_tsv=str(results),
        archive_root=str(tmp_path / "archive"), repo_dir=".", to_master=False,
    )

    assert result["n_trials"] == 6
    assert result["switched_to_master"] is False
    assert result["archived"] == str(tmp_path / "archive" / "testrun")

    # The run's ledger (incl. the generated decisions.csv) is archived; originals moved.
    arch = tmp_path / "archive" / "testrun"
    assert (arch / "logs" / "runs.jsonl").exists()
    assert (arch / "logs" / "decisions.csv").exists()
    assert (arch / "results.tsv").exists()
    assert not logs.exists()
    assert not results.exists()


@pytest.mark.unit
def test_end_session_no_logs_is_safe(tmp_path):
    # Nothing to wrap up -> no crash, nothing archived.
    result = end_session(
        logs_dir=str(tmp_path / "logs"), results_tsv=str(tmp_path / "results.tsv"),
        archive_root=str(tmp_path / "archive"), repo_dir=".", to_master=False,
    )
    assert result["n_trials"] == 0
    assert result["archived"] is None
