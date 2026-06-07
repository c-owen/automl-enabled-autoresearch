import json

import pytest

import run_trial


@pytest.mark.unit
def test_resolve_task_env_wins(tmp_path, monkeypatch):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "session.json").write_text('{"task": "credit-g"}', encoding="utf-8")
    monkeypatch.setenv("AUTORESEARCH_TASK", "bank-marketing")
    # Explicit env overrides the session's recorded task.
    assert run_trial._resolve_task(str(logs)) == "bank-marketing"


@pytest.mark.unit
def test_resolve_task_from_session(tmp_path, monkeypatch):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "session.json").write_text('{"task": "credit-g"}', encoding="utf-8")
    monkeypatch.delenv("AUTORESEARCH_TASK", raising=False)
    assert run_trial._resolve_task(str(logs)) == "credit-g"


@pytest.mark.unit
def test_resolve_task_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("AUTORESEARCH_TASK", raising=False)
    assert run_trial._resolve_task(str(tmp_path / "logs")) is None


@pytest.mark.unit
def test_resolve_task_bad_session_json(tmp_path, monkeypatch):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "session.json").write_text("not json", encoding="utf-8")
    monkeypatch.delenv("AUTORESEARCH_TASK", raising=False)
    assert run_trial._resolve_task(str(logs)) is None
