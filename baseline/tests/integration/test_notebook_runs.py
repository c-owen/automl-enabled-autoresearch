"""Execute analysis.ipynb end-to-end against a synthetic session fixture.

Passes iff every cell runs without raising. This guards the notebook (the
visualization deliverable) against bit-rot as the log schema evolves.
"""

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

from tools.make_synthetic_session import build_synthetic_session

REPO = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO / "analysis.ipynb"


@pytest.mark.integration
def test_notebook_runs(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"
    build_synthetic_session(logs_dir)

    monkeypatch.setenv("LOGS_DIR", str(logs_dir))
    nb = nbformat.read(str(NOTEBOOK), as_version=4)
    client = NotebookClient(
        nb,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO)}},
    )
    # Raises CellExecutionError if any cell errors.
    client.execute()
