import pandas as pd
import pytest

from tools.ingest_session import ingest, summarize
from tools.make_synthetic_session import build_synthetic_session


@pytest.mark.unit
def test_ingest_writes_csv_and_summary(tmp_path):
    logs = tmp_path / "logs"
    build_synthetic_session(logs)
    out = tmp_path / "analysis" / "decisions.csv"

    # repo_dir=None: synthetic commits aren't in git, so git-derived fields stay
    # empty, but the table still builds and writes.
    df = ingest(str(logs), repo_dir=None, out=str(out))
    assert out.exists()
    assert len(df) == 6

    reread = pd.read_csv(out)
    assert len(reread) == 6
    assert "kept_on_branch" in reread.columns

    text = summarize(df)
    assert "trials      : 6" in text
    assert "best" in text
