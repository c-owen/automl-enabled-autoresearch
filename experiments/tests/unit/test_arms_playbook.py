"""Step 8: arms-as-configuration + playbook generation (protocol gate 4).

The load-bearing guarantee: C0 and C1 playbooks differ by EXACTLY the bo section.
"""

import difflib
import json
from datetime import datetime

import pytest

import arms
from tools.start_session import choose_initial_model, start_session


def _read(rel):
    import os
    return open(os.path.join(arms._ROOT, rel), encoding="utf-8").read()


@pytest.mark.unit
def test_playbook_diff_is_exactly_bo_section():
    """Gate 4 as executable code: C1 == C0 + the bo section, nothing else."""
    c0 = arms.generate_playbook("C0")
    c1 = arms.generate_playbook("C1")
    bo = _read("playbook/sections/bo.md").strip()

    # The only difference is the bo section appended after a blank-line separator.
    assert c1 == c0.rstrip("\n") + "\n\n" + bo + "\n"

    # And, line-wise, C0 -> C1 only ADDS lines (nothing removed or changed).
    removed = [ln[2:] for ln in difflib.ndiff(c0.splitlines(), c1.splitlines())
               if ln.startswith("- ")]
    assert removed == []
    assert bo in c1 and bo not in c0


@pytest.mark.unit
def test_arms_registry_shape():
    assert set(arms.ARMS) >= {"C0", "C1"}
    assert arms.capabilities_for("C0") == []
    assert arms.capabilities_for("C1") == ["bo"]
    assert arms.CAPABILITIES["bo"].cli == "tools/run_bo.py"


@pytest.mark.unit
def test_arm_stamped_in_session(tmp_path):
    when = datetime(2026, 6, 1, 12, 0, 0)
    meta = start_session(
        logs_dir=str(tmp_path), task="credit-g", seed=4, when=when, arm="C1",
        create_branch=False, archive=False,
        program_md_path=str(tmp_path / "program.md"),
    )
    assert meta["arm"] == "C1"
    assert meta["capabilities"] == ["bo"]
    written = json.loads((tmp_path / "session.json").read_text(encoding="utf-8"))
    assert written["arm"] == "C1" and written["capabilities"] == ["bo"]

    # The generated playbook is the C1 render.
    generated = (tmp_path / "program.md").read_text(encoding="utf-8")
    assert generated == arms.generate_playbook("C1")


@pytest.mark.unit
def test_c0_arm_stamps_empty_capabilities(tmp_path):
    meta = start_session(
        logs_dir=str(tmp_path), task="credit-g", seed=4,
        when=datetime(2026, 6, 1, 12, 0, 0), arm="C0",
        create_branch=False, archive=False,
        program_md_path=str(tmp_path / "program.md"),
    )
    assert meta["arm"] == "C0" and meta["capabilities"] == []
    assert (tmp_path / "program.md").read_text(encoding="utf-8") == \
        arms.generate_playbook("C0")


@pytest.mark.unit
@pytest.mark.parametrize("task", ["credit-g", "balance-scale", "cnae-9"])
@pytest.mark.parametrize("seed", [0, 7, 42])
def test_start_family_arm_independent(tmp_path, task, seed, monkeypatch):
    """Same (seed, task) -> same starting family for C0 and C1 (quasi-pairing)."""
    when = datetime(2026, 6, 1, 12, 0, 0)
    c0 = start_session(
        logs_dir=str(tmp_path / "c0"), task=task, seed=seed, when=when, arm="C0",
        create_branch=False, archive=False,
        program_md_path=str(tmp_path / "c0.md"),
    )
    c1 = start_session(
        logs_dir=str(tmp_path / "c1"), task=task, seed=seed, when=when, arm="C1",
        create_branch=False, archive=False,
        program_md_path=str(tmp_path / "c1.md"),
    )
    assert c0["initial_model"] == c1["initial_model"]
    assert c0["initial_model"] == choose_initial_model(seed, task)


@pytest.mark.unit
def test_start_session_rejects_unknown_arm(tmp_path):
    with pytest.raises(ValueError, match="arm"):
        start_session(
            logs_dir=str(tmp_path), arm="C9", create_branch=False, archive=False,
            program_md_path=str(tmp_path / "program.md"),
        )
