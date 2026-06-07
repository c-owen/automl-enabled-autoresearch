"""Arms-as-configuration: the enable/disable mechanism for agent capabilities.

An *arm* is a named configuration declaring which capabilities are enabled. A
*capability* is a self-contained module bundling a CLI the agent can call and the
playbook text describing it. The single difference between C0 (LLM-only) and C1
(LLM+BO) is the `bo` capability — which adds exactly one playbook section and the
`run_bo.py` CLI. Adding a future tool = one CAPABILITIES entry + one ARMS line +
one section file; no copies of the harness.

``generate_playbook(arm)`` renders ``program.md`` for an arm: the shared
``playbook/base.md`` with its ``{{CAPABILITY_SECTIONS}}`` anchor replaced by the
enabled capabilities' section text (empty for C0). The guarantee that C0 and C1
differ by exactly the bo section is protocol gate 4, enforced as a test.
"""

import os
from dataclasses import dataclass, field

_ROOT = os.path.dirname(os.path.abspath(__file__))
_ANCHOR = "{{CAPABILITY_SECTIONS}}"


@dataclass(frozen=True)
class Capability:
    """A self-contained agent capability: a CLI + the playbook section for it."""
    cli: str               # path (relative to experiments/) of the tool CLI
    playbook_section: str   # path (relative to experiments/) of the section file


@dataclass
class Arm:
    """A named arm: the set of capabilities it enables."""
    capabilities: list = field(default_factory=list)


# The capability registry. One entry per tool; future tools add one line each.
CAPABILITIES = {
    "bo": Capability(
        cli="tools/run_bo.py",
        playbook_section="playbook/sections/bo.md",
    ),
}

# The arm registry. C0 is the control (no capabilities); C1 adds bo. Future arms
# (sham, stagnation, ...) are one line each.
ARMS = {
    "C0": Arm(capabilities=[]),
    "C1": Arm(capabilities=["bo"]),
}


def _read(rel_path: str) -> str:
    with open(os.path.join(_ROOT, rel_path), "r", encoding="utf-8") as fh:
        return fh.read()


def capabilities_for(arm: str) -> list:
    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}; known arms: {sorted(ARMS)}")
    return list(ARMS[arm].capabilities)


def generate_playbook(arm: str) -> str:
    """Render program.md for ``arm`` = base.md with enabled sections spliced in.

    The capability block is the enabled sections' text joined by blank lines
    (empty string for an arm with no capabilities). The output is normalized to a
    single trailing newline so C0 and C1 differ by exactly the inserted section.
    """
    caps = capabilities_for(arm)
    sections = [_read(CAPABILITIES[cap].playbook_section).strip() for cap in caps]
    block = "\n\n".join(sections)
    rendered = _read("playbook/base.md").replace(_ANCHOR, block)
    return rendered.rstrip("\n") + "\n"
