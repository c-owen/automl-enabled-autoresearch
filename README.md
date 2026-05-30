# Autoresearch for Tabular ML

Summer 2026 master's project. Extends Karpathy's Autoresearch loop into supervised tabular ML and studies whether AutoML-derived tools (stagnation detection, exploration/exploitation policies) improve an LLM agent's experimentation decisions.

Unlike hybrid approaches that subordinate the LLM to a classical optimizer, the LLM here remains the sole decision-maker; AutoML mechanisms are exposed as advisory tools the agent may choose to consult.

## Status

Milestone 1: porting Autoresearch to a tabular tasks and running an LLM-only baseline.

## Roadmap

1. Tabular Autoresearch port and LLM-only baseline.
2. Characterization study of baseline behavior.
3. First AutoML-derived tool, informed by #2.
4. Deirection-selection policy (restart/bandit over model families).
5. Extended analysis.

## Proposal

Docs/proposal.pdf for the full project proposal
