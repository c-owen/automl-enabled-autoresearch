# Autoresearch for Tabular ML

Summer 2026 master's project. Extends Karpathy's Autoresearch loop into supervised tabular ML and studies whether AutoML-derived tools (stagnation detection, exploration/exploitation policies) improve an LLM agent's experimentation decisions.

Unlike hybrid approaches that subordinate the LLM to a classical optimizer, the LLM here remains the sole decision-maker; AutoML mechanisms are exposed as advisory tools the agent may choose to consult.

## Status

Milestone 1: porting Autoresearch to tabular tasks and running an LLM-only baseline.

## Repository layout

- **`baseline/`** — the LLM-only baseline: a port of Karpathy's
  [`autoresearch`](https://github.com/karpathy/autoresearch) loop to supervised
  tabular classification. This is the *reference point*, not the contribution;
  see [`baseline/README.md`](baseline/README.md) and
  [`baseline/ATTRIBUTION.md`](baseline/ATTRIBUTION.md). Run sessions from inside
  this folder.
- *(later)* the AutoML / agentic decision layer — the actual research
  contribution — will live in a sibling top-level folder, compared against the
  baseline at matched trial budgets.

## Roadmap

1. Tabular Autoresearch port and LLM-only baseline.
2. Characterization study of baseline behavior.
3. First AutoML-derived tool, informed by #2.
4. Direction-selection policy (restart/bandit over model families).
5. Extended analysis.

## Proposal

Docs/proposal.pdf for the full project proposal
