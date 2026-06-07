"""Pre-registered primary metrics (protocol §6.1), importable by the notebook
and tests so AUBC is computed one way everywhere."""

import math


def _is_nan(v) -> bool:
    return isinstance(v, float) and math.isnan(v)


def best_so_far(values) -> list:
    """Running best (minimum) val_logloss, monotone non-increasing.

    A failed trial (None / NaN) does not improve the best — it simply carries the
    previous best forward (protocol §6.4). Until the first valid value the best is
    ``+inf``.
    """
    best = float("inf")
    out = []
    for v in values:
        if v is not None and not _is_nan(v) and v < best:
            best = v
        out.append(best)
    return out


def aubc(values) -> float:
    """Area under the best-so-far curve = mean of best-so-far over trials 1..N
    (lower is better; rewards finding good solutions early — protocol §6.1)."""
    series = best_so_far(values)
    if not series:
        return float("nan")
    return sum(series) / len(series)
