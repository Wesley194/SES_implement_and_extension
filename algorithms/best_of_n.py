"""Baseline inference-time scaling algorithms."""
import numpy as np
from .base import SearchResult


def best_of_n(
    objective, shape, budget=200, seed=0, verbose=False, **_kwargs
) -> SearchResult:
    """Best-of-N (random search) over i.i.d. Gaussian noises."""
    rng = np.random.default_rng(seed)
    best_s, best_x = -np.inf, None
    history = []

    for n_eval in range(1, budget + 1):
        x = rng.standard_normal(shape)
        s = objective(x)
        if s > best_s:
            best_s, best_x = s, x
        history.append((n_eval, best_s))

    if verbose:
        print(f"[BoN] NRE={budget:4d}  best={best_s:.4f}", flush=True)
    return SearchResult(best_x0=best_x, best_score=best_s, history=history)