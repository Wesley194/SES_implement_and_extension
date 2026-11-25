"""CMA-ES based Spectral Evolution Search algorithms."""
import cma
import numpy as np
from spaces import LowFreqSpace
from .base import SearchResult


def ses_cma(
    objective,
    shape,
    budget=200,
    level=4,
    pop=10,
    wavelet="haar",
    seed=0,
    diagonal=False,
    verbose=False,
    **_kwargs,
) -> SearchResult:
    """CMA-ES (full or diagonal covariance) in the low-frequency subspace."""
    rng = np.random.default_rng(seed)
    space = LowFreqSpace(shape, level, wavelet, rng)
    es = cma.CMAEvolutionStrategy(
        np.zeros(space.dim),
        1.0,
        {"popsize": pop, "seed": seed + 1, "verbose": -9, "CMA_diagonal": bool(diagonal)},
    )

    best_s, best_u, n_eval = -np.inf, None, 0
    history = []
    tag = "SES-CMA-diag" if diagonal else "SES-CMA"

    while n_eval < budget:
        cands = es.ask()[: budget - n_eval]
        scores = [objective(space.to_noise(u)) for u in cands]
        n_eval += len(cands)
        for u, s in zip(cands, scores):
            if s > best_s:
                best_s, best_u = s, u
        if len(cands) == es.popsize:
            es.tell(cands, [-s for s in scores])  # CMA-ES minimises
        history.append((n_eval, best_s))
        if verbose:
            print(f"[{tag}] NRE={n_eval:4d}  best={best_s:.4f}", flush=True)

    return SearchResult(
        best_x0=space.to_noise(best_u), best_score=best_s, history=history
    )