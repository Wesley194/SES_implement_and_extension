"""Cross-Entropy Method (CEM) based Spectral Evolution Search algorithms."""
import numpy as np
from spaces import LowFreqSpace, WaveletSpace
from .base import SearchResult


def ses_cem(
    objective,
    shape,
    budget=200,
    pop=10,
    elite=5,
    gamma=1e-5,
    level=4,
    wavelet="haar",
    seed=0,
    verbose=False,
    **_kwargs,
) -> SearchResult:
    """Standard SES (Algorithm 1): Diagonal CEM in J-level low-frequency subspace."""
    rng = np.random.default_rng(seed)
    space = LowFreqSpace(shape, level, wavelet, rng)
    mu = np.zeros(space.dim)
    var = np.ones(space.dim)
    pool, n_eval, history = [], 0, []

    while n_eval < budget:
        n = min(pop, budget - n_eval)
        us = mu + np.sqrt(var) * rng.standard_normal((n, space.dim))
        for u in us:
            s = objective(space.to_noise(u))
            pool.append((u, s))
        n_eval += n

        pool.sort(key=lambda t: t[1], reverse=True)
        pool = pool[:elite]
        U = np.stack([u for u, _ in pool])
        mu = (1 - gamma) * U.mean(0) + gamma * mu
        var = (1 - gamma) * U.var(0) + gamma * var
        history.append((n_eval, pool[0][1]))
        if verbose:
            print(
                f"[SES-CEM] NRE={n_eval:4d}  best={pool[0][1]:.4f}  "
                f"mean sigma={np.sqrt(var).mean():.3f}",
                flush=True,
            )

    u_star = mu + np.sqrt(var) * rng.standard_normal(space.dim)
    best_u, best_s = pool[0]
    return SearchResult(
        best_x0=space.to_noise(best_u),
        best_score=best_s,
        history=history,
        final_x0=space.to_noise(u_star),
    )


def ses_progressive(
    objective,
    shape,
    budget=200,
    levels=None,
    level=3,
    pop=10,
    elite=5,
    gamma=1e-5,
    sigma_new=1.0,
    wavelet="haar",
    seed=0,
    verbose=False,
    **_kwargs,
) -> SearchResult:
    """Coarse-to-fine progressive SES unfreezing finer detail bands per stage."""
    rng = np.random.default_rng(seed)

    _, h_dim, w_dim = shape
    max_possible = int(np.log2(min(h_dim, w_dim)))
    if levels is None:
        start_lvl = min(level + 2, max_possible)
        levels = tuple(range(start_lvl, level - 1, -1))
    if not levels:
        levels = (level,)

    space = WaveletSpace(shape, max_level=levels[0], wavelet=wavelet, rng=rng)
    n_stage = len(levels)
    mu = var = pool = None
    n_eval, history, prev_k = 0, [], None

    while n_eval < budget:
        stage = min(int(n_eval / budget * n_stage), n_stage - 1)
        k = space.level_to_k(levels[stage])
        if k != prev_k:
            d_new = space.dim(k)
            ref = space.ref_vec(k)
            if mu is None:
                mu, var, pool = np.zeros(d_new), np.ones(d_new), []
            else:
                d_old = mu.size
                mu = np.concatenate([mu, ref[d_old:]])
                var = np.concatenate([var, np.full(d_new - d_old, sigma_new**2)])
                pool = [(np.concatenate([u, ref[d_old:]]), s) for u, s in pool]
            prev_k = k

        stage_end = int(budget * (stage + 1) / n_stage) if stage < n_stage - 1 else budget
        n = min(pop, stage_end - n_eval) or min(pop, budget - n_eval)
        us = mu + np.sqrt(var) * rng.standard_normal((n, mu.size))
        for u in us:
            pool.append((u, objective(space.to_noise(u, k))))
        n_eval += n

        pool.sort(key=lambda t: t[1], reverse=True)
        pool = pool[:elite]
        U = np.stack([u for u, _ in pool])
        mu = (1 - gamma) * U.mean(0) + gamma * mu
        var = (1 - gamma) * U.var(0) + gamma * var
        history.append((n_eval, pool[0][1]))
        if verbose:
            print(
                f"[SES-Prog] NRE={n_eval:4d} level={levels[stage]} "
                f"dim={mu.size:5d} best={pool[0][1]:.4f}",
                flush=True,
            )

    best_u, best_s = pool[0]
    return SearchResult(
        best_x0=space.to_noise(best_u, k), best_score=best_s, history=history
    )