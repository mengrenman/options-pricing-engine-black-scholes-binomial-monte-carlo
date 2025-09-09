# optpricer/monte_carlo.py

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---- helper: one simulation chunk (no path storage, only terminal S_T) ----

def _mc_chunk_sumstats(
    n: int,
    *,
    S0: float, K: float, T: float, r: float, q: float, sigma: float,
    kind: str, antithetic: bool, seed: np.random.SeedSequence | int | None,
    dtype=np.float64,
):
    """
    Simulate `n` terminal draws of S_T under GBM, compute discounted payoff X and
    control variate Y = e^{-rT} S_T. Return sufficient statistics to aggregate:
        n_eff, sumX, sumX2, sumY, sumY2, sumXY
    """
    if n <= 0:
        return (0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # independent RNG per chunk
    rng = np.random.default_rng(seed)
    dt = T
    mu = (r - q - 0.5 * sigma * sigma) * dt
    sig = sigma * math.sqrt(dt)
    df = math.exp(-r * T)

    # draw normals
    m = n
    Z = rng.standard_normal(m).astype(dtype, copy=False)

    # antithetic doubling (if requested)
    if antithetic:
        Z = np.concatenate([Z, -Z], axis=0)

    # terminal price (exact scheme)
    ST = (S0 * np.exp(mu + sig * Z)).astype(dtype, copy=False)

    # discounted payoff X
    if kind.lower() == "call":
        payoff = np.maximum(ST - K, 0.0, dtype=dtype)
    elif kind.lower() == "put":
        payoff = np.maximum(K - ST, 0.0, dtype=dtype)
    else:
        raise ValueError("kind must be 'call' or 'put'")

    X = df * payoff

    # control variate Y = e^{-rT} S_T, with known expectation EY = S0 * e^{-qT}
    Y = (df * ST).astype(dtype, copy=False)

    # sufficient statistics
    n_eff = X.size
    sumX  = float(X.sum())
    sumX2 = float((X * X).sum())
    sumY  = float(Y.sum())
    sumY2 = float((Y * Y).sum())
    sumXY = float((X * Y).sum())
    return (n_eff, sumX, sumX2, sumY, sumY2, sumXY)


def _aggregate_stats(stats_list):
    n = sum(s[0] for s in stats_list)
    sumX  = sum(s[1] for s in stats_list)
    sumX2 = sum(s[2] for s in stats_list)
    sumY  = sum(s[3] for s in stats_list)
    sumY2 = sum(s[4] for s in stats_list)
    sumXY = sum(s[5] for s in stats_list)
    return n, sumX, sumX2, sumY, sumY2, sumXY


def euro_price_mc(
    opt,
    kind: str, *,
    n_paths: int = 100_000,
    seed: int | None = None,
    chunk_size: int = 100_000,
    antithetic: bool = True,
    control_variate: bool = True,
    n_workers: int = 1,
    dtype=np.float64,
    return_stderr: bool = True
):
    """
    Memory-light European option Monte-Carlo pricer (terminal-only).
    Returns (price, stderr).

    - Streams in chunks to cap memory.
    - Optional antithetic variates.
    - Optional control variate Y = e^{-rT}S_T with E[Y] = S0*exp(-qT).
    - Optional process-level parallelism.

    Notes:
    - Works great in scripts. In Jupyter, prefer n_workers=1 (or use joblib/loky externally).
    - For path-dependent payoffs, use a different routine (time stepping + on-the-fly accumulation).
    """
    S0, K, T, r, sigma = opt.S0, opt.K, opt.T, opt.r, opt.sigma
    q = getattr(opt, "q", 0.0)

    # set up SeedSequence tree so each chunk/worker has an independent stream
    ss_root = np.random.SeedSequence(seed)

    # plan chunks
    chunks = []
    remaining = int(n_paths)
    while remaining > 0:
        m = min(chunk_size, remaining)
        chunks.append(m)
        remaining -= m

    # serial or parallel execution
    stats_list = []
    if n_workers <= 1:
        for i, m in enumerate(chunks):
            ss = ss_root.spawn(1)[0]
            stats = _mc_chunk_sumstats(
                m, S0=S0, K=K, T=T, r=r, q=q, sigma=sigma,
                kind=kind, antithetic=antithetic, seed=ss, dtype=dtype
            )
            stats_list.append(stats)
    else:
        # process pool — safe in scripts; in notebooks prefer n_workers=1
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = []
            # spawn a seed for each chunk
            child_seeds = ss_root.spawn(len(chunks))
            for m, ss in zip(chunks, child_seeds):
                futs.append(ex.submit(
                    _mc_chunk_sumstats, m,
                    S0=S0, K=K, T=T, r=r, q=q, sigma=sigma,
                    kind=kind, antithetic=antithetic, seed=ss, dtype=dtype
                ))
            for f in as_completed(futs):
                stats_list.append(f.result())

    # aggregate
    n, sumX, sumX2, sumY, sumY2, sumXY = _aggregate_stats(stats_list)
    if n == 0:
        return float("nan"), float("nan")

    # plain estimator
    meanX = sumX / n
    varX  = max(0.0, sumX2 / n - meanX * meanX)

    if control_variate:
        # c_hat = Cov(X,Y)/Var(Y)
        meanY = sumY / n
        varY  = max(0.0, sumY2 / n - meanY * meanY)
        covXY = (sumXY / n) - meanX * meanY
        c_hat = 0.0 if varY == 0.0 else (covXY / varY)

        EY = S0 * math.exp(-q * T)  # known under RN measure
        mean_cv = meanX - c_hat * (meanY - EY)

        # variance of the CV estimator:
        var_cv = varX - 2.0 * c_hat * covXY + (c_hat * c_hat) * varY
        se = math.sqrt(max(0.0, var_cv) / n)
        return (float(mean_cv), float(se)) if return_stderr else float(meanX)

    # no control variate
    se = math.sqrt(max(0.0, varX) / n)
    return (float(meanX), float(se)) if return_stderr else float(meanX)
