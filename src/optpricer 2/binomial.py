import numpy as np
from math import exp, sqrt
from typing import Literal
from .core import OptionSpec, CALL, PUT

def crr(opt: OptionSpec, kind: Literal["call","put"]=CALL, N: int = 500, *, american: bool=False) -> float:
    """Cox–Ross–Rubinstein tree. Supports Euro and American (no dividends via discrete steps; q handled in p)."""
    if N <= 0:
        raise ValueError("N must be positive.")
    dt = opt.T / N
    u  = exp(opt.sigma * sqrt(dt))
    d  = 1.0 / u
    disc = exp(-opt.r * dt)
    p = (exp((opt.r - opt.q) * dt) - d) / (u - d)
    if not (0.0 < p < 1.0):
        raise ValueError("Risk-neutral prob p out of (0,1); try larger N or different params.")

    # Payoff at maturity
    j = np.arange(N + 1)
    ST = opt.S0 * (u ** j) * (d ** (N - j))
    if kind == CALL:
        V = np.maximum(ST - opt.K, 0.0)
    else:
        V = np.maximum(opt.K - ST, 0.0)

    # Backward induction
    for k in range(N - 1, -1, -1):
        V = disc * (p * V[1:] + (1.0 - p) * V[:-1])
        if american:
            j = np.arange(k + 1)
            S_k = opt.S0 * (u ** j) * (d ** (k - j))
            if kind == CALL:
                V = np.maximum(V, S_k - opt.K)
            else:
                V = np.maximum(V, opt.K - S_k)

    return float(V[0])
