import numpy as np
from math import exp, sqrt
from typing import Literal
from .core import OptionSpec, CALL, PUT


def crr(opt: OptionSpec, kind: Literal["call","put"]=CALL, N: int = 500, *, american: bool=False) -> float:
    """Cox-Ross-Rubinstein tree. Supports Euro and American (no dividends via discrete steps; q handled in p)."""
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


# ---------------------------------------------------------------------------
# Vectorised CRR — prices a batch of options sharing the same tree params
# ---------------------------------------------------------------------------
def crr_vec(
    S0: float, K, T: float, r: float, q: float, sigma: float,
    kind, N: int = 500, *, american: bool = False,
) -> np.ndarray:
    """Vectorised CRR tree pricing over arrays of strikes and/or kinds.

    Builds **one** tree for (S0, T, r, q, sigma) and evaluates the payoff
    for every (K, kind) pair in a single backward pass.

    Parameters
    ----------
    S0 : float
        Current spot.
    K : array-like
        Strike(s).
    T, r, q, sigma : float
        Shared tree parameters.
    kind : str or array-like of str
        ``"call"`` / ``"put"`` per strike.
    N : int
        Number of time steps.
    american : bool
        Enable early exercise.

    Returns
    -------
    np.ndarray
        Prices, same shape as *K*.
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    kind = np.atleast_1d(np.asarray(kind))
    if kind.shape != K.shape:
        kind = np.broadcast_to(kind, K.shape)

    if N <= 0:
        raise ValueError("N must be positive.")
    dt = T / N
    u  = exp(sigma * sqrt(dt))
    d  = 1.0 / u
    disc = exp(-r * dt)
    p = (exp((r - q) * dt) - d) / (u - d)
    if not (0.0 < p < 1.0):
        raise ValueError("Risk-neutral prob p out of (0,1); try larger N or different params.")

    # Node prices at maturity — shape (N+1,)
    j = np.arange(N + 1)
    ST = S0 * (u ** j) * (d ** (N - j))

    # Payoffs — shape (n_options, N+1)
    is_call = np.array([str(k) == "call" for k in kind.flat], dtype=bool).reshape(K.shape)
    call_pay = np.maximum(ST[np.newaxis, :] - K[:, np.newaxis], 0.0)
    put_pay  = np.maximum(K[:, np.newaxis] - ST[np.newaxis, :], 0.0)
    V = np.where(is_call[:, np.newaxis], call_pay, put_pay)

    # Backward induction
    for step in range(N - 1, -1, -1):
        V = disc * (p * V[:, 1:] + (1.0 - p) * V[:, :-1])
        if american:
            j_k = np.arange(step + 1)
            S_k = S0 * (u ** j_k) * (d ** (step - j_k))
            call_ex = S_k[np.newaxis, :] - K[:, np.newaxis]
            put_ex  = K[:, np.newaxis] - S_k[np.newaxis, :]
            ex_val  = np.where(is_call[:, np.newaxis], call_ex, put_ex)
            V = np.maximum(V, ex_val)

    return V[:, 0]
