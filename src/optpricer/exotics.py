# exotics.py
# Path-dependent exotic option pricing via Monte Carlo.
#
# All functions accept pre-generated paths from ``processes.py``
# (shape ``(n_steps+1, n_paths_eff)`` including the t=0 row)
# and return ``(price, stderr)``.
#
# This design decouples the stochastic process (GBM, Heston, …)
# from the payoff, so any combination is supported.

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------
def _price_from_payoff(
    payoff: np.ndarray, r: float, T: float
) -> tuple[float, float]:
    """Discount a payoff vector and return (price, stderr)."""
    disc = np.exp(-r * T)
    X = disc * payoff
    n = X.size
    mean = float(X.mean())
    se = float(X.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mean, se


# ---------------------------------------------------------------------------
# Barrier options
# ---------------------------------------------------------------------------
def barrier_price(
    paths: np.ndarray,
    K: float,
    r: float,
    T: float,
    kind: str,
    barrier: float,
    barrier_type: str,
    rebate: float = 0.0,
) -> tuple[float, float]:
    """Price a European barrier option with discrete monitoring.

    Parameters
    ----------
    paths : ndarray, shape (n_steps+1, n_paths)
        Asset price paths including t=0 row.
    K : float
        Strike.
    r : float
        Risk-free rate.
    T : float
        Time to expiry.
    kind : str
        ``"call"`` or ``"put"``.
    barrier : float
        Barrier level.
    barrier_type : str
        One of ``"up-and-out"``, ``"up-and-in"``,
        ``"down-and-out"``, ``"down-and-in"``.
    rebate : float
        Amount paid when the option is knocked out (default 0).

    Returns
    -------
    tuple[float, float]
        ``(price, stderr)``
    """
    valid_types = {"up-and-out", "up-and-in", "down-and-out", "down-and-in"}
    if barrier_type not in valid_types:
        raise ValueError(f"barrier_type must be one of {valid_types}, got {barrier_type!r}")

    ST = paths[-1, :]

    # Determine barrier crossing
    if barrier_type.startswith("up"):
        crossed = np.any(paths >= barrier, axis=0)
    else:
        crossed = np.any(paths <= barrier, axis=0)

    # Vanilla payoff at maturity
    if kind == "call":
        vanilla = np.maximum(ST - K, 0.0)
    elif kind == "put":
        vanilla = np.maximum(K - ST, 0.0)
    else:
        raise ValueError("kind must be 'call' or 'put'")

    # Combine payoff based on knock-in / knock-out
    if barrier_type.endswith("out"):
        payoff = np.where(crossed, rebate, vanilla)
    else:  # knock-in
        payoff = np.where(crossed, vanilla, rebate)

    return _price_from_payoff(payoff, r, T)


# ---------------------------------------------------------------------------
# Asian options
# ---------------------------------------------------------------------------
def asian_price(
    paths: np.ndarray,
    K: float,
    r: float,
    T: float,
    kind: str,
    average_type: str = "arithmetic",
    strike_type: str = "fixed",
) -> tuple[float, float]:
    """Price a European Asian option.

    Parameters
    ----------
    paths : ndarray, shape (n_steps+1, n_paths)
        Asset price paths including t=0 row.
    K : float
        Strike (used for fixed-strike; ignored for floating).
    r : float
        Risk-free rate.
    T : float
        Time to expiry.
    kind : str
        ``"call"`` or ``"put"``.
    average_type : str
        ``"arithmetic"`` (default) or ``"geometric"``.
    strike_type : str
        ``"fixed"`` (payoff on avg vs K) or ``"floating"`` (payoff on S_T vs avg).

    Returns
    -------
    tuple[float, float]
        ``(price, stderr)``
    """
    # Exclude t=0 row from average (standard Asian convention)
    monitoring = paths[1:, :]
    ST = paths[-1, :]

    if average_type == "arithmetic":
        avg = monitoring.mean(axis=0)
    elif average_type == "geometric":
        avg = np.exp(np.log(monitoring).mean(axis=0))
    else:
        raise ValueError("average_type must be 'arithmetic' or 'geometric'")

    if strike_type == "fixed":
        if kind == "call":
            payoff = np.maximum(avg - K, 0.0)
        elif kind == "put":
            payoff = np.maximum(K - avg, 0.0)
        else:
            raise ValueError("kind must be 'call' or 'put'")
    elif strike_type == "floating":
        if kind == "call":
            payoff = np.maximum(ST - avg, 0.0)
        elif kind == "put":
            payoff = np.maximum(avg - ST, 0.0)
        else:
            raise ValueError("kind must be 'call' or 'put'")
    else:
        raise ValueError("strike_type must be 'fixed' or 'floating'")

    return _price_from_payoff(payoff, r, T)


# ---------------------------------------------------------------------------
# Digital (binary / cash-or-nothing) options
# ---------------------------------------------------------------------------
def digital_price(
    paths: np.ndarray,
    K: float,
    r: float,
    T: float,
    kind: str,
    payout: float = 1.0,
) -> tuple[float, float]:
    """Price a European cash-or-nothing digital option.

    Pays ``payout`` if ITM at expiry, else 0.

    Parameters
    ----------
    paths : ndarray, shape (n_steps+1, n_paths)
        Asset price paths (only terminal value is used).
    K : float
        Strike.
    r : float
        Risk-free rate.
    T : float
        Time to expiry.
    kind : str
        ``"call"`` (pays if S_T > K) or ``"put"`` (pays if S_T < K).
    payout : float
        Cash payout amount (default 1).

    Returns
    -------
    tuple[float, float]
        ``(price, stderr)``
    """
    ST = paths[-1, :]
    if kind == "call":
        payoff = np.where(ST > K, payout, 0.0)
    elif kind == "put":
        payoff = np.where(ST < K, payout, 0.0)
    else:
        raise ValueError("kind must be 'call' or 'put'")

    return _price_from_payoff(payoff, r, T)


# ---------------------------------------------------------------------------
# Lookback options
# ---------------------------------------------------------------------------
def lookback_price(
    paths: np.ndarray,
    r: float,
    T: float,
    kind: str,
    K: float = 0.0,
    strike_type: str = "floating",
) -> tuple[float, float]:
    """Price a European lookback option.

    Parameters
    ----------
    paths : ndarray, shape (n_steps+1, n_paths)
        Asset price paths including t=0 row.
    r : float
        Risk-free rate.
    T : float
        Time to expiry.
    kind : str
        ``"call"`` or ``"put"``.
    K : float
        Strike (only used for ``strike_type="fixed"``).
    strike_type : str
        ``"floating"`` (default) or ``"fixed"``.

    Returns
    -------
    tuple[float, float]
        ``(price, stderr)``

    Notes
    -----
    Floating lookback call: payoff = S_T - S_min
    Floating lookback put:  payoff = S_max - S_T
    Fixed lookback call:    payoff = max(S_max - K, 0)
    Fixed lookback put:     payoff = max(K - S_min, 0)
    """
    S_max = paths.max(axis=0)
    S_min = paths.min(axis=0)
    ST = paths[-1, :]

    if strike_type == "floating":
        if kind == "call":
            payoff = ST - S_min
        elif kind == "put":
            payoff = S_max - ST
        else:
            raise ValueError("kind must be 'call' or 'put'")
    elif strike_type == "fixed":
        if kind == "call":
            payoff = np.maximum(S_max - K, 0.0)
        elif kind == "put":
            payoff = np.maximum(K - S_min, 0.0)
        else:
            raise ValueError("kind must be 'call' or 'put'")
    else:
        raise ValueError("strike_type must be 'floating' or 'fixed'")

    return _price_from_payoff(payoff, r, T)
