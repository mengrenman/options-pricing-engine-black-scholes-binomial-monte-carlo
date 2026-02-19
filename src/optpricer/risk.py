"""Bump-and-reprice risk engine.

Provides generic numerical Greeks via central finite differences that work
with **any** pricer callable, as well as scenario-grid evaluation,
portfolio-level risk aggregation, and historical VaR / CVaR.
"""

from __future__ import annotations

import numpy as np
from typing import Callable
from dataclasses import replace

__all__ = [
    "numerical_greeks",
    "scenario_grid",
    "portfolio_risk",
    "var_historical",
    "cvar_historical",
]


# ---------------------------------------------------------------------------
# Numerical Greeks
# ---------------------------------------------------------------------------

def numerical_greeks(
    pricer_func: Callable[..., float],
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    kind: str,
    *,
    bump_pct: float = 0.01,
) -> dict[str, float]:
    """Compute Greeks via central finite differences on an arbitrary pricer.

    Parameters
    ----------
    pricer_func : callable
        ``pricer_func(S, K, T, r, q, sigma, kind) -> float``.
    S, K, T, r, q, sigma : float
        Market and instrument parameters.
    kind : str
        ``"call"`` or ``"put"``.
    bump_pct : float
        Relative bump size for spot and vol; absolute for rate (default 0.01).

    Returns
    -------
    dict[str, float]
        Keys: ``delta``, ``gamma``, ``vega``, ``theta``, ``rho``.
    """
    P0 = pricer_func(S, K, T, r, q, sigma, kind)

    # --- Delta & Gamma (spot bump) ---
    eps_S = bump_pct * S
    P_up = pricer_func(S + eps_S, K, T, r, q, sigma, kind)
    P_dn = pricer_func(S - eps_S, K, T, r, q, sigma, kind)
    delta = (P_up - P_dn) / (2.0 * eps_S)
    gamma = (P_up - 2.0 * P0 + P_dn) / (eps_S ** 2)

    # --- Vega (vol bump) ---
    eps_v = max(bump_pct * sigma, 1e-4)
    P_vup = pricer_func(S, K, T, r, q, sigma + eps_v, kind)
    P_vdn = pricer_func(S, K, T, r, q, max(sigma - eps_v, 1e-6), kind)
    vega = (P_vup - P_vdn) / (2.0 * eps_v)

    # --- Theta (time decay, 1-day bump) ---
    dt = 1.0 / 365.0
    if T > dt:
        P_t = pricer_func(S, K, T - dt, r, q, sigma, kind)
        theta_val = (P_t - P0) / dt
    else:
        theta_val = 0.0

    # --- Rho (rate bump) ---
    eps_r = bump_pct
    P_rup = pricer_func(S, K, T, r + eps_r, q, sigma, kind)
    P_rdn = pricer_func(S, K, T, r - eps_r, q, sigma, kind)
    rho = (P_rup - P_rdn) / (2.0 * eps_r)

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta_val),
        "rho": float(rho),
    }


# ---------------------------------------------------------------------------
# Scenario grid
# ---------------------------------------------------------------------------

def scenario_grid(
    pricer_func: Callable[..., float],
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    kind: str,
    spot_range: np.ndarray,
    vol_range: np.ndarray,
) -> dict:
    """Evaluate a pricer across a 2-D (spot × vol) scenario grid.

    Parameters
    ----------
    pricer_func : callable
        ``pricer_func(S, K, T, r, q, sigma, kind) -> float``.
    spot_range : array, shape (n_spot,)
        Spot values to evaluate.
    vol_range : array, shape (n_vol,)
        Volatility values to evaluate.

    Returns
    -------
    dict
        ``"spot_values"``, ``"vol_values"``, ``"prices"`` (shape n_spot×n_vol).
    """
    spot_range = np.asarray(spot_range, dtype=float)
    vol_range = np.asarray(vol_range, dtype=float)
    prices = np.empty((len(spot_range), len(vol_range)))

    for i, s in enumerate(spot_range):
        for j, v in enumerate(vol_range):
            prices[i, j] = pricer_func(float(s), K, T, r, q, float(v), kind)

    return {
        "spot_values": spot_range.copy(),
        "vol_values": vol_range.copy(),
        "prices": prices,
    }


# ---------------------------------------------------------------------------
# Portfolio risk
# ---------------------------------------------------------------------------

def portfolio_risk(
    instruments: list[dict],
    pricer_func: Callable[..., float],
    *,
    bump_pct: float = 0.01,
) -> dict:
    """Aggregate risk for a portfolio of instruments.

    Parameters
    ----------
    instruments : list of dict
        Each dict must have keys ``S``, ``K``, ``T``, ``r``, ``q``,
        ``sigma``, ``kind``, ``position`` (signed notional, +1 long,
        −1 short).
    pricer_func : callable
        ``pricer_func(S, K, T, r, q, sigma, kind) -> float``.

    Returns
    -------
    dict
        ``"total_delta"``, ``"total_gamma"``, ``"total_vega"``,
        ``"total_theta"``, ``"total_rho"``, ``"total_value"``,
        ``"instrument_greeks"`` (list of per-instrument dicts).
    """
    totals = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    total_value = 0.0
    inst_greeks = []

    for inst in instruments:
        pos = inst["position"]
        g = numerical_greeks(
            pricer_func,
            inst["S"], inst["K"], inst["T"], inst["r"], inst["q"],
            inst["sigma"], inst["kind"],
            bump_pct=bump_pct,
        )
        price = pricer_func(
            inst["S"], inst["K"], inst["T"], inst["r"], inst["q"],
            inst["sigma"], inst["kind"],
        )
        scaled = {k: pos * v for k, v in g.items()}
        for k in totals:
            totals[k] += scaled[k]
        total_value += pos * price
        inst_greeks.append({**scaled, "price": pos * price})

    return {
        "total_delta": totals["delta"],
        "total_gamma": totals["gamma"],
        "total_vega": totals["vega"],
        "total_theta": totals["theta"],
        "total_rho": totals["rho"],
        "total_value": total_value,
        "instrument_greeks": inst_greeks,
    }


# ---------------------------------------------------------------------------
# VaR / CVaR
# ---------------------------------------------------------------------------

def var_historical(
    returns: np.ndarray,
    confidence: float = 0.99,
    horizon: int = 1,
) -> float:
    """Historical Value-at-Risk.

    VaR is the loss at the ``(1 - confidence)`` quantile, scaled by
    ``sqrt(horizon)`` for multi-day horizon under i.i.d. assumption.

    Returns a **positive** number representing the loss threshold.
    """
    returns = np.asarray(returns, dtype=float)
    q = np.percentile(returns, (1.0 - confidence) * 100.0)
    return float(-q * np.sqrt(horizon))


def cvar_historical(
    returns: np.ndarray,
    confidence: float = 0.99,
    horizon: int = 1,
) -> float:
    """Conditional VaR (Expected Shortfall).

    Mean of losses beyond the VaR threshold, scaled by ``sqrt(horizon)``.

    Returns a **positive** number.
    """
    returns = np.asarray(returns, dtype=float)
    q = np.percentile(returns, (1.0 - confidence) * 100.0)
    tail = returns[returns <= q]
    if len(tail) == 0:
        return float(-q * np.sqrt(horizon))
    return float(-tail.mean() * np.sqrt(horizon))
