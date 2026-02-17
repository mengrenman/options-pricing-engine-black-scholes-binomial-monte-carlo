# black_scholes_vec.py
# Vectorised Black-Scholes pricing, Greeks, and implied-vol.
# All public functions accept scalars *or* NumPy arrays and broadcast.

from __future__ import annotations
import numpy as np
from scipy.stats import norm

_N = norm.cdf   # vectorised standard-normal CDF
_n = norm.pdf   # vectorised standard-normal PDF


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _d1_d2(S, K, T, r, q, sigma):
    """Compute d1, d2 arrays.  All inputs broadcast."""
    S, K, T, r, q, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, q, sigma))
    sqrt_T = np.sqrt(T)
    sig_sqrt_T = sigma * sqrt_T
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sig_sqrt_T
    d2 = d1 - sig_sqrt_T
    return d1, d2


def _is_call(kind) -> np.ndarray:
    """Return boolean mask: True where kind == 'call'."""
    kind = np.asarray(kind)
    if kind.ndim == 0:
        return np.bool_(str(kind) == "call")
    return np.array([str(k) == "call" for k in kind.flat], dtype=bool).reshape(kind.shape)


# ---------------------------------------------------------------------------
# Vectorised price
# ---------------------------------------------------------------------------
def bs_price_vec(S, K, T, r, q, sigma, kind) -> np.ndarray:
    """Vectorised Black-Scholes price.

    Parameters accept scalars or arrays; NumPy broadcasting rules apply.

    Returns
    -------
    np.ndarray
        Option prices (same shape as broadcasted inputs).
    """
    S, K, T, r, q, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, q, sigma))
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)

    call_px = disc_q * S * _N(d1) - disc_r * K * _N(d2)
    put_px  = disc_r * K * _N(-d2) - disc_q * S * _N(-d1)

    is_call = _is_call(kind)
    return np.where(is_call, call_px, put_px)


# ---------------------------------------------------------------------------
# Vectorised Greeks
# ---------------------------------------------------------------------------
def bs_greeks_vec(S, K, T, r, q, sigma, kind) -> dict[str, np.ndarray]:
    """Vectorised Black-Scholes Greeks.

    Returns dict with keys: delta, gamma, vega, theta, rho.
    Vega is dPrice/dSigma (absolute), theta is dPrice/dT (per year).
    """
    S, K, T, r, q, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, q, sigma))
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    sqrt_T = np.sqrt(T)
    n_d1 = _n(d1)
    is_call = _is_call(kind)

    # Common
    gamma = disc_q * n_d1 / (S * sigma * sqrt_T)
    vega  = S * disc_q * n_d1 * sqrt_T

    # Call-specific
    delta_c = disc_q * _N(d1)
    theta_c = (-S * disc_q * n_d1 * sigma / (2 * sqrt_T)
               - r * K * disc_r * _N(d2)
               + q * S * disc_q * _N(d1))
    rho_c   = K * T * disc_r * _N(d2)

    # Put-specific
    delta_p = disc_q * (_N(d1) - 1.0)
    theta_p = (-S * disc_q * n_d1 * sigma / (2 * sqrt_T)
               + r * K * disc_r * _N(-d2)
               - q * S * disc_q * _N(-d1))
    rho_p   = -K * T * disc_r * _N(-d2)

    delta = np.where(is_call, delta_c, delta_p)
    theta = np.where(is_call, theta_c, theta_p)
    rho   = np.where(is_call, rho_c, rho_p)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


# ---------------------------------------------------------------------------
# Vectorised implied-vol (Newton-Raphson — vectorisable, unlike Brent)
# ---------------------------------------------------------------------------
def bs_implied_vol_vec(
    S, K, T, r, q, target_prices, kind,
    *, tol: float = 1e-8, maxiter: int = 50,
    init_vol: float = 0.3,
) -> np.ndarray:
    """Recover implied vol from market prices via Newton-Raphson on vega.

    Parameters
    ----------
    target_prices : array-like
        Observed market prices.
    init_vol : float
        Initial guess for sigma (same for all entries).

    Returns
    -------
    np.ndarray
        Implied volatilities.  Entries that fail to converge are ``NaN``.
    """
    S, K, T, r, q, target_prices = (
        np.asarray(x, dtype=float) for x in (S, K, T, r, q, target_prices)
    )
    # broadcast to common shape
    shape = np.broadcast_shapes(
        S.shape, K.shape, T.shape, r.shape, q.shape, target_prices.shape
    )
    sigma = np.full(shape, init_vol, dtype=float)

    for _ in range(maxiter):
        px = bs_price_vec(S, K, T, r, q, sigma, kind)
        d1, _ = _d1_d2(S, K, T, r, q, sigma)
        disc_q = np.exp(-q * T)
        vega = S * disc_q * _n(d1) * np.sqrt(T)

        # Avoid division by zero
        vega_safe = np.where(vega > 1e-15, vega, np.nan)
        step = (px - target_prices) / vega_safe
        sigma = sigma - step

        # Clamp to sensible range
        sigma = np.clip(sigma, 1e-6, 10.0)

        if np.all(np.abs(step) < tol):
            break

    # Mark non-converged entries
    px_final = bs_price_vec(S, K, T, r, q, sigma, kind)
    bad = np.abs(px_final - target_prices) > tol * 100
    sigma = np.where(bad, np.nan, sigma)
    return sigma
