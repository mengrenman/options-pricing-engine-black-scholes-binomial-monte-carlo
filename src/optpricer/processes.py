# processes.py
# Path generators for Monte Carlo pricing.
# All functions return an array of shape (n_steps+1, n_paths_eff)
# that includes the t=0 row with S0. If antithetic=True, the number
# of returned paths is doubled (n_paths_eff = 2 * n_paths).

from __future__ import annotations
import numpy as np
from typing import Callable, Optional


__all__ = [
    "gbm_paths",
    "merton_jump_paths",
    "heston_paths",
    "sabr_paths",
    "local_vol_paths",
    "gbm_milstein_paths",
    "milstein_local_vol_paths",
]


def _rng(seed: Optional[int]):
    return np.random.default_rng(seed)


# -----------------------------
# 1) Geometric Brownian Motion
# -----------------------------
def gbm_paths(
    S0: float, r: float, q: float, sigma: float,
    T: float, n_steps: int, n_paths: int,
    *, antithetic: bool = True, seed: Optional[int] = None
) -> np.ndarray:
    """
    Exact-discretization GBM under Q:
        dS/S = (r - q) dt + sigma dW
        S_{t+dt} = S_t * exp((r - q - 0.5*sigma^2) dt + sigma * sqrt(dt) * Z)
    """
    if n_steps <= 0 or n_paths <= 0:
        raise ValueError("n_steps and n_paths must be positive.")

    rng = _rng(seed)
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma * sigma) * dt
    vol = sigma * np.sqrt(dt)

    Z = rng.standard_normal((n_steps, n_paths))
    if antithetic:
        Z = np.concatenate([Z, -Z], axis=1)

    log_increments = drift + vol * Z
    log_paths = np.cumsum(log_increments, axis=0)
    S = S0 * np.exp(log_paths)
    S = np.vstack([np.full((1, S.shape[1]), S0, dtype=S.dtype), S])
    return S


# ------------------------------------
# 2) Merton Jump-Diffusion (lognormal)
# ------------------------------------
def merton_jump_paths(
    S0: float, r: float, q: float, sigma: float,
    T: float, n_steps: int, n_paths: int,
    *, lam: float,           # jump intensity λ (per year)
    mJ: float,               # mean of jump log-size Y ~ N(mJ, sJ^2)
    sJ: float,               # std of jump log-size
    antithetic: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Merton model under Q:
        dS/S = (r - q - λκ) dt + σ dW + (e^Y - 1) dN,
    where N is Poisson(λ t), Y ~ N(mJ, sJ^2), κ = E[e^Y - 1] = exp(mJ + 0.5 sJ^2) - 1.
    Implementation uses exact GBM step + compound Poisson jump in log space.
    """
    if n_steps <= 0 or n_paths <= 0:
        raise ValueError("n_steps and n_paths must be positive.")
    if lam < 0 or sJ < 0:
        raise ValueError("lam and sJ must be non-negative.")

    rng = _rng(seed)
    dt = T / n_steps
    kappa = np.exp(mJ + 0.5 * sJ * sJ) - 1.0
    drift = (r - q - 0.5 * sigma * sigma - lam * kappa) * dt
    vol = sigma * np.sqrt(dt)

    # Diffusion shocks
    Z = rng.standard_normal((n_steps, n_paths))

    # Jump draws BEFORE antithetic doubling so they pair correctly
    K_base = rng.poisson(lam * dt, size=(n_steps, n_paths))
    ZJ_base = rng.standard_normal(size=(n_steps, n_paths))

    if antithetic:
        Z = np.concatenate([Z, -Z], axis=1)
        # Reuse same Poisson counts; negate jump normals for antithetic pair
        K = np.concatenate([K_base, K_base], axis=1)
        ZJ = np.concatenate([ZJ_base, -ZJ_base], axis=1)
    else:
        K = K_base
        ZJ = ZJ_base

    # Sum of K normal jump sizes ~ Normal(K*mJ, sqrt(K)*sJ)
    Y_sum = mJ * K + sJ * np.sqrt(K) * ZJ  # 0 where K=0

    log_increments = drift + vol * Z + Y_sum
    log_paths = np.cumsum(log_increments, axis=0)
    S = S0 * np.exp(log_paths)
    S = np.vstack([np.full((1, S.shape[1]), S0, dtype=S.dtype), S])
    return S


# -------------------------------
# 3) Heston (CIR variance process)
# -------------------------------
def heston_paths(
    S0: float, r: float, q: float,
    v0: float, kappa: float, theta: float, xi: float, rho: float,
    T: float, n_steps: int, n_paths: int,
    *, antithetic: bool = True, seed: Optional[int] = None, return_variance: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Heston under Q:
        dS = (r - q) S dt + sqrt(v) S dW1
        dv = kappa (theta - v) dt + xi sqrt(v) dW2,   corr(dW1, dW2) = rho
    Discretization: Full-Truncation Euler for v (keeps v >= 0), log-Euler for S using v_t.
    """
    if n_steps <= 0 or n_paths <= 0:
        raise ValueError("n_steps and n_paths must be positive.")
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("rho must be in [-1, 1].")

    rng = _rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    Z2 = rng.standard_normal((n_steps, n_paths))
    Zp = rng.standard_normal((n_steps, n_paths))
    if antithetic:
        Z2 = np.concatenate([Z2, -Z2], axis=1)
        Zp = np.concatenate([Zp, -Zp], axis=1)
    Z1 = rho * Z2 + np.sqrt(max(0.0, 1.0 - rho * rho)) * Zp  # correlate

    n_cols = Z1.shape[1]
    S = np.empty((n_steps + 1, n_cols), dtype=float)
    v = np.empty_like(S)

    S[0, :] = S0
    v_t = np.full(n_cols, max(v0, 0.0), dtype=float)
    v[0, :] = v_t

    for t in range(n_steps):
        v_eff = np.maximum(v_t, 0.0)
        # Asset step (log-Euler with current variance)
        S[t + 1, :] = S[t, :] * np.exp((r - q - 0.5 * v_eff) * dt + np.sqrt(v_eff) * sqrt_dt * Z1[t, :])
        # Variance step (Full-Truncation Euler)
        v_t = v_t + kappa * (theta - v_eff) * dt + xi * np.sqrt(v_eff) * sqrt_dt * Z2[t, :]
        v_t = np.maximum(v_t, 0.0)
        v[t + 1, :] = v_t

    return (S, v) if return_variance else S


# ---------------------------
# 4) SABR (σ lognormal case)
# ---------------------------
def sabr_paths(
    S0: float, r: float, q: float,
    alpha0: float, beta: float, nu: float, rho: float,
    T: float, n_steps: int, n_paths: int,
    *, antithetic: bool = True, seed: Optional[int] = None
) -> np.ndarray:
    """
    SABR (Hagan 2002) with lognormal volatility:
        dS = (r - q) S dt + sigma * S^beta dW1
        d(sigma) = nu * sigma dW2, corr(dW1, dW2) = rho
    Volatility is evolved exactly (lognormal). Price uses log-Euler if beta=1,
    and Euler (positive clamp) otherwise.
    """
    if n_steps <= 0 or n_paths <= 0:
        raise ValueError("n_steps and n_paths must be positive.")
    if not (0.0 <= beta <= 1.0):
        raise ValueError("beta must be in [0, 1].")
    if alpha0 <= 0.0 or nu < 0.0:
        raise ValueError("alpha0 must be >0, nu >= 0.")
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("rho must be in [-1, 1].")

    rng = _rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    Z2 = rng.standard_normal((n_steps, n_paths))
    Zp = rng.standard_normal((n_steps, n_paths))
    if antithetic:
        Z2 = np.concatenate([Z2, -Z2], axis=1)
        Zp = np.concatenate([Zp, -Zp], axis=1)
    Z1 = rho * Z2 + np.sqrt(max(0.0, 1.0 - rho * rho)) * Zp

    n_cols = Z1.shape[1]
    S = np.empty((n_steps + 1, n_cols), dtype=float)
    S[0, :] = S0
    sigma_t = np.full(n_cols, alpha0, dtype=float)

    for t in range(n_steps):
        # evolve sigma exactly (lognormal)
        sigma_t *= np.exp(nu * sqrt_dt * Z2[t, :] - 0.5 * (nu * nu) * dt)
        if beta == 1.0:
            # log-Euler (exact for GBM with sigma_t)
            S[t + 1, :] = S[t, :] * np.exp((r - q - 0.5 * sigma_t**2) * dt + sigma_t * sqrt_dt * Z1[t, :])
        else:
            # Euler with positivity clamp
            S[t + 1, :] = S[t, :] + (r - q) * S[t, :] * dt + sigma_t * (S[t, :] ** beta) * sqrt_dt * Z1[t, :]
            S[t + 1, :] = np.maximum(S[t + 1, :], 1e-12)

    return S


# -----------------------------------------
# 5) Local Volatility (Dupire-style driver)
# -----------------------------------------
def local_vol_paths(
    S0: float, r: float, q: float,
    T: float, n_steps: int, n_paths: int,
    sigma_loc: Callable[[np.ndarray, float], np.ndarray],
    *, antithetic: bool = True, seed: Optional[int] = None
) -> np.ndarray:
    """
    Local-vol paths with user-supplied sigma_loc(S, t) that returns σ(S,t).
    Uses log-Euler to keep prices positive:
        S_{t+dt} = S_t * exp((r - q - 0.5*σ^2) dt + σ sqrt(dt) Z)
    sigma_loc must accept a vector of S_t (shape (n_paths_eff,)) and scalar t,
    and return a vector of the same shape.
    """
    if n_steps <= 0 or n_paths <= 0:
        raise ValueError("n_steps and n_paths must be positive.")

    rng = _rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    Z = rng.standard_normal((n_steps, n_paths))
    if antithetic:
        Z = np.concatenate([Z, -Z], axis=1)

    n_cols = Z.shape[1]
    S = np.empty((n_steps + 1, n_cols), dtype=float)
    S[0, :] = S0

    for t in range(n_steps):
        t_now = t * dt
        sig = np.asarray(sigma_loc(S[t, :], t_now), dtype=float)
        # guard against pathological outputs
        sig = np.clip(sig, 0.0, np.inf)
        S[t + 1, :] = S[t, :] * np.exp((r - q - 0.5 * sig * sig) * dt + sig * sqrt_dt * Z[t, :])

    return S


# ---------------------------------------------------------------------------
# 6) GBM Milstein (constant vol — demonstrates the scheme)
# ---------------------------------------------------------------------------
def gbm_milstein_paths(
    S0: float, r: float, q: float, sigma: float,
    T: float, n_steps: int, n_paths: int,
    *, antithetic: bool = True, seed: Optional[int] = None
) -> np.ndarray:
    """GBM Milstein paths (constant vol).

    For the SDE  dS = (r-q)S dt + σ S dW  the Milstein scheme is::

        S_{n+1} = S_n + (r-q) S_n dt + σ S_n √dt Z
                  + ½ σ² S_n (Z² - 1) dt

    With constant σ this is algebraically equivalent to the exact
    log-Euler discretisation (strong order 1.0), so this function
    exists primarily for **demonstration and convergence testing**.
    """
    if n_steps <= 0 or n_paths <= 0:
        raise ValueError("n_steps and n_paths must be positive.")

    rng = _rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    Z = rng.standard_normal((n_steps, n_paths))
    if antithetic:
        Z = np.concatenate([Z, -Z], axis=1)

    n_cols = Z.shape[1]
    S = np.empty((n_steps + 1, n_cols), dtype=float)
    S[0, :] = S0

    for t in range(n_steps):
        Zt = Z[t, :]
        S_t = S[t, :]
        # Milstein step (explicit)
        S[t + 1, :] = (S_t
                        + (r - q) * S_t * dt
                        + sigma * S_t * sqrt_dt * Zt
                        + 0.5 * sigma**2 * S_t * (Zt**2 - 1.0) * dt)
        S[t + 1, :] = np.maximum(S[t + 1, :], 1e-10)

    return S


# ---------------------------------------------------------------------------
# 7) Milstein for local vol
# ---------------------------------------------------------------------------
def milstein_local_vol_paths(
    S0: float, r: float, q: float,
    T: float, n_steps: int, n_paths: int,
    sigma_loc: Callable[[np.ndarray, float], np.ndarray],
    *, antithetic: bool = True, seed: Optional[int] = None,
    dS_bump: float = 0.01,
) -> np.ndarray:
    """Local-vol Milstein paths (strong order 1.0).

    For the SDE  dS = (r-q) S dt + σ(S,t) S dW  the Milstein scheme is::

        S_{n+1} = S_n + (r-q) S_n dt + σ_n S_n √dt Z
                  + ½ σ_n σ'_n S_n² (Z² - 1) dt

    where σ'(S,t) = ∂[σ(S,t)·S]/∂S is approximated via central finite
    differences using a bump of size ``dS_bump * S``.

    Parameters
    ----------
    S0, r, q, T, n_steps, n_paths : numeric
        Model parameters.
    sigma_loc : callable
        ``sigma_loc(S_array, t) -> sigma_array``.
    dS_bump : float
        Relative bump size for the σ′ finite-difference (default 0.01).

    Returns
    -------
    ndarray, shape (n_steps+1, n_paths_eff)
    """
    if n_steps <= 0 or n_paths <= 0:
        raise ValueError("n_steps and n_paths must be positive.")

    rng = _rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    Z = rng.standard_normal((n_steps, n_paths))
    if antithetic:
        Z = np.concatenate([Z, -Z], axis=1)

    n_cols = Z.shape[1]
    S = np.empty((n_steps + 1, n_cols), dtype=float)
    S[0, :] = S0

    for t in range(n_steps):
        t_now = t * dt
        Zt = Z[t, :]
        S_t = S[t, :]

        sig = np.asarray(sigma_loc(S_t, t_now), dtype=float)
        sig = np.clip(sig, 1e-8, 10.0)

        # Compute d(σ·S)/dS ≈ [σ(S+ε)·(S+ε) - σ(S-ε)·(S-ε)] / (2ε)
        eps = dS_bump * S_t
        S_up = S_t + eps
        S_dn = np.maximum(S_t - eps, 1e-10)
        sig_up = np.asarray(sigma_loc(S_up, t_now), dtype=float)
        sig_dn = np.asarray(sigma_loc(S_dn, t_now), dtype=float)
        # d(σ·S)/dS for the diffusion coefficient a(S) = σ(S)·S
        da_dS = (sig_up * S_up - sig_dn * S_dn) / (S_up - S_dn)

        # Milstein step
        a_t = sig * S_t  # diffusion coefficient
        S[t + 1, :] = (S_t
                        + (r - q) * S_t * dt
                        + a_t * sqrt_dt * Zt
                        + 0.5 * a_t * da_dS * (Zt**2 - 1.0) * dt)
        S[t + 1, :] = np.maximum(S[t + 1, :], 1e-10)

    return S
