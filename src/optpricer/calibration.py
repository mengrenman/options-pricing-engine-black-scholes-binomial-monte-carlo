# calibration.py
# SVI (Stochastic Volatility Inspired) surface fitting and VolSurface.

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# SVI raw parameterisation
# ---------------------------------------------------------------------------
@dataclass
class SVIParams:
    """Raw SVI parameterisation for a single expiry slice.

    The total implied variance is:
        w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

    where k = log(K / F) is log-moneyness (F = forward price).

    Parameters
    ----------
    a, b, rho, m, sigma : float
        SVI parameters.
    expiry : float
        Slice expiry in years (needed to convert between total variance
        and implied vol).
    """
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    expiry: float

    def total_var(self, k: np.ndarray | float) -> np.ndarray:
        """Evaluate total variance w(k)."""
        k = np.asarray(k, dtype=float)
        km = k - self.m
        return self.a + self.b * (
            self.rho * km + np.sqrt(km * km + self.sigma * self.sigma)
        )

    def iv(self, k: np.ndarray | float) -> np.ndarray:
        """Return implied volatility from log-moneyness."""
        w = self.total_var(k)
        return np.sqrt(np.maximum(w, 0.0) / self.expiry)


# ---------------------------------------------------------------------------
# VolSurface — plugs into MarketData.vol_surface
# ---------------------------------------------------------------------------
class VolSurface:
    """Interpolating vol surface built from SVI slices.

    For expiries that fall between calibrated slices, total-variance is
    linearly interpolated (calendar-spread arbitrage free when slices
    are individually arbitrage-free and monotone in total variance).

    Parameters
    ----------
    slices : dict[float, SVIParams]
        Mapping ``{expiry: SVIParams}``.
    forward_curve : dict[float, float] | None
        Mapping ``{expiry: forward}``.  If provided, ``iv()`` can accept
        absolute strikes and convert to log-moneyness automatically.
    """

    def __init__(
        self,
        slices: dict[float, SVIParams],
        forward_curve: dict[float, float] | None = None,
    ):
        if not slices:
            raise ValueError("At least one SVI slice is required.")
        self._slices = dict(sorted(slices.items()))
        self._expiries = np.array(sorted(slices.keys()), dtype=float)
        self._forward_curve = forward_curve or {}

    @property
    def expiries(self) -> np.ndarray:
        return self._expiries.copy()

    def _get_forward(self, T: float) -> float:
        if T in self._forward_curve:
            return self._forward_curve[T]
        # Interpolate / extrapolate from known forwards
        Ts = np.array(sorted(self._forward_curve.keys()), dtype=float)
        Fs = np.array([self._forward_curve[t] for t in sorted(self._forward_curve.keys())],
                       dtype=float)
        if len(Ts) == 0:
            raise ValueError(
                f"Forward not available for T={T}.  Provide forward_curve or "
                "pass log-moneyness directly to iv_from_logm()."
            )
        if len(Ts) == 1:
            return float(Fs[0])
        return float(np.interp(T, Ts, Fs))

    # --- core lookup -------------------------------------------------------
    def iv_from_logm(self, k: np.ndarray | float, T: float) -> np.ndarray:
        """Implied vol from log-moneyness k = log(K/F) at expiry T."""
        k = np.asarray(k, dtype=float)

        # Exact match
        if T in self._slices:
            return self._slices[T].iv(k)

        # Interpolate total variance linearly between nearest slices
        idx = np.searchsorted(self._expiries, T)
        if idx == 0:
            return self._slices[self._expiries[0]].iv(k)
        if idx >= len(self._expiries):
            return self._slices[self._expiries[-1]].iv(k)

        T_lo = self._expiries[idx - 1]
        T_hi = self._expiries[idx]
        w_lo = self._slices[T_lo].total_var(k) * T_lo
        w_hi = self._slices[T_hi].total_var(k) * T_hi

        # Linear interpolation in *total variance × T* space
        alpha = (T - T_lo) / (T_hi - T_lo)
        wT = (1 - alpha) * w_lo + alpha * w_hi
        return np.sqrt(np.maximum(wT, 0.0) / T)

    def iv(self, K: float | np.ndarray, T: float) -> float | np.ndarray:
        """Implied vol from absolute strike(s) and expiry.

        Requires ``forward_curve`` to convert K → log-moneyness.
        """
        F = self._get_forward(T)
        k = np.log(np.asarray(K, dtype=float) / F)
        result = self.iv_from_logm(k, T)
        if result.ndim == 0:
            return float(result)
        return result


# ---------------------------------------------------------------------------
# SVI fitting
# ---------------------------------------------------------------------------
def fit_svi(
    strikes: np.ndarray,
    forward: float,
    expiry: float,
    market_ivs: np.ndarray,
    *,
    initial_guess: Optional[tuple] = None,
    bounds: Optional[tuple] = None,
) -> SVIParams:
    """Fit raw SVI to a single smile slice.

    Parameters
    ----------
    strikes : array-like, shape (N,)
        Absolute strike prices.
    forward : float
        Forward price for this expiry.
    expiry : float
        Time to expiry in years.
    market_ivs : array-like, shape (N,)
        Market implied volatilities (annualised).
    initial_guess : tuple, optional
        ``(a, b, rho, m, sigma)`` starting point for the solver.
    bounds : tuple, optional
        ``(lower, upper)`` each of length 5 for the solver.

    Returns
    -------
    SVIParams
        Fitted SVI slice.
    """
    from scipy.optimize import least_squares

    strikes = np.asarray(strikes, dtype=float)
    market_ivs = np.asarray(market_ivs, dtype=float)
    k = np.log(strikes / forward)                    # log-moneyness
    w_market = market_ivs ** 2 * expiry               # total variance

    if initial_guess is None:
        a0 = float(np.mean(w_market))
        initial_guess = (a0, 0.1, 0.0, 0.0, 0.1)

    if bounds is None:
        #        a      b     rho      m     sigma
        lower = (-0.5,  1e-6, -0.999, -2.0,  1e-4)
        upper = ( 2.0,  5.0,   0.999,  2.0,  5.0)
        bounds = (lower, upper)

    def residuals(params):
        a, b, rho, m, sig = params
        km = k - m
        w_model = a + b * (rho * km + np.sqrt(km * km + sig * sig))
        return w_model - w_market

    result = least_squares(
        residuals,
        x0=initial_guess,
        bounds=bounds,
        method="trf",
        max_nfev=2000,
    )

    a, b, rho, m, sig = result.x
    return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sig, expiry=expiry)


def fit_svi_surface(
    strikes_by_expiry: dict[float, np.ndarray],
    forwards: dict[float, float],
    market_ivs_by_expiry: dict[float, np.ndarray],
) -> VolSurface:
    """Fit SVI slice-by-slice and return a full ``VolSurface``.

    Parameters
    ----------
    strikes_by_expiry : dict[float, ndarray]
        ``{expiry: array_of_strikes}``.
    forwards : dict[float, float]
        ``{expiry: forward_price}``.
    market_ivs_by_expiry : dict[float, ndarray]
        ``{expiry: array_of_ivs}``.

    Returns
    -------
    VolSurface
        Calibrated surface with interpolation between slices.
    """
    slices: dict[float, SVIParams] = {}
    for T in sorted(strikes_by_expiry.keys()):
        slices[T] = fit_svi(
            strikes_by_expiry[T],
            forwards[T],
            T,
            market_ivs_by_expiry[T],
        )
    return VolSurface(slices, forward_curve=forwards)
