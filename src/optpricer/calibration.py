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

    def w_dw_d2w(self, k: np.ndarray | float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Total variance and its first two k-derivatives in a single pass.

        ``total_var``, ``dw_dk`` and ``d2w_dk2`` each recompute ``k - m`` and
        the same square root.  Callers that need all three (notably
        :func:`dupire_local_vol`) should use this instead.
        """
        k = np.asarray(k, dtype=float)
        u = k - self.m
        s2 = self.sigma * self.sigma
        root = np.sqrt(u * u + s2)
        w = self.a + self.b * (self.rho * u + root)
        dw = self.b * (self.rho + u / root)
        d2w = self.b * s2 / (root * root * root)
        return w, dw, d2w

    def iv(self, k: np.ndarray | float) -> np.ndarray:
        """Return implied volatility from log-moneyness."""
        w = self.total_var(k)
        return np.sqrt(np.maximum(w, 0.0) / self.expiry)

    def dw_dk(self, k: np.ndarray | float) -> np.ndarray:
        """First derivative of total variance w.r.t. log-moneyness.

        dw/dk = b * (rho + (k - m) / sqrt((k - m)^2 + sigma^2))
        """
        k = np.asarray(k, dtype=float)
        u = k - self.m
        return self.b * (self.rho + u / np.sqrt(u * u + self.sigma ** 2))

    def d2w_dk2(self, k: np.ndarray | float) -> np.ndarray:
        """Second derivative of total variance w.r.t. log-moneyness.

        d^2w/dk^2 = b * sigma^2 / ((k - m)^2 + sigma^2)^{3/2}
        """
        k = np.asarray(k, dtype=float)
        u = k - self.m
        return self.b * self.sigma ** 2 / (u * u + self.sigma ** 2) ** 1.5


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
    def slices(self) -> dict[float, 'SVIParams']:
        """Mapping ``{expiry: SVIParams}`` (read-only copy)."""
        return dict(self._slices)

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

    def total_var_from_logm(self, k: np.ndarray | float, T: float) -> np.ndarray:
        """Total variance w(k, T), interpolated the same way as ``iv_from_logm``.

        ``iv_from_logm`` returns ``sqrt(w / T)``; callers that immediately
        square it back (the Dupire dw/dT bump) should use this and skip the
        round trip.
        """
        k = np.asarray(k, dtype=float)

        def _from_slice(sl):
            # ``iv_from_logm`` divides by the *slice* expiry in these branches,
            # and the caller multiplies by T, so carry that ratio through.
            return np.maximum(sl.total_var(k), 0.0) * (T / sl.expiry)

        if T in self._slices:
            return _from_slice(self._slices[T])

        idx = np.searchsorted(self._expiries, T)
        if idx == 0:
            return _from_slice(self._slices[self._expiries[0]])
        if idx >= len(self._expiries):
            return _from_slice(self._slices[self._expiries[-1]])

        T_lo = self._expiries[idx - 1]
        T_hi = self._expiries[idx]
        w_lo = self._slices[T_lo].total_var(k) * T_lo
        w_hi = self._slices[T_hi].total_var(k) * T_hi
        alpha = (T - T_lo) / (T_hi - T_lo)
        return np.maximum((1 - alpha) * w_lo + alpha * w_hi, 0.0)

    def _w_slope_on_piece(self, k: np.ndarray, idx: int) -> np.ndarray:
        """Slope in T of the linear piece selected by ``idx``.

        ``idx`` is a ``searchsorted`` index into ``_expiries``: 0 or past the
        end selects the single-slice scaling branch of
        ``total_var_from_logm``, otherwise the interval
        ``(_expiries[idx-1], _expiries[idx])``.
        """
        exp = self._expiries
        if idx <= 0:
            sl = self._slices[exp[0]]
            return np.maximum(sl.total_var(k), 0.0) / sl.expiry
        if idx >= len(exp):
            sl = self._slices[exp[-1]]
            return np.maximum(sl.total_var(k), 0.0) / sl.expiry
        T_lo, T_hi = exp[idx - 1], exp[idx]
        w_lo = self._slices[T_lo].total_var(k) * T_lo
        w_hi = self._slices[T_hi].total_var(k) * T_hi
        return (w_hi - w_lo) / (T_hi - T_lo)

    def dw_dT_from_logm(self, k: np.ndarray | float, T: float) -> np.ndarray:
        """∂w/∂T of the interpolated total-variance surface, in closed form.

        ``total_var_from_logm`` is piecewise linear in ``T`` -- between two
        slices it interpolates ``w * T`` linearly, and outside the quoted
        range it scales one slice by ``T / T_slice``.  Both pieces are linear,
        so no finite-difference bump is needed.

        The surface has a kink at every quoted expiry, where the one-sided
        derivatives differ.  There the two are averaged, which is what a
        centred bump straddling the kink returned.
        """
        k = np.asarray(k, dtype=float)
        exp = self._expiries
        idx_left = int(np.searchsorted(exp, T, side="left"))    # piece below T
        idx_right = int(np.searchsorted(exp, T, side="right"))  # piece above T
        if idx_left == idx_right:                               # away from a kink
            return self._w_slope_on_piece(k, idx_right)
        return 0.5 * (self._w_slope_on_piece(k, idx_left)
                      + self._w_slope_on_piece(k, idx_right))

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


# ---------------------------------------------------------------------------
# Quasi-explicit SVI calibration
# ---------------------------------------------------------------------------
# Raw SVI is linear in three of its five parameters once the shape parameters
# are held fixed.  Substituting y = (k - m) / sigma and s = sqrt(y**2 + 1),
#
#     w = a + d * y + c * s      with   c = b * sigma,  d = rho * b * sigma
#
# so for fixed (m, sigma) the remaining fit is an ordinary linear least
# squares.  That turns the 5-D nonlinear problem into a 2-D search wrapped
# around an exact inner solve.
#
# Validity needs b > 0 and |rho| <= 1, i.e. c >= 0 and |d| <= c, which couple
# two unknowns.  Substituting u = c + d and v = c - d,
#
#     w = a + u * (s + y) / 2 + v * (s - y) / 2
#
# turns them into plain non-negativity, so a bounded linear least squares
# solves the inner problem exactly.  Writing d = rho_cap * (u - v) / 2 instead
# carries the |rho| <= rho_cap bound through the same substitution, so the cap
# is honoured by the solve rather than clamped on afterwards -- clamping a
# finished fit rescales rho while leaving the other parameters stale, which
# badly degrades slices whose optimum sits on the boundary.
#
# Note `a` keeps a negative lower bound, matching ``fit_svi``.  Forcing a >= 0
# (as some published formulations do) measurably degrades the fit here.

_QE_A_LOWER = -0.5
_QE_RHO_CAP = 0.999      # same bound ``fit_svi`` uses
_QE_STARTS = ((None, 0.10), (0.0, 0.30), (None, 0.50))


def _svi_inner_ls(k: np.ndarray, w: np.ndarray, m: float, sigma: float,
                  a_hi: float) -> tuple[float, tuple[float, float, float]]:
    """Exact bounded linear least squares in (a, u, v) for fixed (m, sigma).

    Returns ``(sum_of_squares, (a, d, c))``.
    """
    from scipy.optimize import lsq_linear

    y = (k - m) / sigma
    s = np.sqrt(y * y + 1.0)
    yc = _QE_RHO_CAP * y                      # keeps |rho| <= _QE_RHO_CAP
    A = np.empty((k.size, 3))
    A[:, 0] = 1.0
    A[:, 1] = (s + yc) * 0.5
    A[:, 2] = (s - yc) * 0.5

    res = lsq_linear(
        A, w,
        bounds=([_QE_A_LOWER, 0.0, 0.0], [a_hi, 8.0 * sigma, 8.0 * sigma]),
        method="bvls", lsq_solver="exact",
    )
    a, u, v = res.x
    c = 0.5 * (u + v)
    d = _QE_RHO_CAP * 0.5 * (u - v)
    return float(np.sum((A @ res.x - w) ** 2)), (a, d, c)


def fit_svi_quasi(
    strikes: np.ndarray,
    forward: float,
    expiry: float,
    market_ivs: np.ndarray,
) -> SVIParams:
    """Fit raw SVI via the quasi-explicit reduction.

    A drop-in alternative to :func:`fit_svi`.  Matches its accuracy on
    benchmark slices while running roughly twice as fast, because only two
    parameters are searched numerically.

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

    Returns
    -------
    SVIParams
        Fitted SVI slice.  ``rho`` is bounded by +/-0.999 inside the solve,
        so a degenerate linear wing cannot be returned.
    """
    from scipy.optimize import minimize

    strikes = np.asarray(strikes, dtype=float)
    market_ivs = np.asarray(market_ivs, dtype=float)
    k = np.log(strikes / forward)
    w = market_ivs ** 2 * expiry

    a_hi = float(np.max(w))
    k_min = float(k[np.argmin(w)])

    # Multi-start matters: a single start is faster but misses the best
    # optimum on a small fraction of slices.
    best = (np.inf, k_min, 0.10)
    for m0, s0 in _QE_STARTS:
        m_start = k_min if m0 is None else m0
        res = minimize(
            lambda x: _svi_inner_ls(k, w, x[0], abs(x[1]) + 1e-8, a_hi)[0],
            [m_start, s0], method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-14, "maxiter": 400},
        )
        if res.fun < best[0]:
            best = (float(res.fun), float(res.x[0]), abs(float(res.x[1])) + 1e-8)

    _, m, sigma = best
    _, (a, d, c) = _svi_inner_ls(k, w, m, sigma, a_hi)

    b = c / sigma
    rho = float(d / c) if c > 1e-14 else 0.0   # already within +/-_QE_RHO_CAP
    return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma, expiry=expiry)


def fit_svi_surface(
    strikes_by_expiry: dict[float, np.ndarray],
    forwards: dict[float, float],
    market_ivs_by_expiry: dict[float, np.ndarray],
    *,
    method: str = "trf",
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
    method : str
        ``"trf"`` (default) fits each slice with :func:`fit_svi`;
        ``"quasi"`` uses :func:`fit_svi_quasi`, which is about twice as fast
        at matching accuracy.

    Returns
    -------
    VolSurface
        Calibrated surface with interpolation between slices.
    """
    if method not in ("trf", "quasi"):
        raise ValueError(f"method must be 'trf' or 'quasi', got {method!r}")
    slice_fit = fit_svi if method == "trf" else fit_svi_quasi

    slices: dict[float, SVIParams] = {}
    for T in sorted(strikes_by_expiry.keys()):
        slices[T] = slice_fit(
            strikes_by_expiry[T],
            forwards[T],
            T,
            market_ivs_by_expiry[T],
        )
    return VolSurface(slices, forward_curve=forwards)


# ---------------------------------------------------------------------------
# Dupire local volatility
# ---------------------------------------------------------------------------

def dupire_local_vol(
    surface: VolSurface,
    S: float | np.ndarray,
    t: float,
    r: float,
    q: float,
    *,
    dT: float = 1e-4,
) -> float | np.ndarray:
    """Compute Dupire local volatility at ``(S, t)`` from a calibrated surface.

    Uses Dupire's formula in total-variance / log-moneyness coordinates:

    .. math::

        \\sigma_{\\mathrm{loc}}^2(K,T) =
        \\frac{\\partial w / \\partial T}
             {1 - \\frac{y}{w}\\frac{\\partial w}{\\partial y}
              + \\frac{1}{4}\\left(-\\frac{1}{4} - \\frac{1}{w}
              + \\frac{y^2}{w^2}\\right)\\left(\\frac{\\partial w}{\\partial y}\\right)^2
              + \\frac{1}{2}\\frac{\\partial^2 w}{\\partial y^2}}

    where ``w = IV² T`` and ``y = ln(K/F)``.

    Here *S* plays the role of *K* (strike) at the local-vol evaluation
    point, and the forward ``F(t) = S0 exp((r-q) t)`` is derived from the
    surface's forward curve or approximated.

    Parameters
    ----------
    surface : VolSurface
    S : float or array
        Spot/strike value(s) at which to evaluate local vol.
    t : float
        Time point.
    r, q : float
        Risk-free rate and dividend yield.
    dT : float
        Unused.  ∂w/∂T is now computed in closed form; the parameter is kept
        so existing callers keep working.

    Returns
    -------
    float or ndarray
        Local volatility σ_loc(S, t).
    """
    S_arr = np.asarray(S, dtype=float)
    t = max(t, 1e-8)  # avoid t = 0

    # Forward at time t — use surface's forward curve if available
    try:
        F = surface._get_forward(t)
    except (ValueError, KeyError):
        F = float(S_arr.mean()) if S_arr.ndim > 0 else float(S_arr)

    k = np.log(S_arr / F)

    # Find the SVI slice closest to t for analytical derivatives
    exp_arr = surface._expiries
    idx = int(np.searchsorted(exp_arr, t))
    idx = max(0, min(idx, len(exp_arr) - 1))
    T_near = exp_arr[idx]
    svi_slice = surface._slices[T_near]

    # w and its spatial derivatives (analytical from SVI), one pass
    w, dw, d2w = svi_slice.w_dw_d2w(k)
    w = np.maximum(w, 1e-12)

    # ∂w/∂T in closed form.  The interpolated surface is piecewise linear in
    # T, so a finite-difference bump only added error and two extra passes
    # over the strike array.
    dwdT = surface.dw_dT_from_logm(k, t)

    # Dupire's formula
    numer = np.maximum(dwdT, 1e-12)
    denom = (1.0
             - (k / w) * dw
             + 0.25 * (-0.25 - 1.0 / w + (k / w) ** 2) * dw ** 2
             + 0.5 * d2w)
    denom = np.maximum(denom, 1e-8)  # prevent negative / zero

    sigma_loc_sq = numer / denom
    sigma_loc = np.sqrt(np.maximum(sigma_loc_sq, 0.0))
    sigma_loc = np.clip(sigma_loc, 0.01, 5.0)

    if sigma_loc.ndim == 0:
        return float(sigma_loc)
    return sigma_loc


def dupire_local_vol_func(
    surface: VolSurface,
    r: float,
    q: float,
) -> 'Callable[[np.ndarray, float], np.ndarray]':
    """Return a callable ``sigma_loc(S_array, t) -> sigma_array``.

    The returned function is compatible with:

    - :func:`local_vol_paths` from ``processes.py``
    - :func:`fd_price_local_vol` from ``pde.py``
    - :func:`milstein_local_vol_paths` from ``processes.py``

    Parameters
    ----------
    surface : VolSurface
        Calibrated implied-vol surface.
    r, q : float
        Risk-free rate and dividend yield.

    Returns
    -------
    callable
    """
    def _sigma_loc(S_arr: np.ndarray, t: float) -> np.ndarray:
        result = dupire_local_vol(surface, S_arr, t, r, q)
        return np.asarray(result, dtype=float)

    return _sigma_loc
