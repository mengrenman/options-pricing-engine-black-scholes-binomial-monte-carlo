"""Validated, bounded wrappers over optpricer for the web app.

Everything a request can reach goes through here, so the limits live in one
place and both the HTML and the JSON routes share them:

* Every size parameter is capped, so a single request cannot hang a worker
  (an unbounded ``n_paths`` is a one-line denial of service).
* Monte Carlo runs serially.  ``n_workers > 1`` starts a process pool that
  costs roughly 0.7-0.9 s per request on macOS and needs an
  ``if __name__ == "__main__"`` guard that web workers do not provide.
* Expiry must be strictly positive.  ``bs_price_vec`` returns NaN at T=0
  rather than raising, so the check has to happen before the library sees it.

Library ``ValueError``s are re-raised as :class:`InputError`, which the routes
turn into HTTP 400 (JSON API) or an inline error message (HTMX).
"""

from __future__ import annotations

import json
import math
import warnings

import numpy as np

import optpricer as op
from optpricer.calibration import VolSurface, dupire_local_vol, fit_svi_surface
from optpricer.pde import fd_price

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------
MAX_MC_PATHS = 2_000_000
DEFAULT_MC_PATHS = 200_000
BINOMIAL_STEPS = 500
FD_GRID = 400
MAX_SURFACE_JSON_BYTES = 200_000
MAX_SURFACE_SLICES = 200

SURFACE_EXPIRIES = (0.1, 0.25, 0.5, 1.0, 2.0)
SURFACE_STRIKES = 21

MC_SEED = 12345   # fixed, so the MC price moves smoothly as inputs change

# dupire_local_vol clips its output to [0.01, 5.0] without warning.  Points at
# either bound are clamped, not computed, so the page must say so.
LOCAL_VOL_FLOOR = 0.01
LOCAL_VOL_CAP = 5.0


class InputError(ValueError):
    """A request the app refuses to compute, with a message safe to show."""


# ---------------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------------
def parse_float(raw, name: str) -> float:
    """Parse one form field. Blank, non-numeric and non-finite are all refused."""
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            raise InputError(f"{name} is required.")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise InputError(f"{name} must be a number, got {raw!r}.") from None
    if not math.isfinite(value):
        raise InputError(f"{name} must be finite.")
    return value


def _check_range(value: float, name: str, lo: float, hi: float,
                 *, lo_open: bool = False) -> float:
    below = value <= lo if lo_open else value < lo
    if below or value > hi:
        bracket = "(" if lo_open else "["
        raise InputError(f"{name} must be in {bracket}{lo:g}, {hi:g}], got {value:g}.")
    return value


def validate_option(S0, K, T, r, q, sigma, kind) -> tuple:
    S0 = _check_range(parse_float(S0, "Spot"), "Spot", 0.0, 1e6, lo_open=True)
    K = _check_range(parse_float(K, "Strike"), "Strike", 0.0, 1e6, lo_open=True)
    T = _check_range(parse_float(T, "Expiry"), "Expiry", 0.0, 30.0, lo_open=True)
    r = _check_range(parse_float(r, "Rate"), "Rate", -0.5, 0.5)
    q = _check_range(parse_float(q, "Dividend yield"), "Dividend yield", -0.5, 0.5)
    sigma = _check_range(parse_float(sigma, "Volatility"), "Volatility", 0.0, 5.0,
                         lo_open=True)
    kind = str(kind).strip().lower()
    if kind not in (op.CALL, op.PUT):
        raise InputError(f"Option type must be 'call' or 'put', got {kind!r}.")
    return S0, K, T, r, q, sigma, kind


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------
def price_all(S0, K, T, r, q, sigma, kind, mc_paths=DEFAULT_MC_PATHS) -> dict:
    """Price one contract with every engine, plus Greeks and a spot ladder."""
    S0, K, T, r, q, sigma, kind = validate_option(S0, K, T, r, q, sigma, kind)
    mc_paths = int(_check_range(parse_float(mc_paths, "Monte Carlo paths"),
                                "Monte Carlo paths", 1_000, MAX_MC_PATHS))

    try:
        opt = op.OptionSpec(S0=S0, K=K, T=T, r=r, sigma=sigma, q=q)
        bs = op.bs_price(opt, kind)
        mc_px, mc_se = op.euro_price_mc(opt, kind, n_paths=mc_paths, seed=MC_SEED,
                                        n_workers=1)
        engines = [
            {"name": "Black-Scholes", "price": bs, "note": "closed form"},
            {"name": "Binomial (European)",
             "price": op.crr(opt, kind, N=BINOMIAL_STEPS, american=False),
             "note": f"CRR, {BINOMIAL_STEPS} steps"},
            {"name": "Binomial (American)",
             "price": op.crr(opt, kind, N=BINOMIAL_STEPS, american=True),
             "note": "early exercise allowed"},
            {"name": "Monte Carlo", "price": mc_px, "stderr": mc_se,
             "note": f"{mc_paths:,} paths, antithetic + control variate"},
            {"name": "Finite difference", "price": fd_price(opt, kind, N_S=FD_GRID, N_t=FD_GRID),
             "note": f"theta-scheme, {FD_GRID}x{FD_GRID}"},
        ]
        greeks = op.bs_greeks(opt, kind)

        spots = np.linspace(0.5 * S0, 1.5 * S0, 121)
        ladder_price = op.bs_price_vec(spots, K, T, r, q, sigma, kind)
        ladder_delta = op.bs_greeks_vec(spots, K, T, r, q, sigma, kind)["delta"]
    except ValueError as exc:
        raise InputError(str(exc)) from exc

    for e in engines:
        e["diff_bps"] = (e["price"] - bs) / bs * 1e4 if bs > 1e-12 else None

    return {
        "inputs": {"S0": S0, "K": K, "T": T, "r": r, "q": q, "sigma": sigma, "kind": kind},
        "engines": engines,
        "greeks": {k: float(v) for k, v in greeks.items()},
        "ladder": {"spot": spots.tolist(), "price": ladder_price.tolist(),
                   "delta": ladder_delta.tolist()},
    }


# ---------------------------------------------------------------------------
# Volatility surface
# ---------------------------------------------------------------------------
def synthetic_smile(k: np.ndarray, T: float, atm_vol, skew, curvature, term_slope) -> np.ndarray:
    """A parametric smile standing in for market quotes until real data exists."""
    return np.maximum(atm_vol + term_slope * math.sqrt(T) + skew * k + curvature * k * k, 0.02)


def fit_surface(S0, r, q, atm_vol, skew, curvature, term_slope) -> dict:
    S0 = _check_range(parse_float(S0, "Spot"), "Spot", 0.0, 1e6, lo_open=True)
    r = _check_range(parse_float(r, "Rate"), "Rate", -0.5, 0.5)
    q = _check_range(parse_float(q, "Dividend yield"), "Dividend yield", -0.5, 0.5)
    atm_vol = _check_range(parse_float(atm_vol, "ATM vol"), "ATM vol", 0.01, 2.0)
    skew = _check_range(parse_float(skew, "Skew"), "Skew", -2.0, 2.0)
    curvature = _check_range(parse_float(curvature, "Curvature"), "Curvature", 0.0, 5.0)
    term_slope = _check_range(parse_float(term_slope, "Term slope"), "Term slope", -0.5, 0.5)

    strikes, forwards, quotes = {}, {}, {}
    for T in SURFACE_EXPIRIES:
        F = S0 * math.exp((r - q) * T)
        K = np.linspace(0.6 * F, 1.4 * F, SURFACE_STRIKES)
        strikes[T], forwards[T] = K, F
        quotes[T] = synthetic_smile(np.log(K / F), T, atm_vol, skew, curvature, term_slope)

    try:
        surface = fit_svi_surface(strikes, forwards, quotes, method="quasi")
    except ValueError as exc:
        raise InputError(f"Calibration failed: {exc}") from exc

    fit = []
    for T in SURFACE_EXPIRIES:
        k = np.log(strikes[T] / forwards[T])
        model = surface.iv_from_logm(k, T)
        fit.append({"expiry": T, "forward": forwards[T],
                    "rmse_bps": float(np.sqrt(np.mean((model - quotes[T]) ** 2)) * 1e4)})

    k_grid = np.linspace(-0.4, 0.4, 41)
    T_grid = np.linspace(SURFACE_EXPIRIES[0], SURFACE_EXPIRIES[-1], 30)
    iv_grid = np.array([surface.iv_from_logm(k_grid, T) for T in T_grid])

    return {
        "surface": surface,
        "surface_json": surface.to_json(),
        "fit": fit,
        "grid": {"k": k_grid.tolist(), "T": T_grid.tolist(), "iv": iv_grid.tolist()},
    }


def load_surface(surface_json: str) -> VolSurface:
    """Rehydrate a surface posted back by the client.

    The client holds the surface, which keeps the server stateless -- but it
    also means this is untrusted input, so size is capped before parsing and
    VolSurface.from_json does the strict validation it was built for.
    """
    if not isinstance(surface_json, str) or not surface_json.strip():
        raise InputError("No surface supplied. Fit one first.")
    if len(surface_json.encode()) > MAX_SURFACE_JSON_BYTES:
        raise InputError(f"Surface payload exceeds {MAX_SURFACE_JSON_BYTES:,} bytes.")
    try:
        head = json.loads(surface_json)
        if isinstance(head, dict) and len(head.get("slices") or []) > MAX_SURFACE_SLICES:
            raise InputError(f"Surface has more than {MAX_SURFACE_SLICES} slices.")
        return VolSurface.from_json(surface_json)
    except InputError:
        raise
    except (ValueError, TypeError) as exc:
        raise InputError(f"Invalid surface: {exc}") from exc


def local_vol_slice(surface_json, S0, t, r, q) -> dict:
    surface = load_surface(surface_json)
    S0 = _check_range(parse_float(S0, "Spot"), "Spot", 0.0, 1e6, lo_open=True)
    t = _check_range(parse_float(t, "Time"), "Time", 0.0, 30.0, lo_open=True)
    r = _check_range(parse_float(r, "Rate"), "Rate", -0.5, 0.5)
    q = _check_range(parse_float(q, "Dividend yield"), "Dividend yield", -0.5, 0.5)

    spots = np.linspace(0.6 * S0, 1.4 * S0, 81)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            lv = np.atleast_1d(dupire_local_vol(surface, spots, t, r, q))
        except ValueError as exc:
            raise InputError(str(exc)) from exc
    notes = sorted({str(w.message) for w in caught})
    n_cap = int(np.count_nonzero(lv >= LOCAL_VOL_CAP * (1 - 1e-9)))
    n_floor = int(np.count_nonzero(lv <= LOCAL_VOL_FLOOR * (1 + 1e-9)))
    if n_cap or n_floor:
        notes.append(
            f"{n_cap + n_floor} of {lv.size} points sit at the library's clip bounds "
            f"({LOCAL_VOL_FLOOR} or {LOCAL_VOL_CAP}); those are clamped, not computed."
        )
    implied = np.sqrt(surface.total_var_from_logm(
        np.log(spots / surface.forward_at(t, r, q)), t) / t)
    return {"t": t, "spot": spots.tolist(), "local_vol": lv.tolist(),
            "implied_vol": implied.tolist(), "warnings": notes,
            "clipped": n_cap + n_floor,
            "extrapolated": bool(t < surface.expiries[0] or t > surface.expiries[-1])}
