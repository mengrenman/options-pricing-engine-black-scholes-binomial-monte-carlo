"""Validated, bounded wrappers over optpricer for the web app.

Everything a request can reach goes through here, so the limits live in one
place and both the HTML and the JSON routes share them:

* Every size parameter is capped, so a single request cannot hang a worker
  (an unbounded ``n_paths`` is a one-line denial of service).
* Monte Carlo runs serially.  ``n_workers > 1`` starts a process pool that
  costs roughly 0.7-0.9 s per request on macOS and needs an
  ``if __name__ == "__main__"`` guard that web workers do not provide.
* Inputs have real lower bounds, not just "positive": a vanishing expiry or
  volatility drives the binomial tree's ``u - d`` to zero and the Greeks to
  infinity.
* Total volatility ``sigma * sqrt(T)`` is capped.  Past about 4 the
  finite-difference grid is off by tens of basis points and climbing, and at
  8 Monte Carlo is dozens of standard errors wrong while reporting a tiny
  standard error; at 10 it returns a price of exactly 0 with a standard error
  of exactly 0.
* Every number returned is checked for finiteness.

Library errors are re-raised as :class:`InputError`, which the routes turn
into HTTP 400 (JSON API) or an inline error message (HTMX).

Nothing here uses ``warnings.catch_warnings``.  It swaps process-wide state,
and FastAPI runs these functions on a thread pool, so concurrent requests
could corrupt warning handling for the whole server.  Notes shown on the page
are computed from the data instead; the library's own warnings go to the
server log.
"""

from __future__ import annotations

import json
import math

import numpy as np

import optpricer as op
from optpricer.calibration import (
    LOCAL_VOL_CAP,
    LOCAL_VOL_FLOOR,
    VolSurface,
    dupire_local_vol,
    fit_svi_surface,
)
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

MIN_PRICE = 1e-4        # spot, strike
MIN_EXPIRY = 1e-4       # years, about 53 minutes
MIN_VOL = 1e-3          # 0.1%
MAX_TOTAL_VOL = 4.0     # sigma * sqrt(T)

SURFACE_EXPIRIES = (0.1, 0.25, 0.5, 1.0, 2.0)
SURFACE_STRIKES = 21
SURFACE_LABEL = "web-synthetic"   # constant, so the library's warning logs once

MC_SEED = 12345   # fixed, so the MC price moves smoothly as inputs change


class InputError(ValueError):
    """A request the app refuses to compute, with a message safe to show."""


# ---------------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------------
def parse_float(raw, name: str) -> float:
    """Parse one field. Blank, non-numeric, non-finite and oversized are refused."""
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            raise InputError(f"{name} is required.")
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        raise InputError(f"{name} must be a number, got {raw!r}.") from None
    if not math.isfinite(value):
        raise InputError(f"{name} must be finite.")
    return value


def _check_range(value: float, name: str, lo: float, hi: float) -> float:
    if value < lo or value > hi:
        # repr, not :g -- 6 significant digits would print 30.0000001 as "30",
        # an in-range number, in the message explaining why it is out of range.
        raise InputError(f"{name} must be in [{lo:g}, {hi:g}], got {value!r}.")
    return value


def validate_option(S0, K, T, r, q, sigma, kind) -> tuple:
    S0 = _check_range(parse_float(S0, "Spot"), "Spot", MIN_PRICE, 1e6)
    K = _check_range(parse_float(K, "Strike"), "Strike", MIN_PRICE, 1e6)
    T = _check_range(parse_float(T, "Expiry"), "Expiry", MIN_EXPIRY, 30.0)
    r = _check_range(parse_float(r, "Rate"), "Rate", -0.5, 0.5)
    q = _check_range(parse_float(q, "Dividend yield"), "Dividend yield", -0.5, 0.5)
    sigma = _check_range(parse_float(sigma, "Volatility"), "Volatility", MIN_VOL, 5.0)
    total = sigma * math.sqrt(T)
    if total > MAX_TOTAL_VOL:
        raise InputError(
            f"Volatility x sqrt(expiry) is {total:.3g}. This page caps it at "
            f"{MAX_TOTAL_VOL:g}, past which the finite-difference and Monte Carlo "
            "engines stop being reliable."
        )
    kind = str(kind).strip().lower()
    if kind not in (op.CALL, op.PUT):
        raise InputError(f"Option type must be 'call' or 'put', got {kind!r}.")
    return S0, K, T, r, q, sigma, kind


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------
def _row(name: str, note: str, compute) -> dict:
    """One engine's row. A failing engine is marked unavailable, not fatal."""
    try:
        price = float(compute())
    except (ValueError, ArithmeticError) as exc:
        return {"name": name, "price": None, "note": note, "unavailable": str(exc)}
    if not math.isfinite(price):
        return {"name": name, "price": None, "note": note,
                "unavailable": "the engine returned a non-finite price"}
    return {"name": name, "price": price, "note": note}


def _binomial_row(opt, kind: str, american: bool) -> dict:
    name = "Binomial (American)" if american else "Binomial (European)"
    note = "early exercise allowed" if american else f"CRR, {BINOMIAL_STEPS} steps"
    # CRR needs its risk-neutral probability inside (0, 1), which is exactly
    # |r - q| * sqrt(T / N) < sigma. The library only says p is out of range.
    need = abs(opt.r - opt.q) * math.sqrt(opt.T / BINOMIAL_STEPS)
    if opt.sigma <= need:
        return {"name": name, "price": None, "note": note,
                "unavailable": (f"needs volatility above |r - q| x sqrt(T / "
                                f"{BINOMIAL_STEPS}) = {need:.4g}")}
    return _row(name, note, lambda: op.crr(opt, kind, N=BINOMIAL_STEPS, american=american))


def price_all(S0, K, T, r, q, sigma, kind, mc_paths=DEFAULT_MC_PATHS) -> dict:
    """Price one contract with every engine, plus Greeks and a spot ladder."""
    S0, K, T, r, q, sigma, kind = validate_option(S0, K, T, r, q, sigma, kind)
    mc_paths = int(_check_range(parse_float(mc_paths, "Monte Carlo paths"),
                                "Monte Carlo paths", 1_000, MAX_MC_PATHS))

    # Black-Scholes, its Greeks and the ladder anchor the panel: if they fail the
    # request fails. Every other engine may fail on its own without hiding them.
    try:
        opt = op.OptionSpec(S0=S0, K=K, T=T, r=r, sigma=sigma, q=q)
        bs = float(op.bs_price(opt, kind))
        greeks = {k: float(v) for k, v in op.bs_greeks(opt, kind).items()}
        spots = np.linspace(0.5 * S0, 1.5 * S0, 121)
        ladder_price = op.bs_price_vec(spots, K, T, r, q, sigma, kind)
        ladder_delta = op.bs_greeks_vec(spots, K, T, r, q, sigma, kind)["delta"]
    except (ValueError, ArithmeticError) as exc:
        raise InputError(str(exc)) from exc
    if not (math.isfinite(bs) and all(math.isfinite(v) for v in greeks.values())
            and np.all(np.isfinite(ladder_price)) and np.all(np.isfinite(ladder_delta))):
        raise InputError("These inputs produce a non-finite Black-Scholes price or Greek.")

    mc_note = f"{mc_paths:,} paths, antithetic + control variate"
    try:
        mc_px, mc_se = op.euro_price_mc(opt, kind, n_paths=mc_paths, seed=MC_SEED,
                                        n_workers=1)
        mc_px, mc_se = float(mc_px), float(mc_se)
    except (ValueError, ArithmeticError) as exc:
        mc = {"name": "Monte Carlo", "price": None, "note": mc_note, "unavailable": str(exc)}
    else:
        if math.isfinite(mc_px) and math.isfinite(mc_se):
            mc = {"name": "Monte Carlo", "price": mc_px, "stderr": mc_se, "note": mc_note}
        else:
            mc = {"name": "Monte Carlo", "price": None, "note": mc_note,
                  "unavailable": "the engine returned a non-finite price"}

    engines = [
        {"name": "Black-Scholes", "price": bs, "note": "closed form"},
        _binomial_row(opt, kind, american=False),
        _binomial_row(opt, kind, american=True),
        mc,
        _row("Finite difference", f"theta-scheme, {FD_GRID}x{FD_GRID}",
             lambda: fd_price(opt, kind, N_S=FD_GRID, N_t=FD_GRID)),
    ]
    for e in engines:
        e["diff_bps"] = ((e["price"] - bs) / bs * 1e4
                         if e["price"] is not None and bs > 1e-12 else None)

    return {
        "inputs": {"S0": S0, "K": K, "T": T, "r": r, "q": q, "sigma": sigma, "kind": kind},
        "engines": engines,
        "greeks": greeks,
        # Chart-only data, rounded: full float repr more than doubles the payload.
        "ladder": {"spot": np.round(spots, 6).tolist(),
                   "price": np.round(ladder_price, 6).tolist(),
                   "delta": np.round(ladder_delta, 6).tolist()},
    }


# ---------------------------------------------------------------------------
# Volatility surface
# ---------------------------------------------------------------------------
def synthetic_smile(k: np.ndarray, T: float, atm_vol, skew, curvature, term_slope) -> np.ndarray:
    """A parametric smile standing in for market quotes until real data exists."""
    return np.maximum(atm_vol + term_slope * math.sqrt(T) + skew * k + curvature * k * k, 0.02)


def fit_surface(S0, r, q, atm_vol, skew, curvature, term_slope) -> dict:
    S0 = _check_range(parse_float(S0, "Spot"), "Spot", MIN_PRICE, 1e6)
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
    except (ValueError, ArithmeticError) as exc:
        raise InputError(f"Calibration failed: {exc}") from exc
    surface.label = SURFACE_LABEL

    fit = []
    for T in SURFACE_EXPIRIES:
        k = np.log(strikes[T] / forwards[T])
        model = surface.iv_from_logm(k, T)
        fit.append({"expiry": T, "forward": forwards[T],
                    "rmse_bps": float(np.sqrt(np.mean((model - quotes[T]) ** 2)) * 1e4)})

    k_grid = np.linspace(-0.4, 0.4, 41)
    T_grid = np.linspace(SURFACE_EXPIRIES[0], SURFACE_EXPIRIES[-1], 30)
    iv_grid = np.array([surface.iv_from_logm(k_grid, T) for T in T_grid])
    if not np.all(np.isfinite(iv_grid)):
        raise InputError("The fitted surface produces non-finite implied volatility.")

    return {
        "surface": surface,
        "surface_json": surface.to_json(),
        "fit": fit,
        # Chart-only, rounded: 25.9 KB unrounded, 11.2 KB at these precisions.
        "grid": {"k": np.round(k_grid, 4).tolist(), "T": np.round(T_grid, 4).tolist(),
                 "iv": np.round(iv_grid, 5).tolist()},
    }


def load_surface(surface_json: str) -> VolSurface:
    """Rehydrate a surface posted back by the client.

    The client holds the surface, which keeps the server stateless -- but it
    also means this is untrusted input. Size is capped before parsing, and
    every way the parse or validation can fail becomes InputError: deeply
    nested JSON raises RecursionError, and a 310-digit integer used to raise
    OverflowError, neither of which is a ValueError.
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
    except (ValueError, TypeError, ArithmeticError, RecursionError) as exc:
        raise InputError(f"Invalid surface: {exc}") from exc


def local_vol_slice(surface_json, S0, t, r, q) -> dict:
    surface = load_surface(surface_json)
    S0 = _check_range(parse_float(S0, "Spot"), "Spot", MIN_PRICE, 1e6)
    t = _check_range(parse_float(t, "Time"), "Time", MIN_EXPIRY, 30.0)
    r = _check_range(parse_float(r, "Rate"), "Rate", -0.5, 0.5)
    q = _check_range(parse_float(q, "Dividend yield"), "Dividend yield", -0.5, 0.5)

    spots = np.linspace(0.6 * S0, 1.4 * S0, 81)
    try:
        lv = np.atleast_1d(dupire_local_vol(surface, spots, t, r, q))
        w = surface.total_var_from_logm(np.log(spots / surface.forward_at(t, r, q)), t)
        implied = np.sqrt(np.maximum(w, 0.0) / t)
    except (ValueError, ArithmeticError) as exc:
        raise InputError(str(exc)) from exc
    if not (np.all(np.isfinite(lv)) and np.all(np.isfinite(implied))):
        raise InputError(f"The surface yields non-finite volatility at t={t:g}.")

    notes = []
    if np.any(w <= 0.0):
        notes.append("The surface has non-positive total variance at some of these "
                     "nodes; it is arbitrage-violating or over-extrapolated there.")
    n_cap = int(np.count_nonzero(lv >= LOCAL_VOL_CAP * (1 - 1e-9)))
    n_floor = int(np.count_nonzero(lv <= LOCAL_VOL_FLOOR * (1 + 1e-9)))
    if n_cap or n_floor:
        notes.append(f"{n_cap + n_floor} of {lv.size} points sit at the library's clip "
                     f"bounds ({LOCAL_VOL_FLOOR} or {LOCAL_VOL_CAP}); those are clamped, "
                     "not computed.")
    return {"t": t, "spot": spots.tolist(), "local_vol": lv.tolist(),
            "implied_vol": implied.tolist(), "warnings": notes,
            "clipped": n_cap + n_floor,
            "extrapolated": bool(t < surface.expiries[0] or t > surface.expiries[-1])}
