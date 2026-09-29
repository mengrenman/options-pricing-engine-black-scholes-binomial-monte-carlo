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
* Every finite-difference and finite-element solve uses one grid rule
  (:func:`fd_grid`), which widens the range for far strikes and large drift.
* Every pricing view settles its volatility in one place
  (:func:`resolve_option`): given directly, or solved from a market price.

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
from dataclasses import replace
from typing import NamedTuple

import numpy as np

import optpricer as op
from optpricer.calibration import (
    LOCAL_VOL_CAP,
    LOCAL_VOL_FLOOR,
    VolSurface,
    dupire_local_vol,
    fit_svi_surface,
)
from optpricer.fem import fem_price
from optpricer.pde import fd_greeks, fd_price

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

# Pricing pane: finite-difference grid range, American vega and rho, scenario grid.
FD_BASE_MULT = 4.0      # grid half-width in sigma*sqrt(T), before any widening
FD_MARGIN = 3.0         # widened grids reach the strike or drift plus this many sigma*sqrt(T)
FD_MAX_N_S = 800        # cap on the spatial intervals when the grid is widened
AMERICAN_VEGA_BUMP = 0.01       # cap on the vega difference's half-width in volatility
AMERICAN_VEGA_REL_BUMP = 0.02   # half-width as a share of sigma, so low vols are not over-bumped
AMERICAN_RHO_BUMP = 0.001       # central-difference half-width in the rate (absolute)
FD_BUMP_N_T = 200               # time steps of the bumped solves behind the American vega and rho
AMERICAN_MAX_TOTAL_VOL = 2.5    # above this the grid's own error puts American vega/rho past 1%
AMERICAN_ZERO_RATE_BAND = 0.03  # |r| T under this, with total vol above the next, is a zero rate
AMERICAN_ZERO_RATE_MAX_TOTAL_VOL = 1.0   # for rho near a zero rate: past either of these
AMERICAN_ZERO_RATE_MAX_SIGMA = 1.0       # limits the spurious premium moves rho by over 1%
MC_SE_FLOOR = 1e-7              # a standard error below this fraction of the price carries no z
LOW_VEGA = 0.01                 # below this, a price error of 1e-4 moves the solved vol by a point
SCENARIO_SPOT_SHOCKS = tuple(np.round(np.linspace(0.7, 1.3, 13), 4).tolist())   # multiplicative
SCENARIO_VOL_SHOCKS = tuple(np.round(np.linspace(-0.10, 0.10, 9), 4).tolist())  # additive
SCENARIO_RATE_SHOCKS = (-0.02, -0.01, 0.0, 0.01, 0.02)                          # additive
SCENARIO_NOISE = 1e-8    # a P&L below this fraction of max(spot, strike) is shown as zero


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


def _ladder(S0, K, T, r, q, sigma, kind) -> dict:
    """Black-Scholes price and delta across spot, 50% to 150% of S0, rounded for the chart.
    Raises InputError if any value is not finite."""
    spots = np.linspace(0.5 * S0, 1.5 * S0, 121)
    try:
        price = op.bs_price_vec(spots, K, T, r, q, sigma, kind)
        delta = op.bs_greeks_vec(spots, K, T, r, q, sigma, kind)["delta"]
    except (ValueError, ArithmeticError) as exc:
        raise InputError(str(exc)) from exc
    if not (np.all(np.isfinite(price)) and np.all(np.isfinite(delta))):
        raise InputError("These inputs produce a non-finite Black-Scholes price or Greek.")
    # Full float repr more than doubles the payload.
    return {"spot": np.round(spots, 6).tolist(), "price": np.round(price, 6).tolist(),
            "delta": np.round(delta, 6).tolist()}


class Contract(NamedTuple):
    """One validated contract. ``iv`` is None when the volatility was given, and the
    record of the solve (``market_price``, ``sigma``, ``vega``, ``weak``) when it was
    solved from a market price."""
    S0: float
    K: float
    T: float
    r: float
    q: float
    sigma: float
    kind: str
    iv: dict | None = None


def vol_source(solve, field) -> tuple:
    """Split the pricing form's volatility field by its "Solve for" mode.

    In "price" mode the field holds a volatility; in "iv" mode it holds the market
    price to solve a volatility from. Returns ``(sigma, market_price)``, one of them
    None, ready for :func:`resolve_option`.
    """
    mode = str(solve).strip().lower() or "price"
    if mode == "price":
        return field, None
    if mode == "iv":
        return None, field
    raise InputError(f"Solve for must be 'price' or 'iv', got {solve!r}.")


def _vol_bracket(T: float) -> tuple[float, float]:
    """The volatilities the app prices: MIN_VOL up to min(5, MAX_TOTAL_VOL / sqrt(T))."""
    root_T = math.sqrt(T)
    hi = min(5.0, MAX_TOTAL_VOL / root_T)
    while hi * root_T > MAX_TOTAL_VOL:      # the division can round the wrong way by an ulp
        hi = math.nextafter(hi, 0.0)
    return MIN_VOL, hi


def _distinct(a: float, b: float) -> tuple[str, str]:
    """Format two numbers with as few significant digits (at least 6) as tell them apart,
    so "price X is below bound X" cannot happen when they differ in the seventh digit."""
    for digits in range(6, 18):
        fa, fb = f"{a:.{digits}g}", f"{b:.{digits}g}"
        if fa != fb:
            return fa, fb
    return repr(a), repr(b)


def solve_implied_vol(S0, K, T, r, q, price, kind) -> dict:
    """Solve the Black-Scholes volatility that prices a call or put at ``price``.

    The library's scalar Brent solver does the root find, after the price is checked
    against the bracket ends by evaluating Black-Scholes there. The library's own
    bracket widening ignores the bracket it is given, and its vectorized solver returns
    NaN on many out-of-the-money prices, so neither is used. Inputs are already
    validated. Raises :class:`InputError` with a specific message when no volatility
    the app can price reproduces ``price``.
    """
    lo, hi = _vol_bracket(T)

    def bs_at(sigma):
        return float(op.bs_price(op.OptionSpec(S0=S0, K=K, T=T, r=r, sigma=sigma, q=q), kind))

    try:
        f_lo, f_hi = bs_at(lo), bs_at(hi)
    except (ValueError, ArithmeticError) as exc:
        raise InputError(str(exc)) from exc
    fwd, strike = S0 * math.exp(-q * T), K * math.exp(-r * T)
    lower = max(fwd - strike, 0.0) if kind == op.CALL else max(strike - fwd, 0.0)

    if price < lower:
        p, b = _distinct(price, lower)
        raise InputError(f"Market price {p} is below the no-arbitrage lower bound (the "
                         f"discounted intrinsic value S e^(-qT) - K e^(-rT) = {b}), so no "
                         "volatility prices it.")
    if price <= lower:
        raise InputError(f"Market price {price:.6g} has no time value: it equals the "
                         f"no-arbitrage lower bound, which Black-Scholes only reaches as the "
                         "volatility goes to zero.")
    if price <= f_lo:
        p, b = _distinct(price, f_lo)
        raise InputError(f"Market price {p} implies a volatility below this page's minimum of "
                         f"{lo:.1%} (Black-Scholes gives {b} there).")
    if price >= f_hi:
        p, b = _distinct(price, f_hi)
        raise InputError(f"Market price {p} is above the Black-Scholes price of {b} at this "
                         f"page's maximum volatility of {hi:.3g}.")
    try:
        sigma = op.implied_vol(op.OptionSpec(S0=S0, K=K, T=T, r=r, sigma=lo, q=q), price, kind,
                               tol=1e-12, bracket=(lo, hi))
        sigma = min(max(float(sigma), lo), hi)
        vega = float(op.bs_greeks(op.OptionSpec(S0=S0, K=K, T=T, r=r, sigma=sigma, q=q),
                                  kind)["vega"])
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        raise InputError(f"The volatility solver failed: {exc}") from exc
    if not (math.isfinite(sigma) and math.isfinite(vega)):
        raise InputError("The volatility solver returned a non-finite value.")
    return {"market_price": price, "sigma": sigma, "vega": vega, "weak": vega < LOW_VEGA}


def resolve_option(S0, K, T, r, q, sigma, kind, market_price=None) -> Contract:
    """Validate one contract and settle its volatility. Every pricing view calls this,
    so in implied-vol mode they all use the same solved volatility.

    Give ``sigma``, or ``market_price`` to have the volatility solved from it; not both.
    """
    if market_price is None:
        if sigma is None:
            raise InputError("Give a volatility (sigma) or a market price (market_price).")
        return Contract(*validate_option(S0, K, T, r, q, sigma, kind))
    if sigma is not None:
        raise InputError("Give either a volatility or a market price, not both.")
    # A placeholder volatility lets validate_option check everything else.
    S0, K, T, r, q, _, kind = validate_option(S0, K, T, r, q, MIN_VOL, kind)
    iv = solve_implied_vol(S0, K, T, r, q, parse_float(market_price, "Market price"), kind)
    return Contract(S0, K, T, r, q, iv["sigma"], kind, iv)


def spot_ladder(S0, K, T, r, q, sigma, kind, market_price=None) -> dict:
    """The pricing pane's spot-ladder view: price and delta across spot."""
    c = resolve_option(S0, K, T, r, q, sigma, kind, market_price)
    return {"K": c.K, "ladder": _ladder(c.S0, c.K, c.T, c.r, c.q, c.sigma, c.kind)}


def fd_grid(S0, K, T, r, q, sigma) -> dict:
    """The one grid the app gives every finite-difference and finite-element solve.

    The library's grid spans ln(S0) +/- S_max_mult * sigma * sqrt(T) and knows nothing
    about the strike or the drift, so a far strike or a large |r - q| T leaves the payoff
    kink or the forward outside it. This widens the range only when needed, to reach
    whichever of ln(K / S0) and (r - q) T is further out plus FD_MARGIN of sigma*sqrt(T),
    never below FD_BASE_MULT. Widening at fixed N_S coarsens the spacing, so N_S grows
    with the range (kept even, so a node sits on S0), up to FD_MAX_N_S.

    On the t21 samples at 400 x 400 (FD / FEM, prices off by more than 10 bp of BS):
    at the app's full caps 37.3% / 37.3% with the library's fixed 4 sigma*sqrt(T) grid,
    8.7% / 8.7% with this rule, 10.1% / 9.5% widening without scaling N_S.
    """
    total_vol = sigma * math.sqrt(T)
    reach = max(abs(math.log(K / S0)), abs((r - q) * T)) / total_vol
    mult = max(FD_BASE_MULT, reach + FD_MARGIN)
    widened = mult > FD_BASE_MULT
    n_s, capped = FD_GRID, False
    if widened:
        want = 2 * math.ceil(FD_GRID * mult / FD_BASE_MULT / 2)
        n_s, capped = min(want, FD_MAX_N_S), want > FD_MAX_N_S
    return {"S_max_mult": mult, "N_S": n_s, "N_t": FD_GRID, "widened": widened, "capped": capped}


def _grid_solve(opt, kind: str, grid: dict, american: bool) -> tuple[dict | None, str | None]:
    """One finite-difference solve: price, delta, gamma and theta, all from the same
    grid. Returns ``(values, None)`` or ``(None, reason)``."""
    try:
        g = fd_greeks(opt, kind, N_S=grid["N_S"], N_t=grid["N_t"],
                      S_max_mult=grid["S_max_mult"], american=american)
        g = {k: float(v) for k, v in g.items()}
    except (ValueError, ArithmeticError) as exc:
        return None, str(exc)
    if not all(math.isfinite(v) for v in g.values()):
        return None, "the engine returned a non-finite value"
    return g, None


def _american_vega_rho(opt, kind: str, grid: dict) -> tuple[dict | None, str | None]:
    """Vega and rho of the American finite-difference price, by differences of that price.

    Returns ``({"vega", "rho", "rho_note"}, None)`` or ``(None, reason)``; ``rho`` is None
    with its reason in ``rho_note`` when only rho is refused.

    The grid is held fixed. Bumping sigma would move the nodes, since the range is
    S_max_mult * sigma * sqrt(T), so S_max_mult is rescaled to keep the same log-spot
    range; the grid ignores r, so the rate bump needs no such care. The four bumped solves
    (five near r = 0) take FD_BUMP_N_T time steps, half the grid's: both sides of a
    difference share the time step, and against a 4000-step tree this moves rho by 0.1%
    and vega by nothing, for half the time.

    The volatility bump is proportional to sigma (capped at AMERICAN_VEGA_BUMP). A central
    difference errs by h^2 times the third derivative, which grows as sigma falls, so a
    fixed +/-0.01 was 70% off at sigma = 2%.

    Rho is a central difference, except within one bump of r = 0. There early exercise
    switches on (a put for r > 0, a no-dividend call for r < 0) and the American price
    has a kink in r, so a central difference would average two different slopes. It uses
    the second-order one-sided slope on r's own side (up at r = 0) and says so.

    Two regions are refused because the grid's own error, not the bump, is past 1% there:
    total volatility above AMERICAN_MAX_TOTAL_VOL (vega and rho), and a rate near zero
    (|r| T under AMERICAN_ZERO_RATE_BAND) with total volatility above
    AMERICAN_ZERO_RATE_MAX_TOTAL_VOL or sigma above AMERICAN_ZERO_RATE_MAX_SIGMA (rho
    only): the American grid prices a small spurious early-exercise premium there whose
    slope in r is several percent of rho. The library's
    known dividend-boundary error (q != 0, deep in the money, long dated, low volatility)
    is not detectable from here and is left as it is.
    """
    total = opt.sigma * math.sqrt(opt.T)
    if total > AMERICAN_MAX_TOTAL_VOL:
        return None, (f"past a volatility x sqrt(expiry) of {AMERICAN_MAX_TOTAL_VOL:g} the "
                      "grid's own error takes a bumped price more than 1% from the true "
                      "derivative")

    def price(o, mult):
        return fd_price(o, kind, N_S=grid["N_S"], N_t=min(grid["N_t"], FD_BUMP_N_T),
                        S_max_mult=mult, american=True)

    mult, s, r, hr = grid["S_max_mult"], opt.sigma, opt.r, AMERICAN_RHO_BUMP
    hs = min(AMERICAN_VEGA_BUMP, AMERICAN_VEGA_REL_BUMP * s)
    rho, note = None, None
    try:
        vega = (price(replace(opt, sigma=s + hs), mult * s / (s + hs))
                - price(replace(opt, sigma=s - hs), mult * s / (s - hs))) / (2 * hs)
        high_vol = (total > AMERICAN_ZERO_RATE_MAX_TOTAL_VOL
                    or s > AMERICAN_ZERO_RATE_MAX_SIGMA)
        if abs(r) * opt.T < AMERICAN_ZERO_RATE_BAND and high_vol:
            note = ("Not computed: with the rate near zero and a volatility this high, the grid "
                    "prices a small spurious early-exercise premium whose slope in the rate is "
                    "several percent of rho.")
        elif abs(r) < hr:
            side = 1.0 if r >= 0 else -1.0
            p0 = price(opt, mult)
            p1 = price(replace(opt, r=r + side * hr), mult)
            p2 = price(replace(opt, r=r + 2 * side * hr), mult)
            rho = side * (-3 * p0 + 4 * p1 - p2) / (2 * hr)
            which = "r ≥ 0" if side > 0 else "r ≤ 0"
            note = (f"One-sided slope, for {which}: the rate is within {hr:g} of zero, where "
                    "early exercise switches on and the American price has a kink in the rate.")
        else:
            rho = (price(replace(opt, r=r + hr), mult)
                   - price(replace(opt, r=r - hr), mult)) / (2 * hr)
    except (ValueError, ArithmeticError) as exc:
        return None, str(exc)
    if not (math.isfinite(vega) and (rho is None or math.isfinite(rho))):
        return None, "the engine returned a non-finite value"
    return {"vega": float(vega), "rho": None if rho is None else float(rho),
            "rho_note": note}, None


def _grid_detail(grid: dict) -> str:
    """Tooltip for the finite-difference and finite-element Method cells."""
    text = ("Finite difference and finite element share this grid, so they fail together "
            "and their agreement is not independent evidence.")
    if grid["widened"]:
        text = (f"Grid range widened from ±{FD_BASE_MULT:g} to ±{grid['S_max_mult']:.3g} σ√T "
                f"to reach the strike and the forward, with {grid['N_S']} spatial intervals"
                + (" (capped)" if grid["capped"] else "") + ". " + text)
    return text


def price_all(S0, K, T, r, q, sigma, kind, mc_paths=DEFAULT_MC_PATHS, *,
              market_price=None) -> dict:
    """Price one contract with Black-Scholes, the binomial tree (European and
    American), Monte Carlo, finite differences (European and American) and finite
    elements, plus Greeks and a spot ladder.

    Give ``sigma``, or ``market_price`` to price at the volatility that reproduces it.
    """
    c = resolve_option(S0, K, T, r, q, sigma, kind, market_price)
    S0, K, T, r, q, sigma, kind = c[:7]
    mc_paths = int(_check_range(parse_float(mc_paths, "Monte Carlo paths"),
                                "Monte Carlo paths", 1_000, MAX_MC_PATHS))

    # Black-Scholes, its Greeks and the ladder anchor the panel: if they fail the
    # request fails. Every other engine may fail on its own without hiding them.
    try:
        opt = op.OptionSpec(S0=S0, K=K, T=T, r=r, sigma=sigma, q=q)
        bs = float(op.bs_price(opt, kind))
        greeks = {k: float(v) for k, v in op.bs_greeks(opt, kind).items()}
    except (ValueError, ArithmeticError) as exc:
        raise InputError(str(exc)) from exc
    if not (math.isfinite(bs) and all(math.isfinite(v) for v in greeks.values())):
        raise InputError("These inputs produce a non-finite Black-Scholes price or Greek.")
    ladder = _ladder(S0, K, T, r, q, sigma, kind)

    # euro_price_mc counts antithetic pairs: n_paths pairs are 2 * n_paths terminal prices.
    mc_note = f"{mc_paths:,} pairs, antithetic + control variate"
    mc_head = {"name": "Monte Carlo", "note": mc_note, "pairs": mc_paths,
               "terminal_prices": 2 * mc_paths,
               "detail": f"Each antithetic pair is two terminal prices: {2 * mc_paths:,} in all."}
    try:
        mc_px, mc_se = op.euro_price_mc(opt, kind, n_paths=mc_paths, seed=MC_SEED,
                                        n_workers=1)
        mc_px, mc_se = float(mc_px), float(mc_se)
    except (ValueError, ArithmeticError) as exc:
        mc = {**mc_head, "price": None, "unavailable": str(exc)}
    else:
        if math.isfinite(mc_px) and math.isfinite(mc_se):
            # z is how many standard errors the estimate sits from Black-Scholes. Deep in the
            # money the control variate absorbs the whole payoff and the standard error
            # collapses to rounding level while the estimate keeps a tiny bias (the unsampled
            # out-of-the-money tail), so z would flag an accurate price as a failure.
            if mc_se > MC_SE_FLOOR * max(abs(bs), abs(mc_px)):
                mc = {**mc_head, "price": mc_px, "stderr": mc_se,
                      "z": (mc_px - bs) / mc_se, "z_note": None}
            else:
                mc = {**mc_head, "price": mc_px, "stderr": mc_se, "z": None,
                      "z_note": ("No z: the standard error is below "
                                 f"{MC_SE_FLOOR:g} of the price, because the control variate "
                                 "absorbs the payoff. It then says nothing about the distance "
                                 "to Black-Scholes.")}
        else:
            mc = {**mc_head, "price": None,
                  "unavailable": "the engine returned a non-finite price"}

    # One grid for every finite-difference and finite-element solve. The European and
    # American solves each give price, delta, gamma and theta at once.
    grid = fd_grid(S0, K, T, r, q, sigma)
    fd_eu, fd_eu_err = _grid_solve(opt, kind, grid, american=False)
    fd_am, fd_am_err = _grid_solve(opt, kind, grid, american=True)
    vega_rho, vega_rho_err = _american_vega_rho(opt, kind, grid) if fd_am else (None, fd_am_err)
    size = f"{grid['N_S']}x{grid['N_t']}"
    widened = ", range widened" if grid["widened"] else ""
    detail = _grid_detail(grid)

    def solved(name, note, values, reason):
        e = ({"name": name, "price": values["price"], "note": note} if values
             else {"name": name, "price": None, "note": note, "unavailable": reason})
        e["detail"] = detail
        return e

    engines = [
        {"name": "Black-Scholes", "price": bs, "note": "closed form"},
        _binomial_row(opt, kind, american=False),
        _binomial_row(opt, kind, american=True),
        mc,
        solved("Finite difference", f"theta-scheme, {size}{widened}", fd_eu, fd_eu_err),
        solved("Finite difference (American)", f"theta-scheme, {size}{widened}, early exercise",
               fd_am, fd_am_err),
        {**_row("Finite element", f"Galerkin, {size}{widened}",
                lambda: fem_price(opt, kind, N_S=grid["N_S"], N_t=grid["N_t"],
                                  S_max_mult=grid["S_max_mult"])), "detail": detail},
    ]
    for e in engines:
        e["diff_bps"] = ((e["price"] - bs) / bs * 1e4
                         if e["price"] is not None and bs > 1e-12 else None)
    by_name = {e["name"]: e for e in engines}

    def premium(american, european):
        a, b = by_name[american]["price"], by_name[european]["price"]
        # rounded, with no -0.0, so a rounding-level difference does not show as "-0.0000"
        return None if a is None or b is None else round(a - b, 6) + 0.0

    def grid_greeks(values):
        return {k: values[k] if values else None for k in ("delta", "gamma", "theta")}

    greeks_fd = {"european": {**grid_greeks(fd_eu), "vega": None, "rho": None},
                 "american": {**grid_greeks(fd_am),
                              "vega": vega_rho["vega"] if vega_rho else None,
                              "rho": vega_rho["rho"] if vega_rho else None,
                              "rho_note": vega_rho["rho_note"] if vega_rho else None}}

    # The table folds American into a column of its row; the flat list above stays the
    # JSON API's, so the routes drop "rows" as they drop "ladder".
    rows = [
        {"label": "Black-Scholes", "main": by_name["Black-Scholes"], "american": None},
        {"label": "Binomial", "main": by_name["Binomial (European)"],
         "american": by_name["Binomial (American)"]},
        {"label": "Monte Carlo", "main": mc, "american": None},
        {"label": "Finite difference", "main": by_name["Finite difference"],
         "american": by_name["Finite difference (American)"]},
        {"label": "Finite element", "main": by_name["Finite element"], "american": None},
    ]

    out = {
        "inputs": {"S0": S0, "K": K, "T": T, "r": r, "q": q, "sigma": sigma, "kind": kind},
        "engines": engines,
        "greeks": greeks,
        "greeks_fd": greeks_fd,
        "american_vega_rho_unavailable": None if vega_rho else vega_rho_err,
        "early_exercise_premium": {
            "tree": premium("Binomial (American)", "Binomial (European)"),
            "grid": premium("Finite difference (American)", "Finite difference"),
        },
        "grid": grid,
        "rows": rows,
        "ladder": ladder,
    }
    if c.iv is not None:
        out["implied_vol"] = c.iv
    return out


def scenario_grid(S0, K, T, r, q, sigma, kind, market_price=None) -> dict:
    """The pricing pane's scenarios view: Black-Scholes P&L against the base price over
    spot, volatility and rate shocks, rounded for the chart.

    ``optpricer.validation.stress_test`` prices the cube. It floors a shocked volatility
    at 1e-6 without saying so, so every cell whose volatility falls below MIN_VOL is
    masked (None) rather than shown as a price at a volatility nobody asked for.
    ``pnl`` is indexed ``[rate][vol][spot]``.
    """
    c = resolve_option(S0, K, T, r, q, sigma, kind, market_price)
    opt = op.OptionSpec(S0=c.S0, K=c.K, T=c.T, r=c.r, sigma=c.sigma, q=c.q)
    spot_shock = np.array(SCENARIO_SPOT_SHOCKS)
    vol_shock = np.array(SCENARIO_VOL_SHOCKS)
    rate_shock = np.array(SCENARIO_RATE_SHOCKS)
    try:
        cube = op.stress_test(opt, c.kind, spot_shock, vol_shock, rate_shock, pricer="bs")
    except (ValueError, ArithmeticError) as exc:
        raise InputError(str(exc)) from exc
    base = float(cube[SCENARIO_SPOT_SHOCKS.index(1.0), SCENARIO_VOL_SHOCKS.index(0.0),
                      SCENARIO_RATE_SHOCKS.index(0.0)])
    if not np.all(np.isfinite(cube)):
        raise InputError("These inputs produce a non-finite Black-Scholes price.")
    masked_vol = (c.sigma + vol_shock) < MIN_VOL
    raw = cube - base                                                 # [spot, vol, rate]
    raw[:, masked_vol, :] = np.nan
    valid = np.isfinite(raw)
    # A P&L under 1e-8 of the underlying's scale is rounding noise, so it is zero. Above that,
    # keep six significant digits of the largest cell: six decimals would leave a spot of 1e-4,
    # whose P&L cells are around 1e-6, with a few dozen distinct values.
    raw[valid & (np.abs(np.nan_to_num(raw)) < SCENARIO_NOISE * max(c.S0, c.K))] = 0.0
    scale = float(np.max(np.abs(raw[valid])))
    decimals = 6 if scale == 0.0 else int(min(15, max(6, 6 - math.floor(math.log10(scale)))))
    pnl = np.round(raw, decimals) + 0.0                               # no -0.0
    cells = np.transpose(pnl, (2, 1, 0)).astype(object)               # [rate, vol, spot]
    cells[~np.transpose(valid, (2, 1, 0))] = None
    vol = np.round(c.sigma + vol_shock, 6).astype(object)
    vol[masked_vol] = None
    return {
        "S0": c.S0, "sigma": c.sigma, "r": c.r, "base": base,
        "spot": [float(f"{x:.10g}") for x in c.S0 * spot_shock],
        "spot_shock_pct": np.round((spot_shock - 1.0) * 100.0, 4).tolist(),
        "vol": vol.tolist(),
        "vol_shock_pts": np.round(vol_shock * 100.0, 4).tolist(),
        "rate_shock_pts": np.round(rate_shock * 100.0, 4).tolist(),
        "pnl": cells.tolist(),
        "masked_vols": int(masked_vol.sum()),
        "all_zero": scale == 0.0,
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


FORWARD_TOLERANCE = 0.005   # a surface's forward may differ from S0 e^{(r-q)t} by 0.5%


def _forward_refusal(F: float, carry: float, t: float) -> InputError:
    return InputError(
        f"The surface's forward at t={t:g} is {F:.6g}, but spot x e^((r - q) t) is "
        f"{carry:.6g}. It was fitted at a different spot, rate or dividend yield, or "
        "its forward curve was changed after fitting; refit it here."
    )


def check_forward(surface: VolSurface, S0: float, t: float, r: float, q: float,
                  *, through_expiry: bool = False) -> float:
    """The surface's forward at ``t``, refused unless it agrees with ``S0 e^{(r-q)t}``.

    The surface JSON carries the forward it was fitted with, and Dupire local
    vol is evaluated at that forward, not at the spot the request names.  A
    surface fitted at another spot or rate, or one whose forward curve was
    edited, therefore prices a different market without any sign of it (a
    feasibility study priced a call at 54.39 at S0 = 100 with the forward
    multiplied by 1e6).  Every local-vol route calls this before it computes
    anything.

    The tolerance is 0.5%, widened by the slack of the surface's own linear
    interpolation between quoted forwards.  That interpolation is exact at each
    quoted expiry but sits above the exponential in between, by 0.03% at a 5%
    rate and 3.2% at the 50% rate cap; without the allowance a correctly fitted
    surface would be refused at a high rate.

    A route that reads the forward at one time only, as the local-vol slice does, checks
    ``t`` alone.  The PDE views read it at every time step from 0 to ``t``, so with
    ``through_expiry`` every quoted forward up to the first quoted expiry at or after ``t``
    is checked as well (all of them if ``t`` is past the last).  Those are the points the
    forward at any earlier time is interpolated from.  A correct fit stores
    ``S0 e^{(r-q)T}`` at each of them, so the plain 0.5% applies, with no interpolation
    slack.  Without this a curve edited at an earlier expiry passed whenever ``t`` sat at
    or beyond the next one.

    Raises InputError, including for a surface with no forward curve at all
    (``forward_at`` raises a bare ValueError there).
    """
    try:
        F = float(surface.forward_at(t, r, q))
        curve = [(float(T), float(F_T)) for T, F_T in surface.to_dict()["forward_curve"]]
    except (ValueError, ArithmeticError) as exc:
        raise InputError(f"Invalid surface: {exc}") from exc
    carry = S0 * math.exp((r - q) * t)
    expected = [carry]
    quoted = [T for T, _ in curve]
    if len(quoted) > 1 and quoted[0] <= t <= quoted[-1]:
        # The forward a correctly fitted surface interpolates to at t.
        expected.append(float(np.interp(
            t, quoted, [S0 * math.exp((r - q) * T) for T in quoted])))
    if not math.isfinite(F) or min(abs(F / e - 1.0) for e in expected) > FORWARD_TOLERANCE:
        raise _forward_refusal(F, carry, t)
    if through_expiry:
        at_or_after = [T for T in quoted if T >= t]
        horizon = min(at_or_after) if at_or_after else math.inf
        for T, F_T in curve:
            exact = S0 * math.exp((r - q) * T)
            if T <= horizon and (not math.isfinite(F_T)
                                 or abs(F_T / exact - 1.0) > FORWARD_TOLERANCE):
                raise _forward_refusal(F_T, exact, T)
    return F


def local_vol_slice(surface_json, S0, t, r, q) -> dict:
    surface = load_surface(surface_json)
    S0 = _check_range(parse_float(S0, "Spot"), "Spot", MIN_PRICE, 1e6)
    t = _check_range(parse_float(t, "Time"), "Time", MIN_EXPIRY, 30.0)
    r = _check_range(parse_float(r, "Rate"), "Rate", -0.5, 0.5)
    q = _check_range(parse_float(q, "Dividend yield"), "Dividend yield", -0.5, 0.5)
    forward = check_forward(surface, S0, t, r, q)

    spots = np.linspace(0.6 * S0, 1.4 * S0, 81)
    try:
        lv = np.atleast_1d(dupire_local_vol(surface, spots, t, r, q))
        w = surface.total_var_from_logm(np.log(spots / forward), t)
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
