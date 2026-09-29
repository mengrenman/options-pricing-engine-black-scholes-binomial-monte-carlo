"""Local-vol views for the surface pane, on top of the finite-difference PDE.

The fitted SVI surface fixes an implied-volatility smile.  Dupire's formula turns
it into a local-volatility function, and the local-vol PDE turns that function
back into option prices.  Two questions follow, one view each:

* :func:`lv_smile` -- does the round trip reproduce the smile it started from?
  Price a strip of out-of-the-money strikes with the PDE, invert the prices to
  implied vols, and compare them with the surface's own.
* :func:`lv_dynamics` -- what does that local-vol model say about delta, gamma
  and how the smile moves with spot?

The views work the same way in these respects:

* **One grid, one memo.**  The PDE grid depends on ``(S0, T, ref_vol, N_S)`` and
  never on the strike, so every solve of a request reads the same local-vol
  values at the same nodes.  :class:`LocalVolTable` keeps them for the request
  (never in module state, so requests cannot see each other), which turns eleven
  strikes from eleven Dupire evaluations per time step into one.  A memoized
  price is bit-for-bit the unmemoized one.
* **Local vol is read at the middle of each time step.**  The library's solver asks
  ``sigma_func`` for the local vol at the *end* of each backward step, the earlier of its
  two times.  With a local vol that depends on time that is a first-order error at 100
  steps: on a surface whose local vol depends on time alone, where the round trip is exact,
  it showed 7 to 32 vol bps of gap.  The tables add half a step, ``T / (2 N_t)``, before
  they ask Dupire's formula, which costs nothing and leaves 0 to 3 bps (see
  :func:`_mid_step`).
* **The smile's strikes sit on grid nodes.**  Each is snapped to the nearest node, and the
  snapped strike is the one displayed and the one Black-Scholes is evaluated at.  On a flat
  surface, where the answer is known, this moves the worst strike from 0.3 to 0.2 vol bps
  off, so it is tidiness more than accuracy.  The dynamics view prices the strike as typed:
  its spots move by whole nodes, so the strike keeps its place among the nodes whatever it is.
* **The grid is sized by the largest vol in play**, not the ATM vol.  A feasibility
  study that used the ATM vol had a short-dated deep strike four nodes from the
  boundary, repricing 394 vol bps low.
* **One count of clamped nodes.**  Both views report the share of local-vol nodes at the
  library's floor or cap among the nodes within four ATM standard deviations of the forward
  (:func:`_clip_window`), not among their whole grids, so one surface gives one share on
  either tab.
* **The forward is checked along the whole path.**  The PDE reads the surface's forward at
  every time step, so :func:`webapp.engine.check_forward` checks every quoted forward up to
  the expiry, not only the one at ``T``.

Every input is checked here as well as in :mod:`webapp.engine`, and every number
returned is finite or ``None``.
"""

from __future__ import annotations

import math

import numpy as np

import optpricer as op
from optpricer.calibration import (
    LOCAL_VOL_CAP,
    LOCAL_VOL_FLOOR,
    VolSurface,
    dupire_local_vol_func,
)

from . import engine
from .engine import InputError, _check_range, parse_float

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------
LV_N_S = 300               # PDE space steps
LV_N_T = 100               # PDE time steps
LV_S_MAX_MULT = 4.0        # grid half-range, in ref_vol * sqrt(T)
LV_MIN_T = 0.05            # shortest expiry these views accept, years
LV_N_STRIKES = 11
LV_MAX_LOGM = 0.5          # widest |ln(K / F)| priced
LV_MAX_STD = 3.0           # ... or this many ATM standard deviations, if fewer
LV_MIN_REF_VOL = 0.05      # floor on the grid-sizing vol
LV_MAX_TOTAL_VOL = 2.0     # ref_vol * sqrt(T)
LV_MAX_CARRY_SHARE = 0.3   # |r - q| * T, as a share of the grid half-range
LV_EDGE_MARGIN = 0.75      # standard deviations between the farthest strike and the grid edge

LV_GREEKS_SHIFT = 1        # delta and gamma come from spot moved by this many nodes
LV_LADDER_SPOTS = 11       # spots in the dynamics chart
LV_LADDER_SPAN = 0.05      # ... reaching about this far each way
# An upper bound on how far the ladder reaches, in log spot.  Rounded to whole nodes it can
# pass the span; five nodes are 0.13 standard deviations of the grid, and 0.2 standard
# deviations covers that on a grid widened by a third (see lv_dynamics).
LV_LADDER_REACH = 0.075
MAX_STRIKE = 1e6           # the largest strike the dynamics view accepts


# ---------------------------------------------------------------------------
# The per-request memo
# ---------------------------------------------------------------------------
def _mid_step(func, T: float):
    """``func`` read half a time step later than the solver asks.

    ``fd_price_local_vol`` divides ``[0, T]`` into ``LV_N_T`` steps and, for the step from
    ``t + dt`` back to ``t``, asks for the local vol at ``t``.  The middle of the step,
    ``t + dt / 2``, is where a second-order scheme wants it.  The shift is applied here so
    every table reads the same one.
    """
    half_step = 0.5 * T / LV_N_T

    def shifted(S: np.ndarray, t: float) -> np.ndarray:
        return func(S, t + half_step)

    return shifted


def _count_at_bounds(pairs, lo: float, hi: float) -> tuple[int, int]:
    """``(nodes at the library's floor or cap, nodes counted)`` over ``(spots, values)``
    pairs, counting only the nodes whose spot lies in ``[lo, hi]``."""
    at_bound = counted = 0
    for spots, values in pairs:
        inside = (spots >= lo) & (spots <= hi)
        hit = (values >= LOCAL_VOL_CAP * (1 - 1e-9)) | (values <= LOCAL_VOL_FLOOR * (1 + 1e-9))
        at_bound += int(np.count_nonzero(hit & inside))
        counted += int(np.count_nonzero(inside))
    return at_bound, counted


class LocalVolTable:
    """A memo of ``dupire_local_vol_func``, valid for the life of one request.

    The PDE solver calls ``sigma_func(S_nodes, t)`` once per time step.  Solves
    for different strikes but one ``(S0, T, ref_vol, N_S)`` ask for identical
    nodes at identical times, so the second and later solves read the first's
    answers.  The key holds the grid's size and end nodes as well as ``t``, so a
    solve on a different grid can never be handed another grid's values.  The key
    is the time the solver asks for; the value is Dupire's local vol half a step
    later (see :func:`_mid_step`).

    The cached arrays are read-only: a solver that modified one in place would
    silently corrupt every later strike.
    """

    def __init__(self, surface: VolSurface, r: float, q: float, T: float):
        self._func = _mid_step(dupire_local_vol_func(surface, r, q), T)
        self._memo: dict[tuple, np.ndarray] = {}
        self._spots: dict[tuple, np.ndarray] = {}

    def __call__(self, S: np.ndarray, t: float) -> np.ndarray:
        key = (t, S.shape[0], float(S[0]), float(S[-1]))
        values = self._memo.get(key)
        if values is None:
            values = np.asarray(self._func(S, t), dtype=float)
            values.setflags(write=False)
            self._memo[key] = values
            self._spots[key] = np.array(S, dtype=float)
        return values

    def clipped(self, lo: float = 0.0, hi: float = math.inf) -> tuple[int, int]:
        """``(nodes at the library's floor or cap, nodes counted)``, for the nodes whose
        spot lies in ``[lo, hi]``.

        Counted from the table itself: the library warns once per process, so its
        warnings say nothing about this request.  The range is the strikes the view
        prices, so the share describes the surface where it matters and does not depend
        on how far out a particular view happens to size its grid.
        """
        return _count_at_bounds(((self._spots[k], v) for k, v in self._memo.items()), lo, hi)


class _ShiftedTable:
    """Local vol on one wide grid, read through windows shifted by whole nodes.

    Solves at spots ``S0 e^{j dx}`` each build their own grid, the base grid
    moved ``j`` nodes.  Every such grid is a window onto one long grid, so its
    local vol is a slice of one table, computed once per time step.
    """

    def __init__(self, surface, r, q, T, S0, half_range, n_s, max_shift):
        self.n_s = n_s
        self.max_shift = max_shift
        dx = 2.0 * half_range / n_s
        u = math.log(S0) - half_range + dx * np.arange(-max_shift, n_s + max_shift + 1)
        self.spots = np.exp(u)
        self._func = _mid_step(dupire_local_vol_func(surface, r, q), T)
        self._memo: dict[float, np.ndarray] = {}

    def _at(self, t: float) -> np.ndarray:
        values = self._memo.get(t)
        if values is None:
            values = np.asarray(self._func(self.spots, t), dtype=float)
            values.setflags(write=False)
            self._memo[t] = values
        return values

    def window(self, shift: int):
        lo = self.max_shift + shift
        hi = lo + self.n_s + 1

        def sigma(S: np.ndarray, t: float) -> np.ndarray:
            # The solver builds its own grid; check it is the window it is given.
            if S.shape[0] != self.n_s + 1 or not (
                math.isclose(float(S[0]), float(self.spots[lo]), rel_tol=1e-9)
                and math.isclose(float(S[-1]), float(self.spots[hi - 1]), rel_tol=1e-9)
            ):
                raise ValueError("The PDE grid does not line up with the local-vol table.")
            return self._at(t)[lo:hi]

        return sigma

    def clipped(self, lo: float = 0.0, hi: float = math.inf) -> tuple[int, int]:
        """As :meth:`LocalVolTable.clipped`, over the whole wide table."""
        return _count_at_bounds(((self.spots, v) for v in self._memo.values()), lo, hi)


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------
def round_strike(x: float) -> float:
    """Three significant figures: the default strike is the ATM forward, rounded."""
    return float(f"{x:.3g}") if math.isfinite(x) and x > 0 else x


def default_strike(S0: float, t: float, r: float, q: float) -> float:
    """The ATM forward ``S0 e^{(r-q)t}``, rounded, and kept inside the strike limits so that
    a default is never refused for being out of range (a spot near its cap, carried
    forward, is above the cap on strikes)."""
    return min(max(round_strike(S0 * math.exp((r - q) * t)), engine.MIN_PRICE), MAX_STRIKE)


def sig(x, digits: int = 4, trim: bool = False) -> str:
    """``x`` to ``digits`` significant figures, without an exponent, or ``n/a`` for ``None``.

    Fixed decimals hide small numbers: gamma is 2e-5 at a spot of 50,000 and printed as
    ``0.0000``, so both models seemed to have none.  Here the decimals follow the size of the
    number, and trailing zeros stay so that a column lines up (``trim`` drops them: a strike
    typed as 105 reads 105, not 105.00).
    """
    if x is None or not math.isfinite(x):
        return "n/a"
    if x == 0.0:
        return "0"
    decimals = min(max(digits - 1 - math.floor(math.log10(abs(x))), 0), 12)
    text = f"{x:.{decimals}f}"
    return text.rstrip("0").rstrip(".") if trim and "." in text else text


def _context(surface_json, S0, t, r, q):
    """Parse and validate what every local-vol view shares.

    Returns ``(surface, S0, T, r, q, forward)``.  The time is the option's expiry,
    restricted to the surface's quoted range: beyond the last quoted expiry
    the surface is an extrapolation, and a local-vol price of an extrapolated
    smile is a statement about the extrapolation, not the market.

    The forward is checked at ``T`` *and* at every quoted expiry up to the first one at or
    after ``T``: the PDE reads Dupire's formula, and so the surface's forward, at every
    time step from 0 to ``T``, not only at ``T``.
    """
    surface = engine.load_surface(surface_json)
    S0 = _check_range(parse_float(S0, "Spot"), "Spot", engine.MIN_PRICE, 1e6)
    r = _check_range(parse_float(r, "Rate"), "Rate", -0.5, 0.5)
    q = _check_range(parse_float(q, "Dividend yield"), "Dividend yield", -0.5, 0.5)
    last = float(surface.expiries[-1])
    if last < LV_MIN_T:
        raise InputError(f"The surface's last quoted expiry, {last:g} years, is shorter than "
                         f"the {LV_MIN_T:g} years these views need.")
    T = _check_range(parse_float(t, "Time"), "Time", LV_MIN_T, last)
    forward = engine.check_forward(surface, S0, T, r, q, through_expiry=True)
    return surface, S0, T, r, q, forward


def _atm_vol(surface: VolSurface, T: float) -> float:
    iv = float(surface.iv_from_logm(0.0, T))
    if not (math.isfinite(iv) and iv > 0.0):
        raise InputError(f"The surface has no positive implied volatility at t={T:g}.")
    return iv


def _forward_gap(F: float, S0: float, T: float, r: float, q: float) -> float:
    """``F / (S0 e^{(r-q)T}) - 1``: how far the surface's forward at ``T`` sits from the one
    the PDE grows spot at.

    Zero at a quoted expiry.  Between two the surface interpolates its forwards linearly,
    which lies above the exponential (0.03% at a 5% rate, 1.1% at 30%), and the comparison
    then mixes two forwards: Dupire and the surface's own smile use one, the PDE and the
    Black-Scholes inversion the other.  The page says so when the gap is not negligible.
    """
    return F / (S0 * math.exp((r - q) * T)) - 1.0


def _clip_window(F: float, atm_iv: float, T: float) -> tuple[float, float]:
    """The spots whose clamped local vol the status line counts: ``LV_S_MAX_MULT`` ATM
    standard deviations either side of the forward.

    Both views count over this same window, whatever size grid each builds (each grid is at
    least this wide, up to the carry), so one surface reports one share on either tab.  It
    is wider than the priced strikes on purpose: a put's price depends on the local vol
    below its strike as much as at it, and a surface's bad wing typically starts just
    outside the strikes (at about ``ln(K / F) = -0.55`` on the old default smile).
    """
    half = LV_S_MAX_MULT * atm_iv * math.sqrt(T)
    return F * math.exp(-half), F * math.exp(half)


def _logm_limit(atm_iv: float, T: float) -> float:
    """Widest ``|ln(K / F)|`` these views price: ``LV_MAX_LOGM`` or, for short expiries,
    three ATM standard deviations.  The synthetic quotes reach only ``ln(0.6)`` to
    ``ln(1.4)`` around the forward, so the outer dots test the SVI extrapolation, not quoted
    data; and a time below the first quoted expiry is itself extrapolated."""
    return min(LV_MAX_LOGM, LV_MAX_STD * atm_iv * math.sqrt(T))


def _grid(T: float, r: float, q: float, ref_vol: float,
          reach: float) -> tuple[float, float, float]:
    """``(grid_vol, half_range, dx)`` of the log-spot grid, after the two size caps.

    ``ref_vol`` sizes the grid: its half-range is ``LV_S_MAX_MULT`` standard
    deviations of ``ref_vol``.  ``reach`` is the farthest ``|ln(K / S0)|`` any
    solve must price, and the grid is widened, only when it must be, until that
    strike is ``LV_EDGE_MARGIN`` standard deviations inside the edge.  The
    boundary condition ignores time value, so a strike near the edge reprices
    low: on a flat 20% surface the error is 62 vol bps at 0.1 standard
    deviations from the edge, 16 at 0.25, 1.1 at 0.5 and under 0.5 from 0.75.
    A large carry against a low vol is what pushes a strike out there, because the
    grid is centered on spot and the strikes on the forward.
    """
    if ref_vol * math.sqrt(T) > LV_MAX_TOTAL_VOL:
        raise InputError(
            f"Volatility x sqrt(time) is {ref_vol * math.sqrt(T):.3g}; the local-vol grid "
            f"is capped at {LV_MAX_TOTAL_VOL:g}.")
    half_range = LV_S_MAX_MULT * ref_vol * math.sqrt(T)
    if abs(r - q) * T > LV_MAX_CARRY_SHARE * half_range:
        raise InputError(
            f"Carry |r - q| x t is {abs(r - q) * T:.3g}, more than {LV_MAX_CARRY_SHARE:g} of "
            f"the grid half-range {half_range:.3g}; the forward would sit too near its edge.")
    grid_vol = max(ref_vol, (reach / math.sqrt(T)) / (LV_S_MAX_MULT - LV_EDGE_MARGIN))
    half_range = LV_S_MAX_MULT * grid_vol * math.sqrt(T)
    return grid_vol, half_range, 2.0 * half_range / LV_N_S


def _snap(S0: float, dx: float, strikes) -> np.ndarray:
    """Move each strike to the nearest node of the grid centered on ``S0``."""
    return np.array([S0 * math.exp(round(math.log(K / S0) / dx) * dx) for K in strikes])


def _num(x, ndigits: int):
    """A rounded float, or ``None`` if it is not finite (JSON has no NaN)."""
    if x is None:
        return None
    x = float(x)
    return round(x, ndigits) if math.isfinite(x) else None


def _sf(x, digits: int = 6):
    """``x`` rounded to ``digits`` significant figures, or ``None`` if it is not finite.
    For numbers whose size varies by orders of magnitude with spot (gamma)."""
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    return round(x, digits - 1 - math.floor(math.log10(abs(x)))) if x != 0.0 else 0.0


def implied_vol(S0, K, T, r, q, price, kind) -> float | None:
    """Black-Scholes implied vol of ``price``, or ``None`` if there is none.

    Brent's method through the library's scalar ``implied_vol``, entered only
    for a price strictly inside the no-arbitrage bounds.  It replaces a Newton
    iteration from one starting vol: ``bs_implied_vol_vec`` returns NaN on many
    out-of-the-money prices, where vega is tiny and the first step overshoots.
    A price the PDE rounded to zero, or below intrinsic value, has no implied
    vol; the caller shows a gap rather than a made-up number.
    """
    if not math.isfinite(price):
        return None
    disc_r, disc_q = math.exp(-r * T), math.exp(-q * T)
    if kind == op.CALL:
        lower, upper = max(S0 * disc_q - K * disc_r, 0.0), S0 * disc_q
    else:
        lower, upper = max(K * disc_r - S0 * disc_q, 0.0), K * disc_r
    if not lower < price < upper:
        return None
    try:
        vol = op.implied_vol(op.OptionSpec(S0=S0, K=K, T=T, r=r, sigma=0.2, q=q), price, kind)
    except (ValueError, RuntimeError, ArithmeticError):
        return None      # no root in the bracket, e.g. a price beyond the 500% vol cap
    return vol if math.isfinite(vol) and vol > 0.0 else None


def _solve(compute):
    """Run PDE solves, turning a library failure into a refusal the page can show."""
    try:
        return compute()
    except (ValueError, ArithmeticError) as exc:
        raise InputError(str(exc)) from exc


# ---------------------------------------------------------------------------
# View 1: the smile, repriced
# ---------------------------------------------------------------------------
def lv_smile(surface_json, S0, t, r, q) -> dict:
    """Reprice the fitted smile with its own Dupire local vol.

    Eleven strikes, evenly spaced in ``ln(K / F)`` across
    ``|ln(K / F)| <= min(0.5, 3 sigma_ATM sqrt(T))``, snapped to grid nodes and
    priced out of the money (puts below the forward, calls above).  Each price
    is inverted to an implied vol and compared with the surface's own at the
    same strike.

    ``ref_vol``, which sizes the grid, is the largest implied vol over the
    strikes: the wing vol, not the ATM vol.

    ``clipped_share`` counts the PDE nodes whose local vol sits at the library's floor or
    cap, over all time steps, among those within four ATM standard deviations of the forward
    (``clip_range``; see :func:`_clip_window`).  Counting the whole grid would describe how
    wide each view happens to size it: the same surface read 31% on this view and 8% on the
    dynamics view, whose grid reaches less far out.
    """
    surface, S0, T, r, q, F = _context(surface_json, S0, t, r, q)
    atm_iv = _atm_vol(surface, T)
    limit = _logm_limit(atm_iv, T)
    targets = np.linspace(-limit, limit, LV_N_STRIKES)
    edge_iv = np.asarray(surface.iv_from_logm(targets, T), dtype=float)
    if not np.all(np.isfinite(edge_iv)):
        raise InputError(f"The surface yields non-finite implied volatility at t={T:g}.")
    ref_vol = max(float(edge_iv.max()), LV_MIN_REF_VOL)
    reach = abs(math.log(F / S0)) + limit
    grid_vol, _, dx = _grid(T, r, q, ref_vol, reach)

    strikes = _snap(S0, dx, F * np.exp(targets))
    logm = np.log(strikes / F)
    kinds = [op.PUT if K < F else op.CALL for K in strikes]
    iv_svi = np.asarray(surface.iv_from_logm(logm, T), dtype=float)

    table = LocalVolTable(surface, r, q, T)
    prices = _solve(lambda: [
        op.fd_price_local_vol(S0, K, T, r, q, table, kind, N_S=LV_N_S, N_t=LV_N_T,
                              ref_vol=grid_vol, S_max_mult=LV_S_MAX_MULT)
        for K, kind in zip(strikes, kinds)])
    iv_pde = [implied_vol(S0, K, T, r, q, p, kind)
              for K, p, kind in zip(strikes, prices, kinds)]
    diff = [(a - b) * 1e4 if a is not None and math.isfinite(b) and b > 1e-6 else None
            for a, b in zip(iv_pde, iv_svi)]

    atm = int(np.argmin(np.abs(logm)))
    finite = [abs(d) for d in diff if d is not None]
    lo, hi = _clip_window(F, atm_iv, T)
    at_bound, nodes = table.clipped(lo, hi)

    # The chart's smile line, over the same strikes as the dots.
    half_width = float(np.max(np.abs(logm)))
    line_logm = np.linspace(-half_width, half_width, 61)
    line_iv = np.asarray(surface.iv_from_logm(line_logm, T), dtype=float)
    return {
        "t": T, "forward": _sf(F, 8), "forward_gap": _num(_forward_gap(F, S0, T, r, q), 6),
        "ref_vol": _num(ref_vol, 5), "grid_vol": _num(grid_vol, 5),
        "strike": [_sf(K, 8) for K in strikes], "kind": kinds,
        "iv_svi": [_num(v, 5) for v in iv_svi],
        "iv_pde": [_num(v, 5) for v in iv_pde],
        "diff_bps": [_num(d, 1) for d in diff],
        "line": {"strike": [_sf(F * math.exp(k), 8) for k in line_logm],
                 "iv": [_num(v, 5) for v in line_iv]},
        "atm_strike": _sf(strikes[atm], 8), "atm_gap_bps": _num(diff[atm], 1),
        "max_gap_bps": _num(max(finite), 1) if finite else None,
        "unpriced": sum(v is None for v in iv_pde),
        "clipped_nodes": at_bound, "grid_nodes": nodes,
        "clipped_share": _num(at_bound / nodes, 6) if nodes else 0.0,
        "clip_range": [_sf(lo, 8), _sf(hi, 8)],
        "bounds": [LOCAL_VOL_FLOOR, LOCAL_VOL_CAP],
    }


# ---------------------------------------------------------------------------
# View 2: delta and dynamics
# ---------------------------------------------------------------------------
def lv_dynamics(surface_json, S0, K, t, r, q) -> dict:
    """Local-vol delta and gamma, and how the smile moves with spot.

    All prices come from PDE solves whose spot is the base spot moved by a whole
    number of grid nodes, while the strike stays where it is.  Each solve's grid
    is then the base grid shifted, the strike keeps its position relative to the
    nodes, and the discretization error is the same in every solve, so it cancels
    in the differences that make delta and gamma.  All of them read one local-vol
    table on a grid wide enough to hold every shift.  The strike is priced as given, not
    moved to a node: on flat surfaces the price, delta and gamma errors are the same
    either way.

    The option is European and out of the money against the forward (a put below it, a
    call above), as in the smile view.

    Local-vol delta holds the local-vol *function* fixed in absolute spot
    (sticky local vol).  Black-Scholes delta at the surface's implied vol holds
    the strike's implied vol fixed (sticky strike).  They differ by design.
    """
    surface, S0, T, r, q, F = _context(surface_json, S0, t, r, q)
    atm_iv = _atm_vol(surface, T)
    if K is None or (isinstance(K, str) and not K.strip()):
        K = default_strike(S0, T, r, q)
    K = _check_range(parse_float(K, "Strike"), "Strike", engine.MIN_PRICE, MAX_STRIKE)
    limit = _logm_limit(atm_iv, T)
    if abs(math.log(K / F)) > limit:
        raise InputError(
            f"Strike {K:g} is too far from the forward {F:.4g}: this view takes strikes "
            f"from {F * math.exp(-limit):.4g} to {F * math.exp(limit):.4g} at t={T:g}.")
    iv_K = float(surface.iv_from_logm(math.log(K / F), T))
    if not (math.isfinite(iv_K) and iv_K > 0.0):
        raise InputError(f"The surface has no positive implied volatility at strike {K:g}.")
    kind = op.PUT if K < F else op.CALL

    ref_vol = max(iv_K, atm_iv, LV_MIN_REF_VOL)
    # Spot moves up to the chart's reach each way while the strike stays put, so the
    # strike sits that much nearer one edge of the shifted grid.
    reach = abs(math.log(K / S0)) + max(LV_LADDER_REACH, 0.2 * ref_vol * math.sqrt(T))
    grid_vol, half_range, dx = _grid(T, r, q, ref_vol, reach)

    m = LV_GREEKS_SHIFT
    half_n = (LV_LADDER_SPOTS - 1) // 2
    step = max(1, round(math.log1p(LV_LADDER_SPAN) / (half_n * dx)))
    ladder = [j * step for j in range(-half_n, half_n + 1)]
    shifts = sorted({-m, 0, m, *ladder})
    table = _ShiftedTable(surface, r, q, T, S0, half_range, LV_N_S,
                          max(abs(s) for s in shifts))

    def price_at(shift: int) -> float:
        return op.fd_price_local_vol(
            S0 * math.exp(shift * dx), K, T, r, q, table.window(shift), kind,
            N_S=LV_N_S, N_t=LV_N_T, ref_vol=grid_vol, S_max_mult=LV_S_MAX_MULT)

    prices = _solve(lambda: {s: price_at(s) for s in shifts})
    spot = {s: S0 * math.exp(s * dx) for s in shifts}
    lo, mid, hi = prices[-m], prices[0], prices[m]
    h_lo, h_hi = spot[0] - spot[-m], spot[m] - spot[0]
    lv_delta = (hi - lo) / (h_hi + h_lo)
    lv_gamma = 2.0 * ((hi - mid) / h_hi - (mid - lo) / h_lo) / (h_hi + h_lo)

    try:
        opt = op.OptionSpec(S0=S0, K=K, T=T, r=r, sigma=iv_K, q=q)
        bs_price = float(op.bs_price(opt, kind))
        bs = {k: float(v) for k, v in op.bs_greeks(opt, kind).items()}
    except (ValueError, ArithmeticError) as exc:
        raise InputError(str(exc)) from exc

    ladder_iv = [implied_vol(spot[s], K, T, r, q, prices[s], kind) for s in ladder]
    clip_lo, clip_hi = _clip_window(F, atm_iv, T)
    at_bound, nodes = table.clipped(clip_lo, clip_hi)
    return {
        "t": T, "spot0": _sf(S0, 8), "forward": _sf(F, 8),
        "forward_gap": _num(_forward_gap(F, S0, T, r, q), 6),
        "strike": _sf(K, 8), "kind": kind, "iv_strike": _num(iv_K, 5),
        "ref_vol": _num(ref_vol, 5), "grid_vol": _num(grid_vol, 5),
        "node_pct": _num((math.exp(dx) - 1.0) * 100.0, 4),
        "rows": [
            {"name": "Price", "lv": _sf(mid), "bs": _sf(bs_price)},
            {"name": "Delta", "lv": _sf(lv_delta), "bs": _sf(bs["delta"])},
            {"name": "Gamma", "lv": _sf(lv_gamma), "bs": _sf(bs["gamma"])},
        ],
        "spot": [_sf(spot[s], 8) for s in ladder],
        "spot_pct": [_num((math.exp(s * dx) - 1.0) * 100.0, 3) for s in ladder],
        "iv_lv": [_num(v, 5) for v in ladder_iv],
        "unpriced": sum(v is None for v in ladder_iv),
        "clipped_nodes": at_bound, "grid_nodes": nodes,
        "clipped_share": _num(at_bound / nodes, 6) if nodes else 0.0,
        "clip_range": [_sf(clip_lo, 8), _sf(clip_hi, 8)],
        "bounds": [LOCAL_VOL_FLOOR, LOCAL_VOL_CAP],
    }
