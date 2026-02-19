"""Model validation framework.

Provides cross-model benchmarking, convergence analysis, stress-testing,
and delta-hedge backtesting — the core toolkit of a model-validation
quant (SR 11-7 style independent review and testing).
"""

from __future__ import annotations

import numpy as np
from typing import Optional
from dataclasses import replace

from .core import OptionSpec, CALL, PUT

__all__ = [
    "cross_validate",
    "convergence_analysis",
    "stress_test",
    "backtest_delta_hedge",
]


# ---------------------------------------------------------------------------
# Cross-model benchmarking
# ---------------------------------------------------------------------------

def cross_validate(
    opt: OptionSpec,
    kind: str = CALL,
    *,
    methods: Optional[list[str]] = None,
    mc_paths: int = 100_000,
    mc_seed: int = 42,
    tree_N: int = 500,
    fd_N_S: int = 200,
    fd_N_t: int = 200,
    fem_N_S: int = 200,
    fem_N_t: int = 200,
) -> dict:
    """Cross-validate option price across all available methods.

    Parameters
    ----------
    opt : OptionSpec
    kind : str
    methods : list of str, optional
        Subset of ``{"bs", "mc", "tree", "fdm", "fem"}``.  Default: all.

    Returns
    -------
    dict
        ``"bs"``, ``"mc"`` (price, stderr), ``"tree"``, ``"fdm"``, ``"fem"``,
        ``"max_discrepancy"`` (vs BS).
    """
    if methods is None:
        methods = ["bs", "mc", "tree", "fdm", "fem"]

    results: dict = {}

    if "bs" in methods:
        from .black_scholes import price as bs_price
        results["bs"] = bs_price(opt, kind)

    if "mc" in methods:
        from .monte_carlo import euro_price_mc
        p, se = euro_price_mc(opt, kind, n_paths=mc_paths, seed=mc_seed,
                              return_stderr=True)
        results["mc"] = (p, se)

    if "tree" in methods:
        from .binomial import crr
        results["tree"] = crr(opt, kind, N=tree_N)

    if "fdm" in methods:
        from .pde import fd_price
        results["fdm"] = fd_price(opt, kind, N_S=fd_N_S, N_t=fd_N_t)

    if "fem" in methods:
        from .fem import fem_price
        results["fem"] = fem_price(opt, kind, N_S=fem_N_S, N_t=fem_N_t)

    # Compute max discrepancy versus BS
    ref = results.get("bs")
    if ref is not None:
        discs = []
        for k, v in results.items():
            if k == "bs":
                continue
            p = v[0] if isinstance(v, tuple) else v
            discs.append(abs(p - ref))
        results["max_discrepancy"] = max(discs) if discs else 0.0
    else:
        results["max_discrepancy"] = float("nan")

    return results


# ---------------------------------------------------------------------------
# Convergence analysis
# ---------------------------------------------------------------------------

def convergence_analysis(
    opt: OptionSpec,
    kind: str,
    method: str,
    param_name: str,
    param_values: list | np.ndarray,
    *,
    reference: Optional[float] = None,
) -> dict:
    """Analyse convergence of a numerical method as a parameter varies.

    Parameters
    ----------
    method : str
        One of ``"mc"``, ``"tree"``, ``"fdm"``, ``"fem"``.
    param_name : str
        Parameter to vary: ``"n_paths"`` for MC, ``"N"`` for tree,
        ``"N_S"`` for FDM/FEM.
    param_values : array-like
        Values to test.
    reference : float, optional
        True price for error computation.  Default: BS analytical.

    Returns
    -------
    dict
        ``"params"``, ``"prices"``, ``"errors"``, ``"order"`` (estimated).
    """
    param_values = list(param_values)

    if reference is None:
        from .black_scholes import price as bs_price
        reference = bs_price(opt, kind)

    prices = []
    for val in param_values:
        val = int(val)
        if method == "mc":
            from .monte_carlo import euro_price_mc
            p = euro_price_mc(opt, kind, n_paths=val, seed=42,
                              return_stderr=False)
        elif method == "tree":
            from .binomial import crr
            p = crr(opt, kind, N=val)
        elif method == "fdm":
            from .pde import fd_price
            p = fd_price(opt, kind, N_S=val, N_t=val)
        elif method == "fem":
            from .fem import fem_price
            p = fem_price(opt, kind, N_S=val, N_t=val)
        else:
            raise ValueError(f"Unknown method: {method}")
        prices.append(float(p))

    errors = [abs(p - reference) for p in prices]

    # Estimate convergence order from log-log regression
    order = float("nan")
    valid = [(v, e) for v, e in zip(param_values, errors) if e > 0]
    if len(valid) >= 2:
        log_v = np.log([v for v, _ in valid])
        log_e = np.log([e for _, e in valid])
        # error ~ C / v^order  => log(e) = -order * log(v) + const
        coeffs = np.polyfit(log_v, log_e, 1)
        order = -float(coeffs[0])

    return {
        "params": param_values,
        "prices": prices,
        "errors": errors,
        "order": order,
    }


# ---------------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------------

def stress_test(
    opt: OptionSpec,
    kind: str,
    spot_shocks: np.ndarray,
    vol_shocks: np.ndarray,
    rate_shocks: np.ndarray,
    *,
    pricer: str = "bs",
) -> np.ndarray:
    """Evaluate option price across a 3-D grid of market shocks.

    Parameters
    ----------
    spot_shocks : array, shape (n_spot,)
        Multiplicative shocks to S0 (e.g. [0.8, 1.0, 1.2]).
    vol_shocks : array, shape (n_vol,)
        Additive shocks to sigma (e.g. [-0.05, 0, 0.05]).
    rate_shocks : array, shape (n_rate,)
        Additive shocks to r.
    pricer : str
        ``"bs"`` (default), ``"fdm"``, ``"tree"``.

    Returns
    -------
    ndarray, shape (n_spot, n_vol, n_rate)
    """
    spot_shocks = np.asarray(spot_shocks, dtype=float)
    vol_shocks = np.asarray(vol_shocks, dtype=float)
    rate_shocks = np.asarray(rate_shocks, dtype=float)

    if pricer == "bs":
        from .black_scholes import price as _price
    elif pricer == "fdm":
        from .pde import fd_price as _price_raw
        _price = lambda o, k: _price_raw(o, k)
    elif pricer == "tree":
        from .binomial import crr as _price
    else:
        raise ValueError(f"Unknown pricer: {pricer}")

    result = np.empty((len(spot_shocks), len(vol_shocks), len(rate_shocks)))

    for i, ds in enumerate(spot_shocks):
        for j, dv in enumerate(vol_shocks):
            new_sig = max(opt.sigma + dv, 1e-6)
            for k_idx, dr in enumerate(rate_shocks):
                shocked = replace(opt, S0=opt.S0 * ds, sigma=new_sig,
                                  r=opt.r + dr)
                result[i, j, k_idx] = _price(shocked, kind)

    return result


# ---------------------------------------------------------------------------
# Delta-hedge backtest
# ---------------------------------------------------------------------------

def backtest_delta_hedge(
    opt: OptionSpec,
    kind: str,
    paths: np.ndarray,
    rebalance_freq: int = 1,
    *,
    pricer: str = "bs",
) -> dict:
    """Simulate delta-hedging along pre-generated paths.

    Parameters
    ----------
    opt : OptionSpec
    kind : str
    paths : ndarray, shape (n_steps+1, n_paths)
        Asset price paths (from ``processes.py``).
    rebalance_freq : int
        Rebalance every N time steps (1 = every step).
    pricer : str
        ``"bs"`` (default) for delta computation.

    Returns
    -------
    dict
        ``"pnl"`` (ndarray), ``"mean_pnl"``, ``"std_pnl"``,
        ``"max_drawdown"``.
    """
    from .black_scholes import price as bs_price, greeks as bs_greeks

    n_steps = paths.shape[0] - 1
    n_paths = paths.shape[1]
    dt = opt.T / n_steps

    # Initial option value
    V0 = bs_price(opt, kind)

    # Track cash and shares for each path
    pnl = np.zeros(n_paths)

    # Delta at t=0
    g0 = bs_greeks(opt, kind)
    delta_prev = g0["delta"]

    # Initial position: short option, long delta shares
    cash = np.full(n_paths, V0 - delta_prev * opt.S0)
    shares = np.full(n_paths, delta_prev)

    for step in range(1, n_steps + 1):
        S_t = paths[step, :]
        tau = opt.T - step * dt

        # Accrue interest on cash
        cash *= np.exp(opt.r * dt)

        if tau > 1e-10 and step % rebalance_freq == 0:
            # Compute new delta
            new_opt = replace(opt, S0=1.0, T=tau)  # placeholder S0
            # Vectorise delta via BS vec
            from .black_scholes_vec import bs_greeks_vec
            greeks_now = bs_greeks_vec(S_t, opt.K, tau, opt.r, opt.q,
                                       opt.sigma, kind)
            delta_new = greeks_now["delta"]

            # Rebalance cost
            cash -= (delta_new - shares) * S_t
            shares = delta_new

    # At maturity: close position
    S_T = paths[-1, :]
    if kind == CALL:
        option_payoff = np.maximum(S_T - opt.K, 0.0)
    else:
        option_payoff = np.maximum(opt.K - S_T, 0.0)

    # P&L = final cash + shares * S_T - option payoff
    pnl = cash + shares * S_T - option_payoff

    return {
        "pnl": pnl,
        "mean_pnl": float(pnl.mean()),
        "std_pnl": float(pnl.std()),
        "max_drawdown": float(np.min(pnl)),
    }
