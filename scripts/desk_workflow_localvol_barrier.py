#!/usr/bin/env python3
"""Desk-style workflow: local-vol barrier pricing end-to-end.

Demonstrates a realistic quant-desk pipeline:

    synthetic vol quotes  →  SVI calibration  →  Dupire local vol
    →  barrier pricing (FDM + MC/Milstein)  →  Greeks  →  report

Usage
-----
    python scripts/desk_workflow_localvol_barrier.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optpricer.core import OptionSpec, CALL
from optpricer.calibration import (
    fit_svi_surface,
    dupire_local_vol_func,
)
from optpricer.pde import fd_price, fd_price_barrier, fd_price_local_vol, fd_greeks
from optpricer.processes import milstein_local_vol_paths
from optpricer.exotics import barrier_price
from optpricer.risk import numerical_greeks
from optpricer.black_scholes import price as bs_price


# ── helpers ────────────────────────────────────────────────────────────────
def _header(title: str) -> None:
    width = 68
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def _fmt(x: float, dp: int = 4) -> str:
    return f"{x:>{dp + 6}.{dp}f}"


# ── 1. Synthetic market data ──────────────────────────────────────────────
_header("Step 1 — Synthetic Market Data")

S0 = 100.0  # spot
r = 0.05  # risk-free rate
q = 0.02  # continuous dividend yield

# Three expiry slices with mild skew + convexity
expiries = [0.25, 0.50, 1.00]
forwards = {T: S0 * np.exp((r - q) * T) for T in expiries}

# Strikes around each forward ± 25 %
strikes_by_T: dict[float, np.ndarray] = {}
ivs_by_T: dict[float, np.ndarray] = {}

base_vol = 0.20
for T in expiries:
    F = forwards[T]
    K_arr = np.linspace(0.75 * F, 1.25 * F, 21)
    k = np.log(K_arr / F)
    # Synthetic skew: slight smile with T-dependent curvature
    iv = base_vol + 0.05 * k**2 - 0.02 * k + 0.005 * np.sqrt(T)
    strikes_by_T[T] = K_arr
    ivs_by_T[T] = iv

n_quotes = sum(len(v) for v in strikes_by_T.values())
print(f"Generated {n_quotes} synthetic quotes across {len(expiries)} expiries")
print(f"Spot: {S0}  |  Rate: {r}  |  Div yield: {q}  |  Base vol: {base_vol}")


# ── 2. SVI Calibration ───────────────────────────────────────────────────
_header("Step 2 — SVI Calibration")

t0 = time.perf_counter()
surface = fit_svi_surface(strikes_by_T, forwards, ivs_by_T)
t_cal = time.perf_counter() - t0

print(f"Calibrated SVI surface in {t_cal:.3f}s")
for T, svi in sorted(surface.slices.items()):
    k = np.log(strikes_by_T[T] / forwards[T])
    fitted = svi.iv(k)
    rmse = float(np.sqrt(np.mean((fitted - ivs_by_T[T]) ** 2)))
    print(f"  T={T:.2f}:  a={svi.a:.4f}  b={svi.b:.4f}  "
          f"rho={svi.rho:+.4f}  RMSE={rmse:.6f}")


# ── 3. Dupire Local Vol Extraction ───────────────────────────────────────
_header("Step 3 — Dupire Local Vol Surface")

sigma_loc = dupire_local_vol_func(surface, r=r, q=q)

# Probe the surface at a few (S, t) points
print(f"  {'S':>8s}  {'t':>6s}  {'σ_loc':>10s}")
for S_probe in [85.0, 100.0, 115.0]:
    for t_probe in [0.1, 0.5]:
        lv = sigma_loc(np.array([S_probe]), t_probe)[0]
        print(f"  {S_probe:>8.1f}  {t_probe:>6.2f}  {lv:>10.4f}")


# ── 4. Barrier Option Pricing ────────────────────────────────────────────
_header("Step 4 — Barrier Option Pricing (FDM + MC)")

K = 100.0  # strike
T = 1.0  # 1-year expiry
barrier = 130.0  # up-and-out barrier
barrier_type = "up-and-out"

opt = OptionSpec(S0=S0, K=K, T=T, r=r, sigma=base_vol, q=q)

# ---- 4a. FDM: vanilla European (constant vol) ----
t0 = time.perf_counter()
fdm_vanilla = fd_price(opt, CALL)
t_fdm_v = time.perf_counter() - t0

# ---- 4b. FDM: barrier (constant vol) ----
t0 = time.perf_counter()
fdm_barrier = fd_price_barrier(opt, CALL, barrier, barrier_type)
t_fdm_b = time.perf_counter() - t0

# ---- 4c. FDM: local vol vanilla ----
t0 = time.perf_counter()
fdm_lv_vanilla = fd_price_local_vol(S0, K, T, r, q, sigma_loc, CALL)
t_fdm_lv = time.perf_counter() - t0

# ---- 4d. BS benchmark ----
bs_vanilla = bs_price(opt, CALL)

# ---- 4e. MC + Milstein local vol: barrier ----
n_paths = 200_000
n_steps = 500
t0 = time.perf_counter()
paths = milstein_local_vol_paths(
    S0, r, q, T, n_steps, n_paths, sigma_loc, seed=42
)
mc_barrier, mc_se = barrier_price(paths, K, r, T, CALL, barrier, barrier_type)
t_mc = time.perf_counter() - t0

# ---- 4f. MC vanilla (from same paths, no barrier) ----
ST = paths[-1, :]
mc_vanilla = float(np.exp(-r * T) * np.maximum(ST - K, 0.0).mean())

print(f"\nContract:  S0={S0}  K={K}  T={T}  barrier={barrier} ({barrier_type})")
print(f"{'':>4s}{'Method':>25s} {'Vanilla':>10s} {'Barrier':>10s}  {'Time':>8s}")
print(f"{'':>4s}{'-' * 57}")
print(f"{'':>4s}{'Black-Scholes (const σ)':>25s} {_fmt(bs_vanilla):>10s} {'—':>10s}  {'—':>8s}")
print(f"{'':>4s}{'FDM (const σ)':>25s} {_fmt(fdm_vanilla):>10s} "
      f"{_fmt(fdm_barrier):>10s}  {t_fdm_v + t_fdm_b:>7.3f}s")
print(f"{'':>4s}{'FDM (local vol)':>25s} {_fmt(fdm_lv_vanilla):>10s} {'—':>10s}  {t_fdm_lv:>7.3f}s")
print(f"{'':>4s}{'MC+Milstein (local vol)':>25s} {_fmt(mc_vanilla):>10s} "
      f"{_fmt(mc_barrier):>10s}  {t_mc:>7.3f}s")
print(f"\n  MC barrier stderr: {mc_se:.4f}  ({n_paths:,} paths, {n_steps} steps)")


# ── 5. Greeks Comparison ─────────────────────────────────────────────────
_header("Step 5 — Greeks (FDM Grid vs Bump-and-Reprice)")

# FDM grid Greeks (constant vol)
fdm_gk = fd_greeks(opt, CALL)

# Bump-and-reprice using FDM as the engine
def _fdm_pricer(S, K, T, r, q, sigma, kind):
    spec = OptionSpec(S0=S, K=K, T=T, r=r, sigma=sigma, q=q)
    return fd_price(spec, kind)

bump_gk = numerical_greeks(_fdm_pricer, S0, K, T, r, q, base_vol, CALL)

print(f"\n{'Greek':>8s} {'FDM Grid':>12s} {'Bump&Reprice':>14s}")
print(f"{'─' * 36}")
for g in ("delta", "gamma", "theta"):
    print(f"{g:>8s} {fdm_gk[g]:>12.6f} {bump_gk[g]:>14.6f}")
for g in ("vega", "rho"):
    print(f"{g:>8s} {'—':>12s} {bump_gk[g]:>14.6f}")


# ── 6. Summary Report ────────────────────────────────────────────────────
_header("Step 6 — Summary")

knockdown = (1 - fdm_barrier / fdm_vanilla) * 100
lv_adj = fdm_lv_vanilla - fdm_vanilla

print(f"  Barrier knock-down:       {knockdown:.1f}% "
      f"(barrier {barrier_type} at {barrier})")
print(f"  Local-vol adjustment:    {lv_adj:+.4f} "
      f"({'+' if lv_adj > 0 else ''}{lv_adj / fdm_vanilla * 100:.2f}% of vanilla)")
print(f"  FDM vs MC barrier diff:   {abs(fdm_barrier - mc_barrier):.4f}")
print(f"  BS vs FDM vanilla diff:   {abs(bs_vanilla - fdm_vanilla):.4f}")
print()
