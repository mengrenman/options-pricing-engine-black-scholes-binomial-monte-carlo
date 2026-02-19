"""Tests for Dupire local volatility extraction."""

import numpy as np
import pytest
from optpricer.calibration import (
    SVIParams, VolSurface, fit_svi, dupire_local_vol, dupire_local_vol_func,
)
from optpricer import OptionSpec, CALL, bs_price
from optpricer.pde import fd_price_local_vol
from optpricer.processes import local_vol_paths


def _flat_surface(flat_vol=0.2, forward=100.0):
    """Build a VolSurface with constant IV = flat_vol across all strikes."""
    slices = {}
    for T in [0.25, 0.5, 1.0]:
        # SVI params that produce flat vol: a = vol^2*T, b ≈ 0
        slices[T] = SVIParams(a=flat_vol**2 * T, b=1e-6, rho=0.0,
                              m=0.0, sigma=0.1, expiry=T)
    fwd = {T: forward for T in slices}
    return VolSurface(slices, forward_curve=fwd)


class TestDupireLocalVol:
    def test_flat_surface_gives_const_local_vol(self):
        """If BS IV is constant, Dupire local vol should equal that constant."""
        surface = _flat_surface(0.2)
        lv = dupire_local_vol(surface, 100.0, 0.5, 0.05, 0.0)
        assert abs(lv - 0.2) < 0.03, f"local vol = {lv}, expected ~0.2"

    def test_positive_local_vol(self):
        surface = _flat_surface(0.3)
        S_arr = np.linspace(80, 120, 20)
        lv = dupire_local_vol(surface, S_arr, 0.5, 0.05, 0.0)
        assert np.all(lv > 0), f"Found non-positive local vols: {lv}"

    def test_callable_interface(self):
        surface = _flat_surface(0.2)
        func = dupire_local_vol_func(surface, 0.05, 0.0)
        S_arr = np.array([90.0, 100.0, 110.0])
        result = func(S_arr, 0.5)
        assert result.shape == (3,)
        assert np.all(result > 0)


class TestDupireIntegration:
    def test_fd_with_constant_local_vol(self):
        """FD + explicitly constant σ(S,t)=0.2 should match BS."""
        sigma_const = lambda S, t: 0.2 * np.ones_like(S)
        lv_price = fd_price_local_vol(100, 100, 1.0, 0.05, 0.0, sigma_const,
                                       CALL, N_S=200, N_t=200, ref_vol=0.2)
        bs = bs_price(OptionSpec(S0=100, K=100, T=1.0, r=0.05, sigma=0.2), CALL)
        assert abs(lv_price - bs) / bs < 0.002, f"LV={lv_price:.4f} BS={bs:.4f}"

    def test_fd_with_dupire_flat_surface(self):
        """FD + Dupire from nearly-flat SVI surface: sensible price."""
        surface = _flat_surface(0.2)
        func = dupire_local_vol_func(surface, 0.05, 0.0)
        lv_price = fd_price_local_vol(100, 100, 1.0, 0.05, 0.0, func, CALL,
                                       N_S=200, N_t=200, ref_vol=0.2)
        bs = bs_price(OptionSpec(S0=100, K=100, T=1.0, r=0.05, sigma=0.2), CALL)
        # SVI approximation of a flat surface introduces some Dupire error
        assert abs(lv_price - bs) / bs < 0.10, f"LV={lv_price:.4f} BS={bs:.4f}"

    def test_mc_with_dupire_flat_surface(self):
        """MC + Dupire from nearly-flat SVI surface: sensible price."""
        surface = _flat_surface(0.2)
        func = dupire_local_vol_func(surface, 0.05, 0.0)
        paths = local_vol_paths(100, 0.05, 0.0, 1.0, 200, 50_000, func,
                                antithetic=True, seed=42)
        ST = paths[-1, :]
        payoff = np.maximum(ST - 100, 0.0)
        mc_price = float(np.exp(-0.05) * payoff.mean())
        bs = bs_price(OptionSpec(S0=100, K=100, T=1.0, r=0.05, sigma=0.2), CALL)
        assert abs(mc_price - bs) / bs < 0.10, f"MC={mc_price:.4f} BS={bs:.4f}"
