"""Tests for the finite-difference PDE solver."""

import numpy as np
import pytest
from optpricer import OptionSpec, CALL, PUT, bs_price, bs_greeks
from optpricer.pde import fd_price, fd_price_barrier, fd_greeks, fd_price_local_vol

OPT = OptionSpec(S0=100, K=100, T=1.0, r=0.05, sigma=0.2)
N_S, N_t = 400, 400


class TestFDEuropean:
    def test_call_vs_bs(self):
        fd = fd_price(OPT, CALL, N_S=N_S, N_t=N_t)
        bs = bs_price(OPT, CALL)
        assert abs(fd - bs) / bs < 0.001, f"FD={fd:.6f} BS={bs:.6f}"

    def test_put_vs_bs(self):
        fd = fd_price(OPT, PUT, N_S=N_S, N_t=N_t)
        bs = bs_price(OPT, PUT)
        assert abs(fd - bs) / bs < 0.001, f"FD={fd:.6f} BS={bs:.6f}"

    def test_put_call_parity(self):
        c = fd_price(OPT, CALL, N_S=N_S, N_t=N_t)
        p = fd_price(OPT, PUT, N_S=N_S, N_t=N_t)
        parity = OPT.S0 * np.exp(-OPT.q * OPT.T) - OPT.K * np.exp(-OPT.r * OPT.T)
        assert abs((c - p) - parity) < 0.05

    def test_deep_itm_call(self):
        opt = OptionSpec(S0=150, K=100, T=1.0, r=0.05, sigma=0.2)
        fd = fd_price(opt, CALL, N_S=N_S, N_t=N_t)
        intrinsic = 150 - 100 * np.exp(-0.05)
        assert fd > intrinsic * 0.99

    def test_deep_otm_put(self):
        opt = OptionSpec(S0=150, K=100, T=1.0, r=0.05, sigma=0.2)
        fd = fd_price(opt, PUT, N_S=N_S, N_t=N_t)
        assert fd < 0.5  # nearly worthless


class TestFDAmerican:
    def test_american_put_geq_european(self):
        eu = fd_price(OPT, PUT, N_S=N_S, N_t=N_t, american=False)
        am = fd_price(OPT, PUT, N_S=N_S, N_t=N_t, american=True)
        assert am >= eu - 0.01

    def test_american_put_geq_intrinsic(self):
        am = fd_price(OPT, PUT, N_S=N_S, N_t=N_t, american=True)
        intrinsic = max(OPT.K - OPT.S0, 0.0)
        assert am >= intrinsic - 0.01

    def test_american_call_eq_european_no_div(self):
        """With q=0, American call = European call (no early exercise)."""
        eu = fd_price(OPT, CALL, N_S=N_S, N_t=N_t)
        am = fd_price(OPT, CALL, N_S=N_S, N_t=N_t, american=True)
        assert abs(am - eu) < 0.05


class TestFDBarrier:
    def test_knockout_leq_vanilla(self):
        opt = OptionSpec(S0=100, K=100, T=1.0, r=0.05, sigma=0.2)
        vanilla = fd_price(opt, CALL, N_S=N_S, N_t=N_t)
        ko = fd_price_barrier(opt, CALL, barrier=130.0,
                              barrier_type="up-and-out", N_S=N_S, N_t=N_t)
        assert ko <= vanilla + 0.01

    def test_in_out_parity(self):
        opt = OptionSpec(S0=100, K=100, T=1.0, r=0.05, sigma=0.2)
        vanilla = fd_price(opt, CALL, N_S=N_S, N_t=N_t)
        ko = fd_price_barrier(opt, CALL, barrier=130.0,
                              barrier_type="up-and-out", N_S=N_S, N_t=N_t)
        ki = fd_price_barrier(opt, CALL, barrier=130.0,
                              barrier_type="up-and-in", N_S=N_S, N_t=N_t)
        assert abs((ki + ko) - vanilla) < 0.1


class TestFDLocalVol:
    def test_constant_vol_matches_bs(self):
        sigma_const = lambda S, t: 0.2 * np.ones_like(S)
        lv = fd_price_local_vol(100, 100, 1.0, 0.05, 0.0, sigma_const, CALL,
                                N_S=N_S, N_t=N_t, ref_vol=0.2)
        bs = bs_price(OPT, CALL)
        assert abs(lv - bs) / bs < 0.002


class TestFDGreeks:
    def test_delta_vs_bs(self):
        fd_g = fd_greeks(OPT, CALL, N_S=N_S, N_t=N_t)
        bs_g = bs_greeks(OPT, CALL)
        assert abs(fd_g["delta"] - bs_g["delta"]) < 0.005

    def test_gamma_vs_bs(self):
        fd_g = fd_greeks(OPT, CALL, N_S=N_S, N_t=N_t)
        bs_g = bs_greeks(OPT, CALL)
        assert abs(fd_g["gamma"] - bs_g["gamma"]) < 0.002

    def test_theta_vs_bs(self):
        fd_g = fd_greeks(OPT, CALL, N_S=N_S, N_t=N_t)
        bs_g = bs_greeks(OPT, CALL)
        assert abs(fd_g["theta"] - bs_g["theta"]) / abs(bs_g["theta"]) < 0.01


class TestFDConvergence:
    def test_convergence_with_refinement(self):
        bs = bs_price(OPT, CALL)
        errors = []
        for n in [50, 100, 200]:
            fd = fd_price(OPT, CALL, N_S=n, N_t=n)
            errors.append(abs(fd - bs))
        # Error should decrease with refinement
        assert errors[1] < errors[0]
        assert errors[2] < errors[1]
