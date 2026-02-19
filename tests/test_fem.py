"""Tests for the finite-element solver."""

import pytest
from optpricer import OptionSpec, CALL, PUT, bs_price
from optpricer.fem import fem_price
from optpricer.pde import fd_price

OPT = OptionSpec(S0=100, K=100, T=1.0, r=0.05, sigma=0.2)
N_S, N_t = 400, 400


class TestFEMEuropean:
    def test_call_vs_bs(self):
        fe = fem_price(OPT, CALL, N_S=N_S, N_t=N_t)
        bs = bs_price(OPT, CALL)
        assert abs(fe - bs) / bs < 0.002, f"FEM={fe:.6f} BS={bs:.6f}"

    def test_put_vs_bs(self):
        fe = fem_price(OPT, PUT, N_S=N_S, N_t=N_t)
        bs = bs_price(OPT, PUT)
        assert abs(fe - bs) / bs < 0.002, f"FEM={fe:.6f} BS={bs:.6f}"

    def test_fem_vs_fdm(self):
        fe = fem_price(OPT, CALL, N_S=N_S, N_t=N_t)
        fd = fd_price(OPT, CALL, N_S=N_S, N_t=N_t)
        assert abs(fe - fd) < 0.05


class TestFEMConvergence:
    def test_convergence_order(self):
        bs = bs_price(OPT, CALL)
        errors = []
        for n in [50, 100, 200]:
            fe = fem_price(OPT, CALL, N_S=n, N_t=n)
            errors.append(abs(fe - bs))
        assert errors[1] < errors[0]
        assert errors[2] < errors[1]
