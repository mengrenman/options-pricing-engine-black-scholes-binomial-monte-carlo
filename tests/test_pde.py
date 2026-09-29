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


class TestFDGreeksNodeAndPrice:
    """fd_greeks reads its derivatives at the grid node nearest ln(S0) and returns the
    price of the same solve."""

    def test_keys_are_kept_and_price_is_added(self):
        g = fd_greeks(OPT, CALL, N_S=N_S, N_t=N_t)
        assert set(g) == {"delta", "gamma", "theta", "price"}

    def test_reproduced_off_center_case(self):
        """S0 = K = 10, sigma 0.3, T = 5: searchsorted picked the node above the center
        and gave a delta of 0.7771 against 0.7606 from Black-Scholes."""
        opt = OptionSpec(S0=10, K=10, T=5.0, r=0.05, sigma=0.3)
        g = fd_greeks(opt, CALL, N_S=400, N_t=400)
        assert abs(g["delta"] - bs_greeks(opt, CALL)["delta"]) < 1e-3

    @pytest.mark.parametrize("kind", [CALL, PUT])
    def test_atm_sweep_delta_stays_close_to_black_scholes(self, kind):
        """Across spots, vols and expiries with S0 = K, the old node choice was off by up
        to 0.017 in delta for some of the cases; the grid itself is good to well under 1e-3."""
        worst = 0.0
        for S0 in (7.5, 10.0, 20.0, 33.3, 50.0, 100.0, 123.45, 250.0, 1000.0):
            for sigma in (0.1, 0.2, 0.3, 0.5):
                for T in (0.25, 1.0, 2.0, 5.0):
                    if sigma * np.sqrt(T) > 4.0:
                        continue
                    opt = OptionSpec(S0=S0, K=S0, T=T, r=0.05, sigma=sigma)
                    g = fd_greeks(opt, kind, N_S=400, N_t=400)
                    worst = max(worst, abs(g["delta"] - bs_greeks(opt, kind)["delta"]))
        assert worst < 1e-3

    def test_derivatives_are_read_at_the_center_node(self):
        """An even N_S puts a node exactly on ln(S0), so delta and gamma must be the central
        differences at that node, whatever rounding does to the grid."""
        from optpricer.pde import _build_grid, _fd_solve

        n = 400
        for S0 in np.geomspace(5.0, 5000.0, 60):
            opt = OptionSpec(S0=S0, K=S0, T=5.0, r=0.05, sigma=0.3)
            x, dx, dt = _build_grid(S0, 5.0, 0.3, n, n, 4.0)
            V, _ = _fd_solve(x, dx, dt, n, S0, 0.05, 0.0, 0.3, CALL, 0.5, False,
                             return_two_layers=True)
            c = n // 2
            d1 = (V[c + 1] - V[c - 1]) / (2 * dx)
            d2 = (V[c + 1] - 2 * V[c] + V[c - 1]) / dx ** 2
            g = fd_greeks(opt, CALL, N_S=n, N_t=n)
            assert g["delta"] == pytest.approx(d1 / S0, rel=1e-12, abs=1e-14)
            assert g["gamma"] == pytest.approx((d2 - d1) / S0 ** 2, rel=1e-12, abs=1e-16)

    def test_odd_grid_still_works(self):
        """An odd N_S has no node on ln(S0); the nearest one is half a cell away."""
        g = fd_greeks(OPT, CALL, N_S=401, N_t=N_t)
        assert abs(g["delta"] - bs_greeks(OPT, CALL)["delta"]) < 0.01
        assert np.isfinite(g["gamma"]) and np.isfinite(g["price"])

    @pytest.mark.parametrize("kind", [CALL, PUT])
    @pytest.mark.parametrize("american", [False, True])
    def test_price_is_what_fd_price_gives_on_the_same_grid(self, kind, american):
        opt = OptionSpec(S0=100, K=95, T=0.75, r=0.03, sigma=0.25, q=0.02)
        kw = {"N_S": 300, "N_t": 250, "S_max_mult": 5.0, "american": american}
        assert fd_greeks(opt, kind, **kw)["price"] == fd_price(opt, kind, **kw)

    def test_price_on_the_default_grid_matches_fd_price(self):
        assert fd_greeks(OPT, PUT)["price"] == fd_price(OPT, PUT)

    def test_american_put_against_a_fine_tree(self):
        from optpricer import crr

        def tree(S0):
            o = OptionSpec(S0=S0, K=110, T=1.0, r=0.05, sigma=0.25)
            return crr(o, PUT, N=2000, american=True)

        opt = OptionSpec(S0=100, K=110, T=1.0, r=0.05, sigma=0.25)
        g = fd_greeks(opt, PUT, N_S=N_S, N_t=N_t, american=True)
        h = 0.5
        assert abs(g["price"] - tree(100.0)) < 5e-3
        assert abs(g["delta"] - (tree(100.0 + h) - tree(100.0 - h)) / (2 * h)) < 5e-3
        euro = fd_greeks(opt, PUT, N_S=N_S, N_t=N_t)
        assert g["price"] > euro["price"] + 0.1        # a real early-exercise premium
        assert g["delta"] < euro["delta"]              # exercising early pushes delta toward -1

    def test_american_call_without_dividend_matches_european(self):
        eu = fd_greeks(OPT, CALL, N_S=N_S, N_t=N_t)
        am = fd_greeks(OPT, CALL, N_S=N_S, N_t=N_t, american=True)
        assert abs(am["price"] - eu["price"]) < 1e-6
        assert abs(am["delta"] - eu["delta"]) < 1e-6


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


# ---------------------------------------------------------------------------
# Shared tridiagonal solver
# ---------------------------------------------------------------------------
class TestTridiagonalSolver:
    """The PDE/FEM inner loop: LAPACK banded solve, must match a dense solve."""

    @staticmethod
    def _dense(a, b, c):
        n = len(b)
        M = np.zeros((n, n))
        M[np.arange(n), np.arange(n)] = b
        M[np.arange(1, n), np.arange(n - 1)] = a[1:]
        M[np.arange(n - 1), np.arange(1, n)] = c[:-1]
        return M

    @pytest.mark.parametrize("n", [1, 2, 3, 10, 257])
    def test_matches_dense_solve(self, n):
        from optpricer._tridiag import solve_tridiagonal

        rng = np.random.default_rng(n)
        # diagonally dominant, as the theta-scheme systems are
        a = rng.uniform(-1.0, -0.5, n)
        c = rng.uniform(-1.0, -0.5, n)
        b = np.abs(a) + np.abs(c) + rng.uniform(0.5, 1.5, n)
        a[0] = 0.0
        c[-1] = 0.0
        d = rng.uniform(-1.0, 1.0, n)

        x = solve_tridiagonal(a, b, c, d)
        assert x.shape == (n,)
        ref = np.linalg.solve(self._dense(a, b, c), d)
        assert np.allclose(x, ref, rtol=1e-11, atol=1e-13)

    def test_residual_is_small(self):
        from optpricer._tridiag import solve_tridiagonal

        rng = np.random.default_rng(0)
        n = 400
        a = rng.uniform(-1.0, -0.5, n)
        c = rng.uniform(-1.0, -0.5, n)
        b = np.abs(a) + np.abs(c) + 1.0
        a[0] = 0.0
        c[-1] = 0.0
        d = rng.uniform(-1.0, 1.0, n)

        x = solve_tridiagonal(a, b, c, d)
        resid = self._dense(a, b, c) @ x - d
        assert np.max(np.abs(resid)) < 1e-10

    def test_inputs_are_not_mutated(self):
        """The engines reuse their coefficient arrays across time steps."""
        from optpricer._tridiag import solve_tridiagonal

        a = np.array([0.0, -1.0, -1.0])
        b = np.array([2.0, 2.0, 2.0])
        c = np.array([-1.0, -1.0, 0.0])
        d = np.array([1.0, 2.0, 3.0])
        before = [arr.copy() for arr in (a, b, c, d)]
        solve_tridiagonal(a, b, c, d)
        for arr, orig in zip((a, b, c, d), before):
            assert np.array_equal(arr, orig)

    def test_pde_and_fem_share_one_implementation(self):
        from optpricer import fem, pde
        from optpricer._tridiag import solve_tridiagonal

        assert pde._thomas_solve is solve_tridiagonal
        assert fem._thomas_solve is solve_tridiagonal
