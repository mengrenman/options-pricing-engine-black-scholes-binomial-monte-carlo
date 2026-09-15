"""Tests for SVI calibration and VolSurface."""

import numpy as np
import pytest
from optpricer.calibration import (
    SVIParams, VolSurface, fit_svi, fit_svi_quasi, fit_svi_surface,
)


# ---------------------------------------------------------------------------
# SVIParams evaluation
# ---------------------------------------------------------------------------
class TestSVIParams:
    def test_total_var_at_money(self):
        p = SVIParams(a=0.04, b=0.1, rho=0.0, m=0.0, sigma=0.1, expiry=1.0)
        w = float(p.total_var(0.0))
        # w(0) = a + b * sqrt(sigma^2) = 0.04 + 0.1 * 0.1 = 0.05
        assert abs(w - 0.05) < 1e-10

    def test_iv_positive(self):
        p = SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.15, expiry=0.5)
        k = np.linspace(-0.5, 0.5, 20)
        ivs = p.iv(k)
        assert np.all(ivs > 0)

    def test_wings_increase(self):
        """Total variance should increase in the wings (b > 0)."""
        p = SVIParams(a=0.04, b=0.2, rho=0.0, m=0.0, sigma=0.1, expiry=1.0)
        w_left  = float(p.total_var(-1.0))
        w_atm   = float(p.total_var(0.0))
        w_right = float(p.total_var(1.0))
        assert w_left > w_atm
        assert w_right > w_atm


# ---------------------------------------------------------------------------
# SVI fitting — round-trip
# ---------------------------------------------------------------------------
class TestFitSVI:
    def test_zero_noise_recovery(self):
        """Fit SVI to data generated from known params — should recover exactly."""
        true = SVIParams(a=0.04, b=0.15, rho=-0.2, m=0.05, sigma=0.10, expiry=0.5)
        k = np.linspace(-0.4, 0.4, 30)
        ivs = true.iv(k)
        strikes = 100.0 * np.exp(k)

        fitted = fit_svi(strikes, forward=100.0, expiry=0.5, market_ivs=ivs)

        # Should recover params closely
        assert abs(fitted.a - true.a) < 0.005
        assert abs(fitted.b - true.b) < 0.01
        assert abs(fitted.rho - true.rho) < 0.05
        assert abs(fitted.m - true.m) < 0.05
        assert abs(fitted.sigma - true.sigma) < 0.01

    def test_noisy_fit_residuals(self):
        """With small noise, residuals should be small."""
        true = SVIParams(a=0.05, b=0.12, rho=-0.15, m=0.0, sigma=0.12, expiry=1.0)
        k = np.linspace(-0.3, 0.3, 20)
        ivs = true.iv(k) + np.random.default_rng(42).normal(0, 0.002, size=k.shape)
        strikes = 100.0 * np.exp(k)

        fitted = fit_svi(strikes, forward=100.0, expiry=1.0, market_ivs=ivs)

        # Fitted IVs should be close to market IVs
        fitted_ivs = fitted.iv(k)
        rmse = float(np.sqrt(np.mean((fitted_ivs - ivs) ** 2)))
        assert rmse < 0.005, f"RMSE too large: {rmse:.6f}"

    def test_butterfly_constraint(self):
        """Fitted total variance should be non-negative everywhere."""
        true = SVIParams(a=0.04, b=0.10, rho=-0.1, m=0.0, sigma=0.1, expiry=0.25)
        k = np.linspace(-0.3, 0.3, 20)
        strikes = 100.0 * np.exp(k)
        ivs = true.iv(k)

        fitted = fit_svi(strikes, forward=100.0, expiry=0.25, market_ivs=ivs)
        k_wide = np.linspace(-1.0, 1.0, 200)
        w = fitted.total_var(k_wide)
        assert np.all(w >= -1e-8), f"Negative total variance found: min={w.min():.6f}"


# ---------------------------------------------------------------------------
# VolSurface
# ---------------------------------------------------------------------------
class TestVolSurface:
    @pytest.fixture
    def surface(self):
        s1 = SVIParams(a=0.03, b=0.10, rho=-0.2, m=0.0, sigma=0.10, expiry=0.25)
        s2 = SVIParams(a=0.05, b=0.12, rho=-0.15, m=0.0, sigma=0.12, expiry=1.0)
        slices = {0.25: s1, 1.0: s2}
        fwds = {0.25: 100.0, 1.0: 100.0}
        return VolSurface(slices, forward_curve=fwds)

    def test_exact_expiry(self, surface):
        iv = surface.iv(100.0, 0.25)
        assert isinstance(iv, float)
        assert iv > 0

    def test_interpolated_expiry(self, surface):
        iv = surface.iv(100.0, 0.5)
        assert iv > 0

    def test_array_strikes(self, surface):
        Ks = np.array([90.0, 100.0, 110.0])
        ivs = surface.iv(Ks, 0.25)
        assert ivs.shape == (3,)
        assert np.all(ivs > 0)

    def test_extrapolation_short(self, surface):
        """Expiry before first slice — uses nearest."""
        iv = surface.iv(100.0, 0.1)
        assert iv > 0

    def test_extrapolation_long(self, surface):
        """Expiry after last slice — uses nearest."""
        iv = surface.iv(100.0, 2.0)
        assert iv > 0


# ---------------------------------------------------------------------------
# fit_svi_surface end-to-end
# ---------------------------------------------------------------------------
class TestFitSVISurface:
    def test_two_slice_surface(self):
        # Generate synthetic data for two expiries
        true_25 = SVIParams(a=0.03, b=0.10, rho=-0.2, m=0.0, sigma=0.1, expiry=0.25)
        true_1  = SVIParams(a=0.05, b=0.12, rho=-0.15, m=0.0, sigma=0.12, expiry=1.0)

        k = np.linspace(-0.3, 0.3, 15)
        strikes = 100.0 * np.exp(k)

        surface = fit_svi_surface(
            strikes_by_expiry={0.25: strikes, 1.0: strikes},
            forwards={0.25: 100.0, 1.0: 100.0},
            market_ivs_by_expiry={0.25: true_25.iv(k), 1.0: true_1.iv(k)},
        )

        assert isinstance(surface, VolSurface)
        assert len(surface.expiries) == 2
        # Check that surface produces reasonable IVs
        for T in [0.25, 0.5, 1.0]:
            iv = surface.iv(100.0, T)
            assert 0.05 < iv < 1.0, f"Unreasonable IV={iv} at T={T}"


# ---------------------------------------------------------------------------
# Quasi-explicit SVI fitting
# ---------------------------------------------------------------------------
def _svi_w(p, k):
    km = k - p.m
    return p.a + p.b * (p.rho * km + np.sqrt(km * km + p.sigma * p.sigma))


def _iv_rmse(p, k, T, ivs):
    """RMSE in implied-vol units between an SVI slice and quoted vols."""
    w = np.maximum(_svi_w(p, k), 1e-12)
    return float(np.sqrt(np.mean((np.sqrt(w / T) - ivs) ** 2)))


def _slice(a, b, rho, m, sigma, T, n=25, lo=60.0, hi=160.0, fwd=100.0):
    """Strikes / forward / ivs generated from known SVI params."""
    K = np.linspace(lo, hi, n)
    k = np.log(K / fwd)
    w = SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma, expiry=T).total_var(k)
    assert np.all(w > 0)
    return K, fwd, T, np.sqrt(w / T)


class TestFitSVIQuasi:
    def test_zero_noise_recovery(self):
        """Data generated from SVI should be recovered essentially exactly."""
        K, fwd, T, ivs = _slice(0.04, 0.15, -0.2, 0.05, 0.10, 0.5)
        p = fit_svi_quasi(K, fwd, T, ivs)
        k = np.log(K / fwd)
        assert np.max(np.abs(_svi_w(p, k) - ivs ** 2 * T)) < 1e-6

    def test_parameters_are_valid(self):
        """b > 0, |rho| <= 1, sigma > 0 and positive total variance."""
        for args in [(0.04, 0.15, -0.2, 0.05, 0.10, 0.5),
                     (0.02, 0.30, -0.7, -0.10, 0.25, 1.0),
                     (0.10, 0.05, 0.4, 0.20, 0.60, 2.0)]:
            K, fwd, T, ivs = _slice(*args)
            p = fit_svi_quasi(K, fwd, T, ivs)
            assert p.b > 0
            assert abs(p.rho) <= 0.999
            assert p.sigma > 0
            assert np.all(_svi_w(p, np.log(K / fwd)) > 0)

    def test_matches_fit_svi_accuracy(self):
        """On a misspecified (quadratic) smile it should be no worse than fit_svi."""
        K = np.linspace(70.0, 130.0, 25)
        fwd, T = 100.0 * np.exp(0.03), 1.0
        k = np.log(K / fwd)
        ivs = 0.20 + 0.12 * k ** 2 - 0.05 * k + 0.01

        rmse_q = _iv_rmse(fit_svi_quasi(K, fwd, T, ivs), k, T, ivs)
        rmse_t = _iv_rmse(fit_svi(K, fwd, T, ivs), k, T, ivs)
        assert rmse_q <= rmse_t + 1e-5

    def test_handles_sparse_quotes(self):
        K, fwd, T, ivs = _slice(0.03, 0.2, -0.4, 0.0, 0.2, 1.0, n=7)
        p = fit_svi_quasi(K, fwd, T, ivs)
        assert p.b > 0 and p.sigma > 0

    def test_surface_method_switch(self):
        K1, fwd1, T1, iv1 = _slice(0.03, 0.20, -0.30, 0.0, 0.20, 0.5)
        K2, fwd2, T2, iv2 = _slice(0.06, 0.25, -0.35, 0.0, 0.25, 1.0)
        strikes = {T1: K1, T2: K2}
        fwds = {T1: fwd1, T2: fwd2}
        ivs = {T1: iv1, T2: iv2}

        surf = fit_svi_surface(strikes, fwds, ivs, method="quasi")
        assert sorted(surf.slices.keys()) == sorted([T1, T2])
        assert np.all(surf.iv(K1, T1) > 0)

    def test_surface_rejects_unknown_method(self):
        K, fwd, T, ivs = _slice(0.03, 0.2, -0.3, 0.0, 0.2, 1.0)
        with pytest.raises(ValueError, match="method must be"):
            fit_svi_surface({T: K}, {T: fwd}, {T: ivs}, method="bogus")

    def test_no_worse_than_fit_svi_across_random_slices(self):
        """Regression: rho hitting its bound must not degrade the fit.

        Clamping rho after the fact (rather than bounding it inside the
        solve) leaves the other parameters stale and blows up the error on
        slices whose optimum sits on the boundary.
        """
        rng = np.random.default_rng(13)
        worst = 0.0
        for _ in range(40):
            n = int(rng.choice([7, 11, 25]))
            K = np.linspace(60.0, 150.0, n)
            k = np.log(K / 100.0)
            T = float(rng.choice([0.02, 0.25, 1.0, 5.0]))
            ivs = np.maximum(
                rng.uniform(0.12, 0.40)
                + rng.uniform(-0.3, 0.0) * k
                + rng.uniform(0.05, 0.5) * k * k,
                0.02,
            )

            worst = max(worst, _iv_rmse(fit_svi_quasi(K, 100.0, T, ivs), k, T, ivs)
                               - _iv_rmse(fit_svi(K, 100.0, T, ivs), k, T, ivs))
        assert worst < 1e-4, f"quasi fit degraded by {worst:.2e} IV RMSE"
