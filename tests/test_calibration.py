"""Tests for SVI calibration and VolSurface."""

import numpy as np
import pytest
from optpricer.calibration import SVIParams, VolSurface, fit_svi, fit_svi_surface


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
