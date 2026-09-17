"""Tests for SVI calibration and VolSurface."""
import json
import warnings

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


# ---------------------------------------------------------------------------
# Fused derivative / total-variance helpers used by dupire_local_vol
# ---------------------------------------------------------------------------
class TestFusedHelpers:
    @staticmethod
    def _surface():
        Ts = [0.1, 0.5, 1.0, 2.0]
        Ks = np.linspace(70.0, 130.0, 21)
        fwd = {T: 100.0 * np.exp(0.03 * T) for T in Ts}
        ivs = {}
        for T in Ts:
            k = np.log(Ks / fwd[T])
            ivs[T] = 0.20 + 0.10 * k * k - 0.05 * k + 0.01 * np.sqrt(T)
        return fit_svi_surface({T: Ks for T in Ts}, fwd, ivs)

    def test_w_dw_d2w_matches_separate_methods(self):
        p = SVIParams(a=0.04, b=0.15, rho=-0.3, m=0.05, sigma=0.12, expiry=1.0)
        k = np.linspace(-1.2, 1.2, 101)
        w, dw, d2w = p.w_dw_d2w(k)
        assert np.allclose(w, p.total_var(k), rtol=0, atol=0)
        assert np.allclose(dw, p.dw_dk(k), rtol=1e-15, atol=1e-16)
        assert np.allclose(d2w, p.d2w_dk2(k), rtol=1e-12, atol=1e-16)

    @pytest.mark.parametrize("T", [0.02, 0.1, 0.3, 1.0, 2.0, 5.0])
    def test_total_var_from_logm_matches_iv_round_trip(self, T):
        """Must equal iv_from_logm(k, T)**2 * T on every branch.

        Regression: the exact-match and extrapolation branches of
        iv_from_logm divide by the *slice* expiry, not by T.  Dropping that
        ratio silently changed local vol outside the quoted tenor range.
        """
        surf = self._surface()
        k = np.linspace(-0.6, 0.6, 33)
        direct = surf.total_var_from_logm(k, T)
        round_trip = surf.iv_from_logm(k, T) ** 2 * T
        assert np.allclose(direct, round_trip, rtol=1e-12, atol=1e-15)

    def test_dupire_unchanged_outside_quoted_tenors(self):
        """Extrapolated tenors are where the scaling bug showed up."""
        from optpricer.calibration import dupire_local_vol

        surf = self._surface()
        S = np.linspace(60.0, 150.0, 9)
        for t in (0.01, 3.0):
            v = np.atleast_1d(dupire_local_vol(surf, S, t, 0.03, 0.0))
            assert np.all(np.isfinite(v))
            assert np.all(v > 0.01)      # not pinned to the clip floor
            assert np.all(v < 5.0)


# ---------------------------------------------------------------------------
# Time interpolation of the vol surface
# ---------------------------------------------------------------------------
class TestSurfaceTimeInterpolation:
    """Regression: iv_from_logm scaled SVIParams.total_var by the slice expiry,
    but total_var already returns total variance (sigma^2 * T).  Time was
    counted twice, so interpolated vols were badly wrong (ATM 0.093 where the
    neighbouring slices were 0.203 and 0.205) and the surface jumped at every
    quoted expiry."""

    @staticmethod
    def _flat_vol_surface(vol=0.2, T1=0.5, T2=2.0):
        """Two slices carrying the SAME flat vol, so the surface is vol-flat
        in T and every interpolated value must come back as `vol`."""
        s1 = SVIParams(a=vol ** 2 * T1, b=0.0, rho=0.0, m=0.0, sigma=0.1, expiry=T1)
        s2 = SVIParams(a=vol ** 2 * T2, b=0.0, rho=0.0, m=0.0, sigma=0.1, expiry=T2)
        return VolSurface({T1: s1, T2: s2}, forward_curve={T1: 100.0, T2: 100.0})

    @pytest.mark.parametrize("T", [0.5, 0.75, 1.0, 1.5, 2.0])
    def test_flat_vol_stays_flat(self, T):
        surf = self._flat_vol_surface(vol=0.2)
        k = np.linspace(-0.3, 0.3, 11)
        assert np.allclose(surf.iv_from_logm(k, T), 0.2, rtol=1e-10)

    def test_interpolated_vol_lies_between_neighbours(self):
        surf = self._surface_two_slices()
        k = np.array([0.0])
        lo = float(np.atleast_1d(surf.slices[0.25].iv(k))[0])
        hi = float(np.atleast_1d(surf.slices[1.0].iv(k))[0])
        for T in (0.4, 0.6, 0.8):
            mid = float(np.atleast_1d(surf.iv_from_logm(k, T))[0])
            assert min(lo, hi) <= mid <= max(lo, hi), f"T={T}: {mid} outside [{lo}, {hi}]"

    def test_continuous_at_quoted_expiries(self):
        surf = self._surface_two_slices()
        k = np.array([0.0])
        for E in (0.25, 1.0):
            below = float(np.atleast_1d(surf.iv_from_logm(k, E - 1e-7))[0])
            at = float(np.atleast_1d(surf.iv_from_logm(k, E))[0])
            above = float(np.atleast_1d(surf.iv_from_logm(k, E + 1e-7))[0])
            assert abs(at - below) < 1e-5
            assert abs(at - above) < 1e-5

    def test_quoted_expiry_returns_the_slice_itself(self):
        surf = self._surface_two_slices()
        k = np.linspace(-0.3, 0.3, 11)
        for E in (0.25, 1.0):
            assert np.allclose(surf.iv_from_logm(k, E), surf.slices[E].iv(k), rtol=1e-12)

    def test_public_iv_uses_the_same_interpolation(self):
        surf = self._flat_vol_surface(vol=0.25)
        assert float(np.atleast_1d(surf.iv(100.0, 1.0))[0]) == pytest.approx(0.25, rel=1e-8)

    @staticmethod
    def _surface_two_slices():
        s1 = SVIParams(a=0.03, b=0.10, rho=-0.2, m=0.0, sigma=0.10, expiry=0.25)
        s2 = SVIParams(a=0.05, b=0.12, rho=-0.15, m=0.0, sigma=0.12, expiry=1.0)
        return VolSurface({0.25: s1, 1.0: s2}, forward_curve={0.25: 100.0, 1.0: 100.0})


# ---------------------------------------------------------------------------
# Forward construction: r and q must actually do something
# ---------------------------------------------------------------------------
class TestForwardCarry:
    """Regression: dupire_local_vol declared r and q and read neither, and
    _get_forward flat-extrapolated outside the quoted range (with a 2y last
    quote and 3% carry, T=5 returned the 2y forward, 106.18 vs 116.18)."""

    @staticmethod
    def _surface(r=0.03):
        Ts = [0.1, 0.25, 0.5, 1.0, 2.0]
        Ks = np.linspace(70.0, 130.0, 21)
        fwd = {T: 100.0 * np.exp(r * T) for T in Ts}
        ivs = {}
        for T in Ts:
            k = np.log(Ks / fwd[T])
            ivs[T] = 0.20 + 0.10 * k * k - 0.05 * k + 0.01 * np.sqrt(T)
        return fit_svi_surface({T: Ks for T in Ts}, fwd, ivs)

    def test_forward_carries_beyond_the_quoted_range(self):
        surf = self._surface(r=0.03)
        for T in (3.0, 5.0, 10.0):
            assert surf.forward_at(T, 0.03, 0.0) == pytest.approx(100.0 * np.exp(0.03 * T), rel=1e-12)
            # the old flat behaviour is still available and is wrong out here
            assert surf._get_forward(T) == pytest.approx(100.0 * np.exp(0.03 * 2.0), rel=1e-12)

    def test_forward_unchanged_inside_the_quoted_range(self):
        surf = self._surface()
        for T in (0.1, 0.4, 1.0, 2.0):
            assert surf.forward_at(T, 0.03, 0.0) == pytest.approx(surf._get_forward(T), rel=1e-12)

    def test_rate_affects_local_vol_beyond_the_curve(self):
        from optpricer.calibration import dupire_local_vol

        surf = self._surface()
        S = np.linspace(80.0, 120.0, 5)
        lo = np.atleast_1d(dupire_local_vol(surf, S, 3.0, 0.03, 0.0))
        hi = np.atleast_1d(dupire_local_vol(surf, S, 3.0, 0.09, 0.0))
        assert np.max(np.abs(lo - hi)) > 1e-4, "r must reach the forward outside the curve"

    def test_rate_does_not_affect_local_vol_inside_the_curve(self):
        from optpricer.calibration import dupire_local_vol

        surf = self._surface()
        S = np.linspace(80.0, 120.0, 5)
        a = np.atleast_1d(dupire_local_vol(surf, S, 0.4, 0.03, 0.0))
        b = np.atleast_1d(dupire_local_vol(surf, S, 0.4, 9.00, 0.0))
        assert np.allclose(a, b, rtol=0, atol=0), "inside the curve the forward comes from the curve"

    def test_missing_forward_curve_raises_instead_of_guessing(self):
        """No curve and no spot means no forward, so refuse rather than guess.

        The original code used mean(S) as the "forward", which made local vol
        depend on which grid it happened to be asked about.
        """
        from optpricer.calibration import dupire_local_vol

        bare = VolSurface(self._surface().slices)
        with pytest.raises(ValueError, match="no forward curve and no spot"):
            dupire_local_vol(bare, np.linspace(80.0, 120.0, 5), 0.4, 0.03, 0.0)

    def test_spot_reproduces_the_forward_curve_answer(self):
        """spot=S0 agrees with an explicit curve, exactly at the quoted nodes.

        Between nodes the two differ at ~1e-6 and must: ``_get_forward`` blends
        the quoted forwards *linearly* in T, while ``spot*exp((r-q)t)`` is
        convex, so they can only coincide where the curve is pinned.
        """
        from optpricer.calibration import dupire_local_vol

        r, q, S0 = 0.03, 0.0, 100.0
        curved = self._surface(r=r)                       # forwards are S0*exp(rT)
        bare = VolSurface(curved.slices)
        S = np.linspace(80.0, 120.0, 9)

        for t in (0.25, 1.0, 2.0):                        # quoted expiries
            a = np.atleast_1d(dupire_local_vol(curved, S, t, r, q))
            b = np.atleast_1d(dupire_local_vol(bare, S, t, r, q, spot=S0))
            assert np.allclose(a, b, rtol=1e-12, atol=1e-14)

        for t in (0.15, 0.4):                             # between nodes
            a = np.atleast_1d(dupire_local_vol(curved, S, t, r, q))
            b = np.atleast_1d(dupire_local_vol(bare, S, t, r, q, spot=S0))
            assert np.allclose(a, b, rtol=1e-4, atol=0.0)
            assert not np.array_equal(a, b)

    def test_grid_independence(self):
        """Local vol at one point must not depend on the grid around it.

        The mean(S) fallback failed exactly this: the same (S, t) returned a
        different number depending on which strikes were evaluated with it.
        """
        from optpricer.calibration import dupire_local_vol

        surf = self._surface()
        narrow = np.array([99.0, 100.0, 101.0])
        wide = np.array([50.0, 100.0, 300.0])
        a = np.atleast_1d(dupire_local_vol(surf, narrow, 0.4, 0.03, 0.0))[1]
        b = np.atleast_1d(dupire_local_vol(surf, wide, 0.4, 0.03, 0.0))[1]
        assert a == pytest.approx(b, rel=1e-12)

    def test_callable_factory_forwards_spot(self):
        from optpricer.calibration import dupire_local_vol_func

        bare = VolSurface(self._surface().slices)
        S = np.linspace(90.0, 110.0, 5)
        with pytest.raises(ValueError, match="no forward curve and no spot"):
            dupire_local_vol_func(bare, 0.03, 0.0)(S, 0.4)
        assert np.all(np.isfinite(dupire_local_vol_func(bare, 0.03, 0.0, spot=100.0)(S, 0.4)))


# ---------------------------------------------------------------------------
# Dupire denominator must be read from the surface at t
# ---------------------------------------------------------------------------
class TestSpatialDerivativesAtT:
    """Regression: the denominator took w, dw/dk, d2w/dk2 from one bracketing
    slice at that slice's own expiry while the numerator used the interpolated
    surface, so the two disagreed (w inflated 1.68x at t=0.15)."""

    def _surf(self):
        return TestForwardCarry._surface()

    @pytest.mark.parametrize("T", [0.05, 0.1, 0.175, 0.25, 0.4, 1.0, 1.5, 2.0, 3.0])
    def test_w_matches_the_surface_exactly(self, T):
        surf = self._surf()
        k = np.linspace(-0.5, 0.5, 51)
        w, _, _ = surf.w_dw_d2w_from_logm(k, T)
        assert np.allclose(w, surf.total_var_from_logm(k, T), rtol=0, atol=0)

    @pytest.mark.parametrize("T", [0.175, 0.4, 1.5, 3.0])
    def test_k_derivatives_match_finite_differences(self, T):
        surf = self._surf()
        k = np.linspace(-0.4, 0.4, 81)
        h = 1e-5
        _, dw, d2w = surf.w_dw_d2w_from_logm(k, T)
        up = surf.total_var_from_logm(k + h, T)
        dn = surf.total_var_from_logm(k - h, T)
        mid = surf.total_var_from_logm(k, T)
        assert np.allclose(dw, (up - dn) / (2 * h), rtol=1e-6, atol=1e-9)
        assert np.allclose(d2w, (up - 2 * mid + dn) / h ** 2, rtol=1e-3, atol=1e-5)

    def test_at_a_quoted_expiry_it_is_the_slice_itself(self):
        surf = self._surf()
        k = np.linspace(-0.4, 0.4, 41)
        for T in (0.25, 1.0):
            got = surf.w_dw_d2w_from_logm(k, T)
            ref = surf.slices[T].w_dw_d2w(k)
            for g, r_ in zip(got, ref):
                assert np.allclose(g, r_, rtol=1e-12, atol=1e-15)

    def test_local_vol_unchanged_at_quoted_expiries(self):
        """The old slice pick happened to be right exactly on a quoted expiry,
        so those values must not move; only interpolated ones should."""
        from optpricer.calibration import dupire_local_vol

        surf = self._surf()
        S = np.linspace(80.0, 120.0, 9)
        for T in (0.25, 1.0):
            v = np.atleast_1d(dupire_local_vol(surf, S, T, 0.03, 0.0))
            assert np.all(np.isfinite(v)) and np.all(v > 0.01)


# ---------------------------------------------------------------------------
# Calibration input validation
# ---------------------------------------------------------------------------
class TestCalibrationInputValidation:
    """Regression: one NaN implied vol made fit_svi_quasi return
    SVIParams(a=nan, b=nan, ...) silently, and made fit_svi raise an unrelated
    'Initial guess is outside of provided bounds'."""

    @staticmethod
    def _slice():
        K = np.linspace(70.0, 130.0, 21)
        F = 100.0 * np.exp(0.03)
        return K, F, 1.0, 0.20 + 0.10 * np.log(K / F) ** 2 - 0.05 * np.log(K / F)

    @pytest.mark.parametrize("fitter", [fit_svi, fit_svi_quasi])
    def test_bad_quotes_are_dropped_not_propagated(self, fitter):
        K, F, T, ivs = self._slice()
        dirty = ivs.copy()
        dirty[3] = np.nan
        dirty[7] = 0.0
        dirty[11] = -1.0
        with pytest.warns(RuntimeWarning, match="dropping 3 of 21"):
            p = fitter(K, F, T, dirty)
        for v in (p.a, p.b, p.rho, p.m, p.sigma):
            assert np.isfinite(v)

    @pytest.mark.parametrize("fitter", [fit_svi, fit_svi_quasi])
    def test_underdetermined_fit_raises(self, fitter):
        K, F, T, ivs = self._slice()
        with pytest.raises(ValueError, match="underdetermined"):
            fitter(K[:4], F, T, ivs[:4])

    @pytest.mark.parametrize("fitter", [fit_svi, fit_svi_quasi])
    def test_bad_scalars_raise(self, fitter):
        K, F, T, ivs = self._slice()
        with pytest.raises(ValueError, match="forward"):
            fitter(K, -1.0, T, ivs)
        with pytest.raises(ValueError, match="expiry"):
            fitter(K, F, 0.0, ivs)

    @pytest.mark.parametrize("fitter", [fit_svi, fit_svi_quasi])
    def test_length_mismatch_raises(self, fitter):
        K, F, T, ivs = self._slice()
        with pytest.raises(ValueError, match="same length"):
            fitter(K, F, T, ivs[:-1])


# ---------------------------------------------------------------------------
# Expiry keys
# ---------------------------------------------------------------------------
class TestExpiryKeys:
    """Two roots expiring on one date (SPX and SPXW) map to the same float key.
    That collapse happens in the caller's dict, so the library cannot see it —
    these are the guards that ARE possible."""

    @staticmethod
    def _slice():
        K = np.linspace(70.0, 130.0, 21)
        F = 100.0 * np.exp(0.03)
        return K, F, 0.20 + 0.10 * np.log(K / F) ** 2 - 0.05 * np.log(K / F)

    def test_mismatched_dicts_raise_a_clear_error(self):
        K, F, ivs = self._slice()
        with pytest.raises(ValueError, match="same expiries"):
            fit_svi_surface({0.5: K}, {1.0: F}, {0.5: ivs})

    def test_near_duplicate_expiries_warn(self):
        K, F, ivs = self._slice()
        T1, T2 = 1.0, 1.0 + 1e-13
        with pytest.warns(RuntimeWarning, match="float noise"):
            fit_svi_surface({T1: K, T2: K}, {T1: F, T2: F}, {T1: ivs, T2: ivs})

    def test_non_positive_expiry_rejected(self):
        p = SVIParams(a=0.04, b=0.1, rho=0.0, m=0.0, sigma=0.1, expiry=1.0)
        with pytest.raises(ValueError, match="finite and positive"):
            VolSurface({0.0: p})
        with pytest.raises(ValueError, match="finite and positive"):
            VolSurface({-1.0: p})

    def test_label_distinguishes_two_roots(self):
        p = SVIParams(a=0.04, b=0.1, rho=0.0, m=0.0, sigma=0.1, expiry=1.0)
        assert VolSurface({1.0: p}).label is None
        assert VolSurface({1.0: p}, label="SPXW").label == "SPXW"


# ---------------------------------------------------------------------------
# Clamp consistency between the surface value and its k-derivatives
# ---------------------------------------------------------------------------
class TestClampConsistency:
    """Regression: w_dw_d2w_from_logm returned the raw blend on its
    interpolation branch while total_var_from_logm clamped at zero, so on a
    surface whose interpolated total variance goes negative the two disagreed
    (-0.0267 vs 0.0). dupire_local_vol then floored w at 1e-12 but left dw
    live, so (k/w)*dw blew up to 2.8e+05 and sigma_loc was silently pinned at
    the 0.01 floor on 6 of 7 nodes."""

    @staticmethod
    def _degenerate():
        """Two slices whose linear blend in total variance dips below zero."""
        lo = SVIParams(a=-0.05, b=1e-6, rho=0.0, m=0.0, sigma=0.1, expiry=0.5)
        hi = SVIParams(a=0.30, b=1e-6, rho=0.0, m=0.0, sigma=0.1, expiry=2.0)
        return VolSurface({0.5: lo, 2.0: hi},
                          forward_curve={0.5: 100.0, 2.0: 100.0})

    @pytest.mark.parametrize("T", [0.55, 0.6, 0.7, 0.8, 1.0, 1.5])
    def test_value_and_derivatives_describe_one_surface(self, T):
        surf = self._degenerate()
        k = np.linspace(-0.3, 0.3, 13)
        w, _, _ = surf.w_dw_d2w_from_logm(k, T)
        assert np.array_equal(w, surf.total_var_from_logm(k, T))
        assert np.all(w >= 0.0)

    def test_derivatives_vanish_where_the_surface_is_clamped(self):
        """max(blend, 0) is flat where the blend is negative, so both
        derivatives are zero there -- not merely small."""
        surf = self._degenerate()
        k = np.linspace(-0.3, 0.3, 13)
        w, dw, d2w = surf.w_dw_d2w_from_logm(k, 0.6)
        clamped = w <= 0.0
        assert clamped.any(), "fixture must exercise the clamp"
        assert np.all(dw[clamped] == 0.0)
        assert np.all(d2w[clamped] == 0.0)

    def test_no_blow_up_in_the_dupire_denominator(self):
        surf = self._degenerate()
        k = np.linspace(-0.3, 0.3, 13)
        w, dw, _ = surf.w_dw_d2w_from_logm(k, 0.6)
        assert np.max(np.abs((k / np.maximum(w, 1e-12)) * dw)) < 1e3

    def test_degenerate_surface_warns_rather_than_silently_flooring(self):
        from optpricer.calibration import dupire_local_vol

        surf = self._degenerate()
        with pytest.warns(RuntimeWarning, match="non-positive total variance"):
            dupire_local_vol(surf, np.linspace(80.0, 120.0, 7), 0.6, 0.03, 0.0)

    def test_warning_text_is_constant_so_it_dedups(self):
        """An earlier warning interpolated node values into the message, so the
        duplicate filter could not collapse it and a 200-step solve emitted 200
        warnings."""
        from optpricer.calibration import dupire_local_vol

        surf = self._degenerate()
        S = np.linspace(80.0, 120.0, 7)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")
            for _ in range(25):
                dupire_local_vol(surf, S, 0.6, 0.03, 0.0)
        assert len(caught) == 1, f"expected dedup to 1, got {len(caught)}"

    def test_healthy_surface_is_untouched(self):
        from optpricer.calibration import dupire_local_vol

        surf = TestForwardCarry._surface()
        S = np.linspace(80.0, 120.0, 9)
        with warnings.catch_warnings():
            warnings.simplefilter("error")          # any warning fails the test
            v = np.atleast_1d(dupire_local_vol(surf, S, 0.4, 0.03, 0.0))
        assert np.all(np.isfinite(v)) and np.all(v > 0.01)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------
class TestSerialisation:
    @staticmethod
    def _surface(label="SPX"):
        Ts = [0.1, 0.25, 0.5, 1.0, 2.0]
        Ks = np.linspace(70.0, 130.0, 21)
        fwd = {T: 100.0 * np.exp(0.03 * T) for T in Ts}
        ivs = {}
        for T in Ts:
            k = np.log(Ks / fwd[T])
            ivs[T] = 0.20 + 0.10 * k * k - 0.05 * k + 0.01 * np.sqrt(T)
        s = fit_svi_surface({T: Ks for T in Ts}, fwd, ivs)
        s.label = label
        return s

    # --- round trip -------------------------------------------------------
    def test_restored_surface_behaves_identically(self):
        """Field equality is not the bar; the restored surface must PRICE the same."""
        from optpricer.calibration import dupire_local_vol

        orig = self._surface()
        back = VolSurface.from_json(orig.to_json())
        k = np.linspace(-0.6, 0.6, 97)
        S = np.linspace(60.0, 160.0, 97)

        for T in (0.05, 0.1, 0.175, 0.4, 1.0, 2.0, 3.0):
            assert np.array_equal(orig.iv_from_logm(k, T), back.iv_from_logm(k, T))
            assert np.array_equal(orig.total_var_from_logm(k, T), back.total_var_from_logm(k, T))
            assert np.array_equal(orig.dw_dT_from_logm(k, T), back.dw_dT_from_logm(k, T))
            for a, b in zip(orig.w_dw_d2w_from_logm(k, T), back.w_dw_d2w_from_logm(k, T)):
                assert np.array_equal(a, b)
        for t in (0.15, 0.4, 1.0, 2.5):
            assert np.array_equal(
                np.atleast_1d(dupire_local_vol(orig, S, t, 0.03, 0.0)),
                np.atleast_1d(dupire_local_vol(back, S, t, 0.03, 0.0)),
            )
            assert orig.forward_at(t, 0.03, 0.0) == back.forward_at(t, 0.03, 0.0)

    def test_round_trip_is_idempotent(self):
        orig = self._surface()
        once = orig.to_json()
        assert VolSurface.from_json(once).to_json() == once

    def test_output_is_strict_json(self):
        """json.dumps emits bare NaN by default, which other parsers reject."""
        text = self._surface().to_json()

        def _boom(token):
            raise AssertionError(f"non-standard literal {token!r} in output")

        json.loads(text, parse_constant=_boom)

    @pytest.mark.parametrize("label", ["SPX", "SPXW", None])
    def test_label_round_trips(self, label):
        s = self._surface(label=label)
        assert VolSurface.from_json(s.to_json()).label == label

    def test_empty_forward_curve_round_trips(self):
        bare = VolSurface(self._surface().slices)
        back = VolSurface.from_json(bare.to_json())
        assert back._forward_curve == {}

    def test_svi_params_round_trip(self):
        p = SVIParams(a=-0.0234, b=0.1415, rho=-0.9265, m=0.3589, sigma=0.7932, expiry=1 / 365)
        q = SVIParams.from_dict(p.to_dict())
        for f in ("a", "b", "rho", "m", "sigma", "expiry"):
            assert getattr(q, f) == getattr(p, f), f

    # --- why a list, not an object ---------------------------------------
    def test_duplicate_expiry_in_a_list_is_caught(self):
        """The reason slices are a list: json.loads keeps only the LAST of two
        identical object keys, and "0.25", "0.250" and "2.5e-1" all parse to the
        same float. Both merges happen before this class sees the payload. A
        list keeps duplicates so they can be rejected."""
        d = self._surface().to_dict()
        with pytest.raises(ValueError, match="Duplicate expiry"):
            VolSurface.from_dict({**d, "slices": d["slices"] + [d["slices"][0]]})

    def test_json_object_keys_give_a_clear_error(self):
        s = self._surface()
        with pytest.raises(ValueError, match="Expiries must be numbers"):
            VolSurface({"0.1": s.slices[0.1]})

    def test_slice_key_and_expiry_must_agree(self):
        """iv_from_logm's exact-match branch divides by the slice's own expiry
        while the interpolation branch scales by T/slice.expiry, so a mismatch
        is a wrong-number bug."""
        s = VolSurface({0.5: SVIParams(a=0.04, b=0.1, rho=0.0, m=0.0, sigma=0.1, expiry=1.0)})
        with pytest.raises(ValueError, match="carries expiry"):
            s.to_dict()

    # --- refusing bad payloads -------------------------------------------
    def test_non_finite_never_serialises(self):
        """Guard lives in to_dict, not only to_json, so json.dumps(s.to_dict())
        cannot write a file to_json would have refused."""
        s = VolSurface({1.0: SVIParams(a=np.nan, b=0.1, rho=0.0, m=0.0, sigma=0.1, expiry=1.0)})
        with pytest.raises(ValueError, match="must be finite"):
            s.to_dict()

    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
    def test_non_standard_literals_are_refused_on_load(self, literal):
        payload = (
            '{"schema":"optpricer.volsurface","version":1,"label":null,"slices":'
            f'[{{"a":{literal},"b":0.1,"rho":0.0,"m":0.0,"sigma":0.1,"expiry":1.0}}],'
            '"forward_curve":[]}'
        )
        with pytest.raises(ValueError, match="Refusing to load"):
            VolSurface.from_json(payload)

    @pytest.mark.parametrize("value", ["1e400", "-1e400"])
    def test_overflow_to_infinity_is_caught(self, value):
        """1e400 is a legal JSON number, so parse_constant never fires on it;
        the finiteness check on every value is what catches it."""
        payload = (
            '{"schema":"optpricer.volsurface","version":1,"label":null,"slices":'
            f'[{{"a":{value},"b":0.1,"rho":0.0,"m":0.0,"sigma":0.1,"expiry":1.0}}],'
            '"forward_curve":[]}'
        )
        with pytest.raises(ValueError, match="must be finite"):
            VolSurface.from_json(payload)

    def test_underflow_to_zero_is_allowed(self):
        """1e-400 underflows to 0.0, which is finite and a legitimate value."""
        payload = ('{"schema":"optpricer.volsurface","version":1,"label":null,"slices":'
                   '[{"a":1e-400,"b":0.1,"rho":0.0,"m":0.0,"sigma":0.1,"expiry":1.0}],'
                   '"forward_curve":[]}')
        assert VolSurface.from_json(payload).slices[1.0].a == 0.0

    def test_future_version_is_refused_distinctly_from_a_foreign_file(self):
        """Separate schema and version fields so a reader can tell 'upgrade
        optpricer' from 'this is not a surface file'."""
        d = self._surface().to_dict()
        with pytest.raises(ValueError, match="Upgrade optpricer"):
            VolSurface.from_dict({**d, "version": 99})
        with pytest.raises(ValueError, match="Not a VolSurface payload"):
            VolSurface.from_dict({**d, "schema": "acme.surface"})

    def test_unknown_fields_are_refused_not_ignored(self):
        """A field this version does not understand may be the one that changes
        what the numbers mean."""
        d = self._surface().to_dict()
        with pytest.raises(ValueError, match="unrecognised fields"):
            VolSurface.from_dict({**d, "surprise": 1})
        with pytest.raises(ValueError, match="unrecognised fields"):
            VolSurface.from_dict({**d, "slices": [{**d["slices"][0], "extra": 1}]})

    @pytest.mark.parametrize("mutate,match", [
        (lambda d: {**d, "slices": []}, "non-empty"),
        (lambda d: {**d, "slices": [{k: v for k, v in d["slices"][0].items() if k != "rho"}]}, "missing"),
        (lambda d: {**d, "slices": [{**d["slices"][0], "a": "0.1"}]}, "expected a number"),
        (lambda d: {**d, "label": 42}, "label must be"),
        (lambda d: {k: v for k, v in d.items() if k != "version"}, "version must be an integer"),
        (lambda d: {**d, "forward_curve": [[1.0, 100.0, 7.0]]}, "two-element"),
        (lambda d: {**d, "forward_curve": [[1.0, 100.0], [1.0, 999.0]]}, "Duplicate forward-curve"),
        (lambda d: {**d, "forward_curve": {"1.0": 100.0}}, "must be a list"),
    ])
    def test_malformed_payloads_raise(self, mutate, match):
        d = self._surface().to_dict()
        with pytest.raises(ValueError, match=match):
            VolSurface.from_dict(mutate(d))

    def test_no_file_io_in_the_library(self):
        """The caller writes the file; the library stays at zero open()."""
        import pathlib
        src = pathlib.Path("src/optpricer")
        offenders = [p.name for p in src.glob("*.py") if "open(" in p.read_text()]
        assert not offenders, f"file I/O appeared in {offenders}"
