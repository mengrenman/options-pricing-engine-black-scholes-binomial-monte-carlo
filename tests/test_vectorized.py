"""Tests for vectorised Black-Scholes and binomial pricing."""

import numpy as np
import pytest
from optpricer.core import OptionSpec, CALL, PUT
from optpricer.black_scholes import price as bs_scalar, greeks as greeks_scalar
from optpricer.black_scholes_vec import bs_price_vec, bs_greeks_vec, bs_implied_vol_vec
from optpricer.binomial import crr, crr_vec


# ---------------------------------------------------------------------------
# bs_price_vec matches scalar bs_price
# ---------------------------------------------------------------------------
class TestBSPriceVec:
    def test_single_call_matches_scalar(self):
        opt = OptionSpec(S0=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.0)
        expected = bs_scalar(opt, CALL)
        got = bs_price_vec(100, 100, 1.0, 0.05, 0.0, 0.2, "call")
        assert abs(float(got) - expected) < 1e-10

    def test_single_put_matches_scalar(self):
        opt = OptionSpec(S0=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.0)
        expected = bs_scalar(opt, PUT)
        got = bs_price_vec(100, 100, 1.0, 0.05, 0.0, 0.2, "put")
        assert abs(float(got) - expected) < 1e-10

    def test_array_of_spots(self):
        spots = np.array([90.0, 100.0, 110.0])
        prices = bs_price_vec(spots, 100, 1.0, 0.05, 0.0, 0.2, "call")
        assert prices.shape == (3,)
        for i, S in enumerate(spots):
            opt = OptionSpec(S0=S, K=100, T=1.0, r=0.05, sigma=0.2)
            assert abs(prices[i] - bs_scalar(opt, CALL)) < 1e-10

    def test_array_of_strikes(self):
        strikes = np.linspace(80, 120, 50)
        prices = bs_price_vec(100, strikes, 1.0, 0.05, 0.0, 0.2, "call")
        assert prices.shape == (50,)
        # Prices should be monotonically decreasing for calls
        assert np.all(np.diff(prices) < 0)

    def test_with_dividend(self):
        opt = OptionSpec(S0=100, K=110, T=0.5, r=0.03, sigma=0.25, q=0.02)
        expected = bs_scalar(opt, CALL)
        got = bs_price_vec(100, 110, 0.5, 0.03, 0.02, 0.25, "call")
        assert abs(float(got) - expected) < 1e-10


# ---------------------------------------------------------------------------
# bs_greeks_vec matches scalar greeks
# ---------------------------------------------------------------------------
class TestBSGreeksVec:
    def test_scalar_greeks_match(self):
        opt = OptionSpec(S0=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.0)
        expected = greeks_scalar(opt, CALL)
        got = bs_greeks_vec(100, 100, 1.0, 0.05, 0.0, 0.2, "call")
        for key in ("delta", "gamma", "vega", "theta", "rho"):
            assert abs(float(got[key]) - expected[key]) < 1e-10, f"{key} mismatch"

    def test_vectorized_greeks(self):
        spots = np.array([90.0, 100.0, 110.0])
        got = bs_greeks_vec(spots, 100, 1.0, 0.05, 0.0, 0.2, "call")
        assert got["delta"].shape == (3,)
        # Call delta should increase with spot
        assert np.all(np.diff(got["delta"]) > 0)


# ---------------------------------------------------------------------------
# bs_implied_vol_vec
# ---------------------------------------------------------------------------
class TestBSImpliedVolVec:
    def test_round_trip(self):
        sigma_true = 0.3
        px = bs_price_vec(100, 100, 1.0, 0.05, 0.0, sigma_true, "call")
        sigma_rec = bs_implied_vol_vec(100, 100, 1.0, 0.05, 0.0, px, "call")
        assert abs(float(sigma_rec) - sigma_true) < 1e-6

    def test_array_round_trip(self):
        strikes = np.array([90.0, 100.0, 110.0])
        sigma_true = 0.25
        prices = bs_price_vec(100, strikes, 1.0, 0.05, 0.0, sigma_true, "call")
        sigmas = bs_implied_vol_vec(100, strikes, 1.0, 0.05, 0.0, prices, "call")
        np.testing.assert_allclose(sigmas, sigma_true, atol=1e-6)


# ---------------------------------------------------------------------------
# crr_vec matches scalar crr
# ---------------------------------------------------------------------------
class TestCRRVec:
    def test_single_call_matches(self):
        opt = OptionSpec(S0=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.0)
        expected = crr(opt, CALL, N=200)
        got = crr_vec(100, 100, 1.0, 0.05, 0.0, 0.2, "call", N=200)
        assert abs(float(got[0]) - expected) < 1e-10

    def test_array_of_strikes(self):
        strikes = np.array([90.0, 100.0, 110.0])
        prices = crr_vec(100, strikes, 1.0, 0.05, 0.0, 0.2, "call", N=200)
        assert prices.shape == (3,)
        for i, K in enumerate(strikes):
            opt = OptionSpec(S0=100, K=K, T=1.0, r=0.05, sigma=0.2)
            expected = crr(opt, CALL, N=200)
            assert abs(prices[i] - expected) < 1e-10

    def test_american_put(self):
        opt = OptionSpec(S0=100, K=110, T=1.0, r=0.05, sigma=0.3, q=0.0)
        expected = crr(opt, PUT, N=200, american=True)
        got = crr_vec(100, 110, 1.0, 0.05, 0.0, 0.3, "put", N=200, american=True)
        assert abs(float(got[0]) - expected) < 1e-10
