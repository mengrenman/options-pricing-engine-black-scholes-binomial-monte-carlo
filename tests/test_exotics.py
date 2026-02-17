"""Tests for exotic option pricing.

Validation strategies:
- Barrier: in/out parity (knock-in + knock-out = vanilla)
- Digital: closed-form BS comparison (e^{-rT} N(d2))
- Asian (geometric): closed-form under GBM
- Lookback: non-negativity + sanity bounds
"""

import math
import numpy as np
import pytest
from statistics import NormalDist

from optpricer.processes import gbm_paths
from optpricer.exotics import barrier_price, asian_price, digital_price, lookback_price
from optpricer.black_scholes_vec import bs_price_vec

_nd = NormalDist()

# Shared test parameters
S0, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.20
N_STEPS, N_PATHS, SEED = 500, 100_000, 42


@pytest.fixture
def paths():
    return gbm_paths(S0, r, q, sigma, T, N_STEPS, N_PATHS, antithetic=True, seed=SEED)


# ---------------------------------------------------------------------------
# Barrier: knock-in + knock-out = vanilla
# ---------------------------------------------------------------------------
class TestBarrier:
    def test_up_in_out_parity_call(self, paths):
        barrier = 120.0
        p_out, _ = barrier_price(paths, K, r, T, "call", barrier, "up-and-out")
        p_in, _  = barrier_price(paths, K, r, T, "call", barrier, "up-and-in")
        vanilla = float(bs_price_vec(S0, K, T, r, q, sigma, "call"))
        # in + out should equal vanilla (within MC noise)
        assert abs((p_in + p_out) - vanilla) < 0.50, \
            f"in/out parity failed: {p_in:.4f} + {p_out:.4f} != {vanilla:.4f}"

    def test_down_in_out_parity_put(self, paths):
        barrier = 85.0
        p_out, _ = barrier_price(paths, K, r, T, "put", barrier, "down-and-out")
        p_in, _  = barrier_price(paths, K, r, T, "put", barrier, "down-and-in")
        vanilla = float(bs_price_vec(S0, K, T, r, q, sigma, "put"))
        assert abs((p_in + p_out) - vanilla) < 0.50

    def test_knockout_leq_vanilla(self, paths):
        """Knock-out price must be <= vanilla."""
        p_out, _ = barrier_price(paths, K, r, T, "call", 130.0, "up-and-out")
        vanilla = float(bs_price_vec(S0, K, T, r, q, sigma, "call"))
        assert p_out <= vanilla + 0.01  # small tolerance for MC noise


# ---------------------------------------------------------------------------
# Digital: compare to closed-form e^{-rT} N(d2)
# ---------------------------------------------------------------------------
class TestDigital:
    def test_digital_call_vs_closed_form(self, paths):
        d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        closed_form = math.exp(-r * T) * _nd.cdf(d2)

        mc_price, se = digital_price(paths, K, r, T, "call", payout=1.0)
        assert abs(mc_price - closed_form) < 3 * se + 0.01, \
            f"Digital call MC={mc_price:.4f} vs CF={closed_form:.4f} (se={se:.4f})"

    def test_digital_put_vs_closed_form(self, paths):
        d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        closed_form = math.exp(-r * T) * _nd.cdf(-d2)

        mc_price, se = digital_price(paths, K, r, T, "put", payout=1.0)
        assert abs(mc_price - closed_form) < 3 * se + 0.01


# ---------------------------------------------------------------------------
# Asian (geometric): closed-form under GBM
# ---------------------------------------------------------------------------
class TestAsian:
    def test_geometric_asian_call(self, paths):
        """Geometric Asian has closed-form under GBM.

        Adjusted parameters:
            sigma_a = sigma / sqrt(3)
            r_a = 0.5 * (r - q - sigma^2/6) + 0.5 * sigma_a^2  (approx)
        For a continuous-monitoring geometric Asian, the exact formula is:
            sigma_a = sigma / sqrt(3)
            drift_a = (r - q - 0.5*sigma^2) / 2 + 0.5*sigma_a^2
        We compare MC to a BS price with adjusted params.
        """
        n = N_STEPS  # number of monitoring dates
        sigma_a = sigma * math.sqrt((n + 1) * (2 * n + 1) / (6 * n * n))
        mu_a = (r - q - 0.5 * sigma**2) * (n + 1) / (2 * n) + 0.5 * sigma_a**2

        # Price as a BS call with adjusted forward
        F_a = S0 * math.exp(mu_a * T)
        d1 = (math.log(F_a / K) + 0.5 * sigma_a**2 * T) / (sigma_a * math.sqrt(T))
        d2 = d1 - sigma_a * math.sqrt(T)
        cf_price = math.exp(-r * T) * (F_a * _nd.cdf(d1) - K * _nd.cdf(d2))

        mc_price, se = asian_price(paths, K, r, T, "call", average_type="geometric")
        # Allow wider tolerance due to discrete vs continuous monitoring
        assert abs(mc_price - cf_price) < max(3 * se, 0.30), \
            f"Geo Asian MC={mc_price:.4f} vs CF={cf_price:.4f}"

    def test_arithmetic_geq_geometric(self, paths):
        """Arithmetic average >= geometric average (Jensen's inequality)
        so arithmetic Asian call >= geometric Asian call."""
        arith, _ = asian_price(paths, K, r, T, "call", average_type="arithmetic")
        geom, _  = asian_price(paths, K, r, T, "call", average_type="geometric")
        assert arith >= geom - 0.05  # small tolerance for MC noise


# ---------------------------------------------------------------------------
# Lookback
# ---------------------------------------------------------------------------
class TestLookback:
    def test_floating_call_nonneg(self, paths):
        """Floating lookback call = S_T - S_min >= 0 always."""
        px, se = lookback_price(paths, r, T, "call", strike_type="floating")
        assert px > 0

    def test_floating_put_nonneg(self, paths):
        """Floating lookback put = S_max - S_T >= 0 always."""
        px, se = lookback_price(paths, r, T, "put", strike_type="floating")
        assert px > 0

    def test_lookback_geq_vanilla(self, paths):
        """Lookback floating call >= vanilla call (always exercises at best)."""
        lb_px, _ = lookback_price(paths, r, T, "call", strike_type="floating")
        van_px = float(bs_price_vec(S0, K, T, r, q, sigma, "call"))
        assert lb_px > van_px - 0.05

    def test_fixed_strike_call(self, paths):
        """Fixed-strike lookback call = max(S_max - K, 0), should be >= vanilla call."""
        px, _ = lookback_price(paths, r, T, "call", K=K, strike_type="fixed")
        van = float(bs_price_vec(S0, K, T, r, q, sigma, "call"))
        assert px >= van - 0.05
