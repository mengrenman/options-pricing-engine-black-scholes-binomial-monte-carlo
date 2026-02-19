"""Tests for the risk engine."""

import numpy as np
import pytest
from optpricer import OptionSpec, CALL, PUT, bs_greeks
from optpricer.black_scholes_vec import bs_price_vec
from optpricer.risk import (
    numerical_greeks, scenario_grid, portfolio_risk,
    var_historical, cvar_historical,
)

OPT = OptionSpec(S0=100, K=100, T=1.0, r=0.05, sigma=0.2)


def _bs_pricer(S, K, T, r, q, sigma, kind):
    """Simple wrapper around vectorized BS for the risk engine."""
    return float(bs_price_vec(S, K, T, r, q, sigma, kind))


class TestNumericalGreeks:
    def test_vs_analytical_bs(self):
        ng = numerical_greeks(_bs_pricer, 100, 100, 1.0, 0.05, 0.0, 0.2, "call")
        ag = bs_greeks(OPT, CALL)
        assert abs(ng["delta"] - ag["delta"]) < 0.005
        assert abs(ng["gamma"] - ag["gamma"]) < 0.002
        assert abs(ng["vega"] - ag["vega"]) < 0.5
        assert abs(ng["rho"] - ag["rho"]) < 0.5

    def test_all_keys(self):
        ng = numerical_greeks(_bs_pricer, 100, 100, 1.0, 0.05, 0.0, 0.2, "call")
        assert set(ng.keys()) == {"delta", "gamma", "vega", "theta", "rho"}

    def test_put_delta_negative(self):
        ng = numerical_greeks(_bs_pricer, 100, 100, 1.0, 0.05, 0.0, 0.2, "put")
        assert ng["delta"] < 0


class TestScenarioGrid:
    def test_output_shape(self):
        spots = np.array([90.0, 100.0, 110.0])
        vols = np.array([0.15, 0.20, 0.25])
        result = scenario_grid(_bs_pricer, 100, 100, 1.0, 0.05, 0.0, 0.2,
                               "call", spots, vols)
        assert result["prices"].shape == (3, 3)

    def test_call_monotone_in_spot(self):
        spots = np.linspace(80, 120, 5)
        vols = np.array([0.2])
        result = scenario_grid(_bs_pricer, 100, 100, 1.0, 0.05, 0.0, 0.2,
                               "call", spots, vols)
        prices = result["prices"][:, 0]
        assert np.all(np.diff(prices) > 0)


class TestPortfolioRisk:
    def test_single_instrument(self):
        instruments = [{"S": 100, "K": 100, "T": 1.0, "r": 0.05, "q": 0.0,
                        "sigma": 0.2, "kind": "call", "position": 1.0}]
        result = portfolio_risk(instruments, _bs_pricer)
        ng = numerical_greeks(_bs_pricer, 100, 100, 1.0, 0.05, 0.0, 0.2, "call")
        assert abs(result["total_delta"] - ng["delta"]) < 1e-10

    def test_long_short_delta_offset(self):
        instruments = [
            {"S": 100, "K": 100, "T": 1.0, "r": 0.05, "q": 0.0,
             "sigma": 0.2, "kind": "call", "position": 1.0},
            {"S": 100, "K": 100, "T": 1.0, "r": 0.05, "q": 0.0,
             "sigma": 0.2, "kind": "call", "position": -1.0},
        ]
        result = portfolio_risk(instruments, _bs_pricer)
        assert abs(result["total_delta"]) < 1e-10


class TestVaR:
    def test_var_positive(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(-0.01, 0.02, 1000)
        v = var_historical(returns)
        assert v > 0

    def test_cvar_geq_var(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(-0.01, 0.02, 1000)
        v = var_historical(returns, confidence=0.95)
        cv = cvar_historical(returns, confidence=0.95)
        assert cv >= v - 1e-10

    def test_known_normal(self):
        """VaR of N(0, 1) at 99% ≈ 2.326."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 1, 100_000)
        v = var_historical(returns, confidence=0.99)
        assert abs(v - 2.326) < 0.1
