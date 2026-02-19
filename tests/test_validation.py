"""Tests for the model validation framework."""

import numpy as np
import pytest
from optpricer import OptionSpec, CALL, PUT, bs_price
from optpricer.validation import (
    cross_validate, convergence_analysis, stress_test, backtest_delta_hedge,
)
from optpricer.processes import gbm_paths

OPT = OptionSpec(S0=100, K=100, T=1.0, r=0.05, sigma=0.2)


class TestCrossValidate:
    def test_all_methods_consistent(self):
        result = cross_validate(OPT, CALL)
        assert result["max_discrepancy"] < 0.5
        assert "bs" in result
        assert "fdm" in result
        assert "fem" in result

    def test_subset_methods(self):
        result = cross_validate(OPT, CALL, methods=["bs", "tree"])
        assert "bs" in result
        assert "tree" in result
        assert "fdm" not in result

    def test_mc_returns_tuple(self):
        result = cross_validate(OPT, CALL, methods=["mc"], mc_paths=10_000)
        assert isinstance(result["mc"], tuple)
        assert len(result["mc"]) == 2


class TestConvergenceAnalysis:
    def test_tree_convergence(self):
        result = convergence_analysis(OPT, CALL, "tree", "N",
                                      [50, 100, 200, 400])
        assert result["errors"][-1] < result["errors"][0]

    def test_fdm_convergence(self):
        result = convergence_analysis(OPT, CALL, "fdm", "N_S",
                                      [50, 100, 200])
        assert result["errors"][-1] < result["errors"][0]

    def test_order_positive(self):
        result = convergence_analysis(OPT, CALL, "fdm", "N_S",
                                      [50, 100, 200, 400])
        assert result["order"] > 0


class TestStressTest:
    def test_output_shape(self):
        spots = np.array([0.9, 1.0, 1.1])
        vols = np.array([-0.05, 0.0, 0.05])
        rates = np.array([-0.01, 0.0, 0.01])
        result = stress_test(OPT, CALL, spots, vols, rates)
        assert result.shape == (3, 3, 3)

    def test_call_monotone_in_spot(self):
        spots = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
        vols = np.array([0.0])
        rates = np.array([0.0])
        result = stress_test(OPT, CALL, spots, vols, rates)
        prices = result[:, 0, 0]
        assert np.all(np.diff(prices) > 0)


class TestBacktestDeltaHedge:
    def test_mean_pnl_near_zero(self):
        """In a BS world, perfect delta-hedging gives ~zero mean P&L."""
        paths = gbm_paths(OPT.S0, OPT.r, OPT.q, OPT.sigma, OPT.T,
                          252, 10_000, antithetic=True, seed=42)
        result = backtest_delta_hedge(OPT, CALL, paths, rebalance_freq=1)
        # Mean P&L should be close to zero (within a few cents)
        assert abs(result["mean_pnl"]) < 0.5, f"mean_pnl = {result['mean_pnl']}"

    def test_output_keys(self):
        paths = gbm_paths(OPT.S0, OPT.r, OPT.q, OPT.sigma, OPT.T,
                          50, 1000, antithetic=True, seed=42)
        result = backtest_delta_hedge(OPT, CALL, paths)
        assert "pnl" in result
        assert "mean_pnl" in result
        assert "std_pnl" in result
        assert "max_drawdown" in result

    def test_pnl_shape(self):
        paths = gbm_paths(OPT.S0, OPT.r, OPT.q, OPT.sigma, OPT.T,
                          50, 1000, antithetic=True, seed=42)
        result = backtest_delta_hedge(OPT, CALL, paths)
        assert result["pnl"].shape == (2000,)  # antithetic doubles
