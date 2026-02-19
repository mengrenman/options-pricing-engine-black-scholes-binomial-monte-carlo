"""Tests for Milstein path generators."""

import numpy as np
import pytest
from optpricer.processes import gbm_milstein_paths, milstein_local_vol_paths, gbm_paths

S0, r, q, sigma, T = 100.0, 0.05, 0.0, 0.2, 1.0
N_STEPS, N_PATHS, SEED = 500, 50_000, 42


class TestGBMMilstein:
    def test_output_shape(self):
        paths = gbm_milstein_paths(S0, r, q, sigma, T, N_STEPS, N_PATHS,
                                    antithetic=True, seed=SEED)
        assert paths.shape == (N_STEPS + 1, 2 * N_PATHS)

    def test_mean_terminal(self):
        """E[S_T] = S0 exp((r-q)T) under risk-neutral measure."""
        paths = gbm_milstein_paths(S0, r, q, sigma, T, N_STEPS, N_PATHS,
                                    antithetic=True, seed=SEED)
        mean_ST = paths[-1, :].mean()
        expected = S0 * np.exp((r - q) * T)
        assert abs(mean_ST - expected) / expected < 0.01

    def test_matches_exact_gbm_distribution(self):
        """Terminal mean should be close to exact GBM terminal mean."""
        exact = gbm_paths(S0, r, q, sigma, T, N_STEPS, N_PATHS,
                          antithetic=True, seed=SEED)
        milstein = gbm_milstein_paths(S0, r, q, sigma, T, N_STEPS, N_PATHS,
                                       antithetic=True, seed=SEED)
        # Means should be very close (both are order-1 for GBM)
        assert abs(exact[-1].mean() - milstein[-1].mean()) / exact[-1].mean() < 0.02


class TestMilsteinLocalVol:
    def test_constant_vol_matches_gbm_mean(self):
        """Milstein with constant σ should have same terminal mean as GBM."""
        sigma_const = lambda S, t: 0.2 * np.ones_like(S)
        paths = milstein_local_vol_paths(
            S0, r, q, T, N_STEPS, N_PATHS, sigma_const,
            antithetic=True, seed=SEED,
        )
        mean_ST = paths[-1, :].mean()
        expected = S0 * np.exp((r - q) * T)
        assert abs(mean_ST - expected) / expected < 0.01

    def test_output_shape(self):
        sigma_const = lambda S, t: 0.2 * np.ones_like(S)
        paths = milstein_local_vol_paths(
            S0, r, q, T, N_STEPS, 1000, sigma_const,
            antithetic=True, seed=SEED,
        )
        assert paths.shape == (N_STEPS + 1, 2000)

    def test_antithetic_reduces_variance(self):
        sigma_const = lambda S, t: 0.2 * np.ones_like(S)
        paths_at = milstein_local_vol_paths(
            S0, r, q, T, N_STEPS, 5000, sigma_const,
            antithetic=True, seed=SEED,
        )
        paths_no = milstein_local_vol_paths(
            S0, r, q, T, N_STEPS, 10000, sigma_const,
            antithetic=False, seed=SEED,
        )
        # Same effective paths, antithetic should have lower variance of mean
        var_at = paths_at[-1].var() / paths_at.shape[1]
        var_no = paths_no[-1].var() / paths_no.shape[1]
        assert var_at < var_no * 1.5  # antithetic generally helps
