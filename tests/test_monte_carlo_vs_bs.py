import numpy as np
import pytest
from optpricer.core import OptionSpec, CALL, PUT
from optpricer.black_scholes import price as bs
from optpricer.monte_carlo import euro_price_mc

def test_mc_matches_bs_within_tol():
    opt = OptionSpec(S0=100, K=100, T=1.0, r=0.03, sigma=0.25, q=0.01)
    for kind in (CALL, PUT):
        mc, se = euro_price_mc(opt, kind, n_paths=40_000, control_variate=True, seed=1)
        assert abs(mc - bs(opt, kind)) / bs(opt, kind) < 0.005  # <0.5% error


# ---------------------------------------------------------------------------
# Standard error under antithetic sampling
# ---------------------------------------------------------------------------
class TestAntitheticStandardError:
    """An antithetic pair is negatively correlated, so the 2n draws are not
    independent.  Statistics are accumulated over pair averages instead."""

    def test_chunk_reports_pairs_not_draws(self):
        from optpricer.monte_carlo import _mc_chunk_sumstats

        n = 1000
        kw = {"S0": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "q": 0.0,
              "sigma": 0.2, "kind": "call", "seed": 7}
        n_anti = _mc_chunk_sumstats(n, antithetic=True, **kw)[0]
        n_plain = _mc_chunk_sumstats(n, antithetic=False, **kw)[0]
        assert n_anti == n, "antithetic n_eff should count pairs, not draws"
        assert n_plain == n

    def test_chunk_stats_match_explicit_pair_averages(self):
        """Recompute the statistics by hand from the same RNG stream."""
        from optpricer.monte_carlo import _mc_chunk_sumstats

        n, S0, K, T, r, q, sigma, seed = 5000, 100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 11
        n_eff, sumX, sumX2, _, _, _ = _mc_chunk_sumstats(
            n, S0=S0, K=K, T=T, r=r, q=q, sigma=sigma,
            kind="call", antithetic=True, seed=seed,
        )

        rng = np.random.default_rng(seed)
        Z = rng.standard_normal(n)
        mu = (r - q - 0.5 * sigma ** 2) * T
        sd = sigma * np.sqrt(T)
        df = np.exp(-r * T)
        X_up = df * np.maximum(S0 * np.exp(mu + sd * Z) - K, 0.0)
        X_dn = df * np.maximum(S0 * np.exp(mu - sd * Z) - K, 0.0)
        P = 0.5 * (X_up + X_dn)

        assert n_eff == P.size
        assert sumX == pytest.approx(P.sum(), rel=1e-12)
        assert sumX2 == pytest.approx((P * P).sum(), rel=1e-12)

    def test_price_is_unaffected_by_pairing(self):
        """Pair averaging preserves the mean, so the estimate is unchanged."""
        opt = OptionSpec(S0=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.0)
        px, _ = euro_price_mc(opt, CALL, n_paths=200_000, seed=3,
                              antithetic=True, control_variate=False, n_workers=1)
        ref = bs(opt, CALL)
        assert abs(px - ref) < 0.05

    def test_reported_se_is_calibrated(self):
        """Regression: with control variate + antithetic the reported SE used
        to understate the true spread by ~32% (95% intervals covered 86%).

        Compares the spread of the estimator across seeds against the SE it
        reports.  Fixed seeds, so this is deterministic.
        """
        opt = OptionSpec(S0=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.0)
        errs, ses = [], []
        ref = bs(opt, CALL)
        for s in range(80):
            px, se = euro_price_mc(opt, CALL, n_paths=100_000, seed=50_000 + s,
                                   chunk_size=100_000, antithetic=True,
                                   control_variate=True, n_workers=1)
            errs.append(px - ref)
            ses.append(se)
        ratio = float(np.std(errs, ddof=1) / np.mean(ses))
        assert 0.75 < ratio < 1.25, f"reported SE off by {ratio:.2f}x"
