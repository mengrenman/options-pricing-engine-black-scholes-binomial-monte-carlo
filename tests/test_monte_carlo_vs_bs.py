from optpricer.core import OptionSpec, CALL, PUT
from optpricer.black_scholes import price as bs
from optpricer.monte_carlo import euro_price_mc

def test_mc_matches_bs_within_tol():
    opt = OptionSpec(S0=100, K=100, T=1.0, r=0.03, sigma=0.25, q=0.01)
    for kind in (CALL, PUT):
        mc = euro_price_mc(opt, kind, n_steps=252, n_paths=40_000, control_variate=True, seed=1)
        assert abs(mc - bs(opt, kind)) / bs(opt, kind) < 0.005  # <0.5% error
