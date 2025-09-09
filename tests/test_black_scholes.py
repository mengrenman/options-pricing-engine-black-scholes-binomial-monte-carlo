from optpricer.core import OptionSpec, CALL, PUT
from optpricer.black_scholes import price

def test_bs_known_values():
    opt = OptionSpec(S0=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.0)
    assert abs(price(opt, CALL) - 10.4506) < 1e-3
    assert abs(price(opt, PUT)  - 5.5735)  < 1e-3
