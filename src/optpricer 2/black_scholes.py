import math
from math import log, sqrt, exp
from typing import Literal, Dict
from statistics import NormalDist
from .core import OptionSpec, CALL, PUT

_nd = NormalDist()

def _d1_d2(S0, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0 or S0 <= 0 or K <= 0:
        raise ValueError("S0,K,T,sigma must be positive.")
    rt = sigma * sqrt(T)
    d1 = (log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / rt
    d2 = d1 - rt
    return d1, d2

def price(opt: OptionSpec, kind: Literal["call","put"]=CALL) -> float:
    d1, d2 = _d1_d2(opt.S0, opt.K, opt.T, opt.r, opt.q, opt.sigma)
    disc_r = exp(-opt.r * opt.T)
    disc_q = exp(-opt.q * opt.T)
    if kind == CALL:
        return disc_q * opt.S0 * _nd.cdf(d1) - disc_r * opt.K * _nd.cdf(d2)
    elif kind == PUT:
        return disc_r * opt.K * _nd.cdf(-d2) - disc_q * opt.S0 * _nd.cdf(-d1)
    else:
        raise ValueError("kind must be 'call' or 'put'")

def greeks(opt: OptionSpec, kind: Literal["call","put"]=CALL) -> Dict[str, float]:
    """Returns greeks with sigma in absolute units (vega is dPrice/dSigma, not per 1%)."""
    d1, d2 = _d1_d2(opt.S0, opt.K, opt.T, opt.r, opt.q, opt.sigma)
    n_d1   = math.exp(-0.5 * d1*d1) / math.sqrt(2*math.pi)  # pdf
    N_d1   = _nd.cdf(d1)
    N_d2   = _nd.cdf(d2)
    disc_r = math.exp(-opt.r * opt.T)
    disc_q = math.exp(-opt.q * opt.T)
    srt    = opt.sigma * math.sqrt(opt.T)

    # Common
    gamma = disc_q * n_d1 / (opt.S0 * srt)
    vega  = opt.S0 * disc_q * n_d1 * math.sqrt(opt.T)

    if kind == CALL:
        delta = disc_q * N_d1
        theta = (-opt.S0 * disc_q * n_d1 * opt.sigma / (2*math.sqrt(opt.T))
                 - opt.r * opt.K * disc_r * N_d2
                 + opt.q * opt.S0 * disc_q * N_d1)
        rho   = opt.K * opt.T * disc_r * N_d2
    else:
        delta = disc_q * (N_d1 - 1.0)
        theta = (-opt.S0 * disc_q * n_d1 * opt.sigma / (2*math.sqrt(opt.T))
                 + opt.r * opt.K * disc_r * _nd.cdf(-d2)
                 - opt.q * opt.S0 * disc_q * _nd.cdf(-d1))
        rho   = -opt.K * opt.T * disc_r * _nd.cdf(-d2)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}

def implied_vol(opt: OptionSpec, target_price: float, kind: Literal["call","put"]=CALL,
                *, tol: float = 1e-8, maxiter: int = 100, bracket=(1e-6, 5.0)) -> float:
    """Brent root find on sigma."""
    from scipy.optimize import brentq
    base = OptionSpec(S0=opt.S0, K=opt.K, T=opt.T, r=opt.r, sigma=0.2, q=opt.q)
    def f(sig):
        return price(base.__class__(**{**base.__dict__, "sigma": sig}), kind) - target_price
    a, b = bracket
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        # widen bracket heuristically
        a, b = 1e-6, max(5.0, 2*opt.sigma if opt.sigma > 0 else 1.0)
    return float(brentq(lambda s: f(s), a, b, xtol=tol, maxiter=maxiter))
