import argparse
import math
from .core import OptionSpec, CALL, PUT
from .black_scholes import price as bs_price
from .binomial import crr
from .monte_carlo import euro_price_mc

def _kind(s: str):
    s = s.lower()
    if s in {"call", "c"}:
        return CALL
    if s in {"put", "p"}:
        return PUT
    raise argparse.ArgumentTypeError("kind must be 'call' or 'put'")

def add_common(parser: argparse.ArgumentParser):
    parser.add_argument("--S0", type=float, required=True)
    parser.add_argument("--K", type=float, required=True)
    parser.add_argument("--T", type=float, required=True, help="years")
    parser.add_argument("--r", type=float, required=True, help="cont. risk-free")
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--q", type=float, default=0.0, help="cont. dividend yield")
    parser.add_argument("--kind", type=_kind, default=CALL, help="call|put")

def cmd_bs(args):
    opt = OptionSpec(args.S0, args.K, args.T, args.r, args.sigma, args.q)
    print(f"{bs_price(opt, args.kind):.10f}")

def cmd_binomial(args):
    opt = OptionSpec(args.S0, args.K, args.T, args.r, args.sigma, args.q)
    px = crr(opt, args.kind, N=args.N, american=args.american)
    print(f"{px:.10f}")

def cmd_mc(args):
    opt = OptionSpec(args.S0, args.K, args.T, args.r, args.sigma, args.q)
    px, se = euro_price_mc(
        opt,
        kind=args.kind,
        n_paths=args.n_paths,
        seed=args.seed,
        antithetic=not args.no_antithetic,
        control_variate=not args.no_cv,
    )
    print(f"{px:.10f}  (stderr {se:.10f})")

def main():
    p = argparse.ArgumentParser(prog="optpricer", description="Options pricing CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # BS
    p_bs = sub.add_parser("bs", help="Black–Scholes price")
    add_common(p_bs)
    p_bs.set_defaults(func=cmd_bs)

    # Binomial
    p_bin = sub.add_parser("binomial", help="CRR binomial price")
    add_common(p_bin)
    p_bin.add_argument("--N", type=int, default=500)
    p_bin.add_argument("--american", action="store_true")
    p_bin.set_defaults(func=cmd_binomial)

    # Monte Carlo (GBM terminal)
    p_mc = sub.add_parser("mc", help="Monte Carlo price (GBM)")
    add_common(p_mc)
    p_mc.add_argument("--n-paths", dest="n_paths", type=int, default=100_000)
    p_mc.add_argument("--seed", type=int, default=None)
    p_mc.add_argument("--no-antithetic", action="store_true")
    p_mc.add_argument("--no-cv", action="store_true", help="disable control variate")
    p_mc.set_defaults(func=cmd_mc)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
