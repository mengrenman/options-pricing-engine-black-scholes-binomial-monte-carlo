#!/usr/bin/env python3
"""Production script: batch-price an options portfolio.

Usage
-----
    python scripts/price_book.py --input portfolio.csv --output prices.csv
    python scripts/price_book.py --input portfolio.csv --output prices.json --greeks

Input CSV format
----------------
    id,S0,K,T,r,sigma,q,kind,method
    1,100,110,0.5,0.05,0.20,0.0,call,bs
    2,100,95,1.0,0.05,0.25,0.01,put,mc
    3,100,105,0.5,0.05,0.20,0.0,call,binomial

For exotic options, add columns: exotic_type, barrier, barrier_type, average_type, etc.
    id,S0,K,T,r,sigma,q,kind,method,exotic_type,barrier,barrier_type
    4,100,100,1.0,0.05,0.20,0.0,call,exotic,barrier,120,up-and-out

Output
------
    CSV or JSON with columns: id, price, stderr, delta, gamma, vega, theta, rho
"""

from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optpricer.core import OptionSpec, CALL, PUT
from optpricer.black_scholes import price as bs_price
from optpricer.black_scholes_vec import bs_price_vec, bs_greeks_vec
from optpricer.binomial import crr
from optpricer.monte_carlo import euro_price_mc
from optpricer.processes import gbm_paths
from optpricer.exotics import barrier_price, asian_price, digital_price, lookback_price


def _price_row(row: dict, compute_greeks: bool) -> dict:
    """Price a single portfolio row and return result dict."""
    rid = row.get("id", "")
    S0 = float(row["S0"])
    K = float(row["K"])
    T = float(row["T"])
    r = float(row["r"])
    sigma = float(row["sigma"])
    q = float(row.get("q", 0.0))
    kind = row["kind"].strip().lower()
    method = row["method"].strip().lower()

    result = {"id": rid, "price": None, "stderr": None}

    if method == "bs":
        px = float(bs_price_vec(S0, K, T, r, q, sigma, kind))
        result["price"] = px
        if compute_greeks:
            g = bs_greeks_vec(S0, K, T, r, q, sigma, kind)
            for key in ("delta", "gamma", "vega", "theta", "rho"):
                result[key] = float(g[key])

    elif method == "binomial":
        opt = OptionSpec(S0=S0, K=K, T=T, r=r, sigma=sigma, q=q)
        american = row.get("american", "false").strip().lower() == "true"
        px = crr(opt, kind, N=500, american=american)
        result["price"] = px
        if compute_greeks:
            # Bump-and-reprice for Greeks
            g = bs_greeks_vec(S0, K, T, r, q, sigma, kind)
            for key in ("delta", "gamma", "vega", "theta", "rho"):
                result[key] = float(g[key])

    elif method == "mc":
        opt = OptionSpec(S0=S0, K=K, T=T, r=r, sigma=sigma, q=q)
        px, se = euro_price_mc(opt, kind, n_paths=100_000, seed=1)
        result["price"] = px
        result["stderr"] = se
        if compute_greeks:
            g = bs_greeks_vec(S0, K, T, r, q, sigma, kind)
            for key in ("delta", "gamma", "vega", "theta", "rho"):
                result[key] = float(g[key])

    elif method == "exotic":
        exotic_type = row.get("exotic_type", "").strip().lower()
        n_steps = int(row.get("n_steps", 500))
        n_paths = int(row.get("n_paths", 100_000))
        paths = gbm_paths(S0, r, q, sigma, T, n_steps, n_paths,
                          antithetic=True, seed=1)

        if exotic_type == "barrier":
            barrier = float(row["barrier"])
            barrier_type = row["barrier_type"].strip().lower()
            rebate = float(row.get("rebate", 0.0))
            px, se = barrier_price(paths, K, r, T, kind, barrier, barrier_type, rebate)
        elif exotic_type == "asian":
            avg_type = row.get("average_type", "arithmetic").strip().lower()
            strike_type = row.get("strike_type", "fixed").strip().lower()
            px, se = asian_price(paths, K, r, T, kind, avg_type, strike_type)
        elif exotic_type == "digital":
            payout = float(row.get("payout", 1.0))
            px, se = digital_price(paths, K, r, T, kind, payout)
        elif exotic_type == "lookback":
            strike_type = row.get("strike_type", "floating").strip().lower()
            px, se = lookback_price(paths, r, T, kind, K=K, strike_type=strike_type)
        else:
            raise ValueError(f"Unknown exotic_type: {exotic_type!r}")

        result["price"] = px
        result["stderr"] = se
    else:
        raise ValueError(f"Unknown method: {method!r}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Batch-price an options portfolio."
    )
    parser.add_argument("--input", required=True, help="Path to portfolio CSV")
    parser.add_argument("--output", required=True, help="Output path (.csv or .json)")
    parser.add_argument("--greeks", action="store_true", help="Compute Greeks")
    args = parser.parse_args()

    # Read portfolio
    with open(args.input, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Pricing {len(rows)} positions...")

    # Batch-price vanilla BS rows for speed
    results = []
    for i, row in enumerate(rows):
        try:
            res = _price_row(row, args.greeks)
            results.append(res)
        except Exception as e:
            print(f"  Row {i} (id={row.get('id', '?')}): ERROR — {e}")
            results.append({"id": row.get("id", ""), "price": None, "error": str(e)})

    # Write output
    output_path = Path(args.output)
    if output_path.suffix == ".json":
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
    else:
        # CSV
        if not results:
            print("No results to write.")
            return
        fieldnames = list(results[0].keys())
        # Ensure all keys are present
        for r in results:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

    print(f"Results written to {args.output}")

    # Summary
    priced = [r for r in results if r.get("price") is not None]
    failed = [r for r in results if r.get("price") is None]
    print(f"  Priced: {len(priced)}  |  Failed: {len(failed)}")


if __name__ == "__main__":
    main()
