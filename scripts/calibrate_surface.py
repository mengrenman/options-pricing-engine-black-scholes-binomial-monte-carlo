#!/usr/bin/env python3
"""Production script: fit SVI volatility surface from market IV data.

Usage
-----
    python scripts/calibrate_surface.py --input market_data.csv --output fitted.json
    python scripts/calibrate_surface.py --input market_data.csv --output fitted.json --plot smile.png

Input CSV format
----------------
    expiry,strike,forward,iv
    0.25,90,100.0,0.22
    0.25,95,100.0,0.20
    ...

Output JSON format
------------------
    {
      "0.25": {"a": ..., "b": ..., "rho": ..., "m": ..., "sigma": ..., "rmse": ...},
      ...
    }
"""

from __future__ import annotations
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Allow running from repo root: python scripts/calibrate_surface.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optpricer.calibration import fit_svi, SVIParams


def _read_csv(path: str):
    """Read market data CSV and group by expiry."""
    strikes_by_T: dict[float, list[float]] = defaultdict(list)
    ivs_by_T: dict[float, list[float]] = defaultdict(list)
    fwd_by_T: dict[float, float] = {}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            T = float(row["expiry"])
            strikes_by_T[T].append(float(row["strike"]))
            ivs_by_T[T].append(float(row["iv"]))
            fwd_by_T[T] = float(row["forward"])

    return {
        T: np.array(strikes_by_T[T]) for T in sorted(strikes_by_T)
    }, fwd_by_T, {
        T: np.array(ivs_by_T[T]) for T in sorted(ivs_by_T)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fit SVI volatility surface to market IV data."
    )
    parser.add_argument("--input", required=True, help="Path to market data CSV")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    parser.add_argument("--plot", default=None, help="Save fitted-vs-market plot to PNG")
    args = parser.parse_args()

    # Read data
    strikes_by_T, fwd_by_T, ivs_by_T = _read_csv(args.input)
    print(f"Loaded {sum(len(v) for v in strikes_by_T.values())} quotes "
          f"across {len(strikes_by_T)} expiries.")

    # Fit SVI per expiry
    results: dict[str, dict] = {}
    for T in sorted(strikes_by_T):
        svi = fit_svi(strikes_by_T[T], fwd_by_T[T], T, ivs_by_T[T])
        k = np.log(strikes_by_T[T] / fwd_by_T[T])
        fitted_ivs = svi.iv(k)
        rmse = float(np.sqrt(np.mean((fitted_ivs - ivs_by_T[T]) ** 2)))
        results[str(T)] = {
            "a": svi.a, "b": svi.b, "rho": svi.rho,
            "m": svi.m, "sigma": svi.sigma, "rmse": rmse,
        }
        print(f"  T={T:.4f}: a={svi.a:.4f} b={svi.b:.4f} "
              f"rho={svi.rho:.4f} m={svi.m:.4f} sig={svi.sigma:.4f} "
              f"RMSE={rmse:.6f}")

    # Write JSON
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFitted params written to {args.output}")

    # Optional plot
    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed — skipping plot.")
            return

        fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4),
                                  squeeze=False)
        for i, (T_str, params) in enumerate(sorted(results.items())):
            T = float(T_str)
            ax = axes[0, i]
            svi = SVIParams(**{k: v for k, v in params.items() if k != "rmse"}, expiry=T)
            k_market = np.log(strikes_by_T[T] / fwd_by_T[T])
            k_fine = np.linspace(k_market.min() - 0.1, k_market.max() + 0.1, 200)

            ax.plot(k_market, ivs_by_T[T], "o", label="Market", markersize=4)
            ax.plot(k_fine, svi.iv(k_fine), "-", label="SVI fit")
            ax.set_title(f"T = {T}")
            ax.set_xlabel("log-moneyness k")
            ax.set_ylabel("Implied Vol")
            ax.legend()

        plt.tight_layout()
        plt.savefig(args.plot, dpi=150)
        print(f"Plot saved to {args.plot}")


if __name__ == "__main__":
    main()
