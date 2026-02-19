# optpricer — Production Options Pricing Engine

A comprehensive options pricing library with **five independent pricing engines** (Black-Scholes, Monte Carlo, Binomial Trees, Finite Differences, Finite Elements), smile-capable stochastic process generators, SVI calibration, Dupire local volatility, Milstein discretisation, risk management, and model validation.

- **Python**: 3.10+
- **Layout**: `src/` (editable installs work cleanly)
- **License**: MIT

---

## Features

### Pricing Engines

| Engine | Module | Capabilities |
|---|---|---|
| **Black-Scholes** | `black_scholes.py` | European calls/puts, Greeks, implied vol solver, vectorised batch pricing |
| **Binomial (CRR)** | `binomial.py` | European & American pricing, vectorised |
| **Monte Carlo** | `monte_carlo.py` | Antithetic variates, control variate, chunked memory, multi-worker |
| **Finite Difference (FDM)** | `pde.py` | θ-scheme on log-spot grid, European/American, barriers (knock-in/out), local-vol pricing, grid-based Greeks |
| **Finite Element (FEM)** | `fem.py` | 1D Galerkin with linear hat functions, consistent mass + stiffness matrices |

### Exotic Options

- **Barrier** options (up/down, in/out) — via Monte Carlo and FDM with Dirichlet BCs
- **Asian** options (arithmetic/geometric, fixed/floating strike)
- **Digital** (cash-or-nothing) options
- **Lookback** (floating-strike) options

### Calibration & Local Volatility

- **SVI** (Stochastic Volatility Inspired) calibration — slice-by-slice with arbitrage-aware constraints
- **Dupire local volatility** extraction from calibrated SVI surface using analytical derivatives
- Callable `σ(S, t)` interface compatible with all local-vol consumers (FDM, MC, Milstein)

### Stochastic Processes

| Process | Scheme | Description |
|---|---|---|
| **GBM** | Exact log-Euler | Constant volatility |
| **Merton Jump-Diffusion** | Euler + Poisson | Log-normal jumps |
| **Heston** | Euler (QE available) | Stochastic variance with mean reversion |
| **SABR** | Euler | Stochastic vol with β-elasticity |
| **Local Vol** | Euler-Maruyama | Deterministic σ(S, t) surface |
| **GBM Milstein** | Milstein (order 1.0) | Constant vol — demonstration/testing |
| **Local-Vol Milstein** | Milstein (order 1.0) | σ'(S, t) via central finite differences |

### Risk Management

- **Numerical Greeks** — model-agnostic bump-and-reprice (Δ, Γ, V, Θ, ρ)
- **Scenario grids** — 2D spot × vol evaluation
- **Portfolio risk** — aggregate Greeks across multiple positions
- **VaR / CVaR** — historical Value-at-Risk and Expected Shortfall

### Model Validation

- **Cross-model benchmarking** — BS vs MC vs Tree vs FDM vs FEM consistency check
- **Convergence analysis** — error decay and order estimation
- **Stress testing** — 3D (spot × vol × rate) shock grid
- **Delta-hedge backtesting** — P&L simulation from hedging along GBM paths

---

## Install (developer)

```bash
# from the repo root
python -m pip install -U pip setuptools wheel
python -m pip install -e ".[dev]"
```

Using the src layout, the import path is `src/optpricer/`.
Editable install ensures changes take effect immediately.

**Optional**: set up a Jupyter kernel for this env

```bash
python -m ipykernel install --user --name optpricer --display-name "Python (optpricer)"
```

---

## Quickstart

```python
import optpricer as op

# Define the contract/market state
opt = op.OptionSpec(S0=100, K=110, T=1.0, r=0.03, sigma=0.20, q=0.0)

# Black-Scholes
print(op.bs_price(opt, op.CALL))
print(op.bs_greeks(opt, op.CALL))        # dict: delta, gamma, vega, theta, rho

# Implied volatility
iv = op.implied_vol(opt, target_price=5.29, kind=op.CALL)

# Binomial (CRR) — American put
print(op.crr(opt, op.PUT, N=500, american=True))

# Monte Carlo — returns (price, standard_error)
px, se = op.euro_price_mc(opt, kind=op.CALL, n_paths=200_000, seed=1)

# Finite Difference (PDE)
print(op.fd_price(opt, op.CALL))                         # European
print(op.fd_price(opt, op.PUT, american=True))            # American
print(op.fd_price_barrier(opt, op.CALL, 130, "up-and-out"))  # Barrier

# Finite Element
print(op.fem_price(opt, op.CALL))

# Greeks from the FD grid
print(op.fd_greeks(opt, op.CALL))  # dict: delta, gamma, theta
```

### Dupire Local Volatility

```python
from optpricer.calibration import fit_svi_surface, dupire_local_vol_func
from optpricer.pde import fd_price_local_vol

# Calibrate SVI surface from market data
surface = fit_svi_surface(strikes_by_expiry, forwards, market_ivs_by_expiry)

# Extract Dupire local vol as a callable σ(S, t)
sigma_loc = dupire_local_vol_func(surface, r=0.05, q=0.02)

# Price with FDM
fd_price_local_vol(S0=100, K=100, T=1.0, r=0.05, q=0.02,
                   sigma_func=sigma_loc, kind="call")
```

### Risk & Validation

```python
from optpricer.risk import numerical_greeks, scenario_grid, var_historical
from optpricer.validation import cross_validate, backtest_delta_hedge

# Model-agnostic numerical Greeks
greeks = numerical_greeks(my_pricer, S=100, K=100, T=1.0, r=0.05,
                          q=0.0, sigma=0.20, kind="call")

# Cross-validate all five engines
results = cross_validate(opt, "call")  # bs, mc, tree, fdm, fem

# Delta-hedge backtest
bt = backtest_delta_hedge(opt, "call", paths, rebalance_freq=1)
print(f"Mean P&L: {bt['mean_pnl']:.4f}, Std: {bt['std_pnl']:.4f}")
```

---

## CLI (command-line interface)

```bash
# Black-Scholes
optpricer bs --S0 100 --K 110 --T 1 --r 0.03 --sigma 0.20 --kind call

# Binomial (CRR)
optpricer binomial --S0 100 --K 110 --T 1 --r 0.03 --sigma 0.20 --N 500 --kind put --american

# Monte Carlo (GBM)
optpricer mc --S0 100 --K 110 --T 1 --r 0.03 --sigma 0.20 --n-paths 200000 --seed 1
```

---

## Notebooks

| # | Notebook | Topics |
|---|---|---|
| 01 | [Pricing Calls and Puts](notebooks/01_Pricing_Calls_and_Puts.ipynb) | OptionSpec, BS/MC/Tree pricing, Greeks, implied vol, stochastic processes |
| 02 | [Visualization](notebooks/02_Visualization.ipynb) | Price/delta surfaces, IV surface, GBM paths on surfaces, put-call parity, MC convergence |
| 03 | [Volatility Smile](notebooks/03_Volatility_Smile.ipynb) | Synthetic smile, local-vol & Heston smiles, model comparison, round-trip verification |
| 04 | [PDE Finite Difference](notebooks/04_PDE_Finite_Difference.ipynb) | FDM pricing, θ-scheme comparison, American options, barriers, grid Greeks, convergence |
| 05 | [Finite Element Method](notebooks/05_Finite_Element_Method.ipynb) | FEM pricing, FEM vs FDM cross-check, convergence analysis, price surface |
| 06 | [Dupire Local Vol](notebooks/06_Dupire_Local_Vol.ipynb) | SVI calibration, Dupire extraction, local-vol heatmap, Milstein paths, MC pricing |
| 07 | [Risk and Validation](notebooks/07_Risk_and_Validation.ipynb) | Numerical Greeks, scenario grids, portfolio risk, VaR/CVaR, cross-validation, stress testing, delta-hedge backtest |

---

## Project Structure

```
src/
  optpricer/
    __init__.py           # public API (41 exports), version 0.3.0
    core.py               # OptionSpec, CALL/PUT constants
    black_scholes.py      # BS price, Greeks, implied vol
    black_scholes_vec.py  # vectorised BS (batch across strikes/spots)
    binomial.py           # CRR tree (European/American)
    monte_carlo.py        # MC with variance reduction, chunked memory
    pde.py                # FDM θ-scheme: vanilla, American, barrier, local vol
    fem.py                # 1D Galerkin FEM with linear hat functions
    exotics.py            # barrier, Asian, digital, lookback
    processes.py          # GBM, Merton, Heston, SABR, local vol, Milstein
    calibration.py        # SVI calibration, VolSurface, Dupire local vol
    risk.py               # numerical Greeks, scenario grid, VaR/CVaR
    validation.py         # cross-validation, convergence, stress test, backtest
    cli.py                # command-line interface
tests/
  test_black_scholes.py
  test_monte_carlo_vs_bs.py
  test_calibration.py
  test_exotics.py
  test_vectorized.py
  test_pde.py             # 15 tests: EU vs BS, American, barriers, Greeks, convergence
  test_fem.py             # 4 tests: FEM vs BS, FEM vs FDM, convergence
  test_milstein.py        # 6 tests: shape, mean, GBM match, antithetic
  test_dupire.py          # 6 tests: flat surface, positivity, FD/MC integration
  test_risk.py            # 10 tests: numerical Greeks, scenarios, portfolio, VaR/CVaR
  test_validation.py      # 11 tests: cross-validate, convergence, stress, backtest
notebooks/
  01_Pricing_Calls_and_Puts.ipynb
  02_Visualization.ipynb
  03_Volatility_Smile.ipynb
  04_PDE_Finite_Difference.ipynb
  05_Finite_Element_Method.ipynb
  06_Dupire_Local_Vol.ipynb
  07_Risk_and_Validation.ipynb
```

---

## Testing

```bash
pytest -q          # 89 tests, all passing
```

---

## FAQ

**Why is kind not part of OptionSpec?**
OptionSpec describes the market/contract state; the payoff (CALL/PUT) is passed to a pricer.
This keeps the spec reusable (e.g., compute both call and put from the same spec).

**What does Monte Carlo return?**
A tuple `(price, stderr)`. The estimator uses antithetic variates and a control variate
$Y = e^{-rT} S_T$ with known expectation $S_0 e^{-qT}$.

**What's the difference between FDM and FEM?**
Both solve the BS PDE on a log-spot grid. FDM approximates derivatives at grid points;
FEM expands the solution in basis functions and enforces the PDE in a weak (integral) sense.
In 1D they give nearly identical results. FEM shines in 2D/3D (multi-asset, unstructured meshes).

**Does the repo name matter?**
No — the importable package is determined by `src/optpricer/` and `pyproject.toml` (`name="optpricer"`).

---

## Contributing

PRs are welcome! Please run ruff/black and pytest before submitting.

```bash
python -m pip install -e ".[dev]"
ruff check src tests
black src tests
pytest -q
```

---

## License

MIT — see `LICENSE`.
