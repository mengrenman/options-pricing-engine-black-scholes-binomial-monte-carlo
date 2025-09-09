# optpricer — A small, fast options-pricing engine

Black–Scholes (closed-form + Greeks + implied vol), Cox-Ross-Rubinstein (CRR) binomial (European & American),  
and Monte Carlo with variance reduction. Includes smile-capable path generators you can  
use in notebooks (local volatility, Heston, etc.).

- **Python**: 3.10+  
- **Layout**: `src/` (editable installs work cleanly)  
- **License**: MIT

---

## Features

- **Black–Scholes**: European calls/puts, Greeks, implied volatility solver  
- **Binomial (CRR)**: European & American pricing  
- **Monte Carlo (GBM)**: terminal-only, memory-light, antithetic variates, control variate, standard error  
- **Smile-capable dynamics (for notebooks)**: Local Vol paths, Heston paths (stochastic vol)  
- **Clean API** via a tiny immutable `OptionSpec` dataclass  

```python
OptionSpec(S0, K, T, r, sigma, q=0.0)
```

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

## Quickstart (Python API)

```python
import optpricer as op

# Define the contract/market state
opt = op.OptionSpec(S0=100, K=110, T=1.0, r=0.03, sigma=0.20, q=0.0)

# Black–Scholes (European)
print(op.bs_price(opt, op.CALL))
print(op.bs_greeks(opt, op.CALL))        # dict: delta, gamma, vega, theta, rho

# Implied volatility from a target price
iv = op.implied_vol(opt, target_price=5.29, kind=op.CALL)
print("IV:", iv)

# Binomial (CRR) — American put example
print(op.crr(opt, op.PUT, N=500, american=True))

# Monte Carlo (GBM terminal) — returns (price, standard_error)
px, se = op.euro_price_mc(opt, kind=op.CALL, n_paths=200_000, seed=1)
print(px, se)
```

---

### Notes

- `q` is a continuous dividend yield.
- Monte Carlo uses a control variate by default and reports a standard error;  
  a 95% CI is roughly `price ± 1.96 * stderr`.

---

## CLI (command-line interface)

After installation, a small `optpricer` CLI is available:

```bash
# Black–Scholes
optpricer bs --S0 100 --K 110 --T 1 --r 0.03 --sigma 0.20 --kind call

# Binomial (CRR)
optpricer binomial --S0 100 --K 110 --T 1 --r 0.03 --sigma 0.20 --N 500 --kind put --american

# Monte Carlo (GBM)
optpricer mc --S0 100 --K 110 --T 1 --r 0.03 --sigma 0.20 --n-paths 200000 --seed 1
```

---

## Volatility smile: what’s supported?

- **Plug-in IV surface**: pass a different sigma for each (K, T) into Black–Scholes.
- **Local Vol paths**: simulate S_t under σ_loc(S,t) and invert MC prices to implied vol.
- **Heston paths**: stochastic volatility → natural smiles/smirks.

See the example notebook:  
`notebooks/03_smile_models.ipynb`

---

## Examples / Notebooks

- `notebooks/01_bs_quickstart.ipynb` — BS pricing & Greeks  
- `notebooks/02_mc_vs_bs.ipynb` — Monte Carlo vs Black–Scholes (accuracy & stderr)  
- `notebooks/03_smile_models.ipynb` — IV surface + Local Vol & Heston smiles  

---

## Project structure

~~~
.
├── src/
│   └── optpricer/
│       ├── __init__.py                # public API + version
│       ├── core.py                    # OptionSpec, CALL/PUT
│       ├── black_scholes.py          # BS price, Greeks, implied vol
│       ├── binomial.py               # CRR tree (Euro/American)
│       ├── monte_carlo.py            # terminal-only MC (price, stderr)
│       └── processes.py              # path generators (GBM, Local Vol, Heston,...)
│ 
├── tests/
│   ├── test_black_scholes.py
│   └── test_monte_carlo_vs_bs.py
│ 
└── notebooks/
    ├── 01_bs_mc_quickstart.ipynb
    ├── 02_mc_vs_bs_and_visualization.ipynb
    └── 03_smile_models.ipynb
~~~

---

## Testing

```bash
pytest -q
```

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
