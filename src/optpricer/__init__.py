# optpricer — options pricing engine
# Public API

# Legacy scalar interface
from .core import OptionSpec, CALL, PUT
from .black_scholes import price as bs_price, greeks as bs_greeks, implied_vol
from .monte_carlo import euro_price_mc
from .binomial import crr

# Production data model
from .core import Instrument, MarketData, to_instrument_market

# Vectorised pricers
from .black_scholes_vec import bs_price_vec, bs_greeks_vec, bs_implied_vol_vec
from .binomial import crr_vec

# Exotic payoffs
from .exotics import barrier_price, asian_price, digital_price, lookback_price

# Calibration & Dupire
from .calibration import (
    SVIParams, VolSurface, fit_svi, fit_svi_surface,
    dupire_local_vol, dupire_local_vol_func,
)

# PDE (Finite Difference)
from .pde import fd_price, fd_price_barrier, fd_greeks, fd_price_local_vol

# FEM (Finite Element)
from .fem import fem_price

# Stochastic processes — Milstein schemes
from .processes import gbm_milstein_paths, milstein_local_vol_paths

# Risk engine
from .risk import (
    numerical_greeks, scenario_grid, portfolio_risk,
    var_historical, cvar_historical,
)

# Model validation
from .validation import (
    cross_validate, convergence_analysis, stress_test, backtest_delta_hedge,
)

__all__ = [
    # Legacy
    "OptionSpec", "CALL", "PUT",
    "bs_price", "bs_greeks", "implied_vol",
    "euro_price_mc", "crr",
    # Production data model
    "Instrument", "MarketData", "to_instrument_market",
    # Vectorised
    "bs_price_vec", "bs_greeks_vec", "bs_implied_vol_vec", "crr_vec",
    # Exotics
    "barrier_price", "asian_price", "digital_price", "lookback_price",
    # Calibration & Dupire
    "SVIParams", "VolSurface", "fit_svi", "fit_svi_surface",
    "dupire_local_vol", "dupire_local_vol_func",
    # PDE (Finite Difference)
    "fd_price", "fd_price_barrier", "fd_greeks", "fd_price_local_vol",
    # FEM (Finite Element)
    "fem_price",
    # Milstein
    "gbm_milstein_paths", "milstein_local_vol_paths",
    # Risk
    "numerical_greeks", "scenario_grid", "portfolio_risk",
    "var_historical", "cvar_historical",
    # Validation
    "cross_validate", "convergence_analysis", "stress_test",
    "backtest_delta_hedge",
]

__version__ = "0.3.0"
