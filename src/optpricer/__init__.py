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

# Calibration
from .calibration import SVIParams, VolSurface, fit_svi, fit_svi_surface

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
    # Calibration
    "SVIParams", "VolSurface", "fit_svi", "fit_svi_surface",
]

__version__ = "0.2.0"
