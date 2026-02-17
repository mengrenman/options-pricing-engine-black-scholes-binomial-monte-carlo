from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .calibration import VolSurface


# ---------------------------------------------------------------------------
# Legacy convenience wrapper — kept for backward compatibility
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OptionSpec:
    """Single-option container bundling instrument + market data.

    Fine for quick calculations; for production batch pricing prefer
    the separated Instrument / MarketData types with vectorised pricers.
    """
    S0: float
    K: float
    T: float          # years
    r: float          # continuous risk-free
    sigma: float
    q: float = 0.0    # continuous dividend yield

    def __post_init__(self):
        if self.S0 <= 0:
            raise ValueError(f"S0 must be positive, got {self.S0}")
        if self.K <= 0:
            raise ValueError(f"K must be positive, got {self.K}")
        if self.T <= 0:
            raise ValueError(f"T must be positive, got {self.T}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")


# ---------------------------------------------------------------------------
# Production data model — instrument / market data separation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Instrument:
    """What the contract *is* — static, does not change when markets move.

    Parameters
    ----------
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    kind : str
        ``"call"`` or ``"put"``.
    exercise : str
        ``"european"`` (default) or ``"american"``.
    """
    K: float
    T: float
    kind: str = "call"
    exercise: str = "european"

    def __post_init__(self):
        if self.K <= 0:
            raise ValueError(f"K must be positive, got {self.K}")
        if self.T <= 0:
            raise ValueError(f"T must be positive, got {self.T}")
        if self.kind not in ("call", "put"):
            raise ValueError(f"kind must be 'call' or 'put', got {self.kind!r}")
        if self.exercise not in ("european", "american"):
            raise ValueError(
                f"exercise must be 'european' or 'american', got {self.exercise!r}"
            )


@dataclass
class MarketData:
    """What is *moving* — updates every tick / snapshot.

    Parameters
    ----------
    spot : float
        Current underlying price.
    rate : float
        Continuously-compounded risk-free rate.
    q : float
        Continuous dividend / funding yield (default 0).
    vol_surface : VolSurface | None
        Calibrated volatility surface.  When present, ``iv()`` queries it.
    flat_vol : float
        Fallback scalar volatility used when no surface is available.
    """
    spot: float
    rate: float
    q: float = 0.0
    vol_surface: VolSurface | None = None
    flat_vol: float = 0.0

    def iv(self, K: float, T: float) -> float:
        """Look up implied vol — from calibrated surface if available, else flat."""
        if self.vol_surface is not None:
            return float(self.vol_surface.iv(K, T))
        return self.flat_vol


def to_instrument_market(
    opt: OptionSpec, kind: str = "call"
) -> tuple[Instrument, MarketData]:
    """Decompose a legacy ``OptionSpec`` into the production pair."""
    inst = Instrument(K=opt.K, T=opt.T, kind=kind)
    mkt = MarketData(spot=opt.S0, rate=opt.r, q=opt.q, flat_vol=opt.sigma)
    return inst, mkt


CALL = "call"
PUT  = "put"
