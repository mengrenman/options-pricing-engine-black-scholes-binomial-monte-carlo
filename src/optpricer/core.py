from dataclasses import dataclass

@dataclass(frozen=True)
class OptionSpec:
    S0: float
    K: float
    T: float          # years
    r: float          # continuous risk-free
    sigma: float
    q: float = 0.0    # continuous dividend yield

CALL = "call"
PUT  = "put"
