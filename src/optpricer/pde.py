"""Finite-difference PDE solver for the Black-Scholes equation.

Implements the θ-scheme (explicit / Crank-Nicolson / fully-implicit) on a
uniform log-spot grid ``x = ln(S)``.  Under this change of variable the
constant-volatility BS PDE becomes

.. math::

    \\frac{\\partial V}{\\partial t}
    + \\frac{\\sigma^2}{2}\\frac{\\partial^2 V}{\\partial x^2}
    + \\left(r - q - \\tfrac{\\sigma^2}{2}\\right)\\frac{\\partial V}{\\partial x}
    - r\\,V = 0

which has **constant coefficients**, yielding a tridiagonal system at each
time step that is solved in O(N) via the Thomas algorithm.

References
----------
- Duffy, D.J. *Numerical Methods in Computational Finance* (Wiley, 2022),
  chapters 14–17.
- Duffy, D.J. *Finite Difference Methods in Financial Engineering* (Wiley,
  2006), chapters 7–10, 22, 28.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Literal

from .core import OptionSpec, CALL, PUT

__all__ = [
    "fd_price",
    "fd_price_barrier",
    "fd_greeks",
    "fd_price_local_vol",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_grid(
    S0: float,
    T: float,
    sigma: float,
    N_S: int,
    N_t: int,
    S_max_mult: float,
) -> tuple[np.ndarray, float, float]:
    """Build a uniform log-spot grid and return ``(x_grid, dx, dt)``."""
    x_range = S_max_mult * sigma * np.sqrt(T)
    x_min = np.log(S0) - x_range
    x_max = np.log(S0) + x_range
    x_grid = np.linspace(x_min, x_max, N_S + 1)
    dx = x_grid[1] - x_grid[0]
    dt = T / N_t
    return x_grid, dx, dt


def _thomas_solve(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    """Solve tridiagonal ``A x = d`` via the Thomas algorithm, O(N).

    Parameters
    ----------
    a : sub-diagonal, shape (N,), ``a[0]`` unused.
    b : main diagonal, shape (N,).
    c : super-diagonal, shape (N,), ``c[-1]`` unused.
    d : right-hand side, shape (N,).
    """
    N = len(b)
    b_ = b.copy()
    d_ = d.copy()
    for i in range(1, N):
        w = a[i] / b_[i - 1]
        b_[i] -= w * c[i - 1]
        d_[i] -= w * d_[i - 1]
    x = np.empty(N)
    x[-1] = d_[-1] / b_[-1]
    for i in range(N - 2, -1, -1):
        x[i] = (d_[i] - c[i] * x[i + 1]) / b_[i]
    return x


def _payoff(x_grid: np.ndarray, K: float, kind: str) -> np.ndarray:
    """Terminal payoff on the log-spot grid."""
    S = np.exp(x_grid)
    if kind == CALL:
        return np.maximum(S - K, 0.0)
    return np.maximum(K - S, 0.0)


# ---------------------------------------------------------------------------
# Core θ-scheme engine
# ---------------------------------------------------------------------------

def _fd_solve(
    x_grid: np.ndarray,
    dx: float,
    dt: float,
    N_t: int,
    K: float,
    r: float,
    q: float,
    sigma: float,
    kind: str,
    theta: float,
    american: bool,
    *,
    sigma_func: Callable[[np.ndarray, float], np.ndarray] | None = None,
    barrier_nodes: np.ndarray | None = None,
    barrier_value: float = 0.0,
    return_two_layers: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Backward θ-scheme on the interior of the log-spot grid.

    When *return_two_layers* is True, returns ``(V_at_t0, V_at_t_dt)`` so
    that theta can be extracted.
    """
    N_S = len(x_grid) - 1
    M = N_S - 1  # number of interior points

    # Terminal condition
    V = _payoff(x_grid, K, kind)

    V_at_dt = None  # second layer for theta extraction

    for n in range(N_t - 1, -1, -1):
        tau = (N_t - n) * dt  # time to expiry from current layer
        t_now = n * dt

        # --- Volatilities per node (constant or local) ---
        if sigma_func is not None:
            sig = np.asarray(sigma_func(np.exp(x_grid), t_now), dtype=float)
        else:
            sig = np.full(N_S + 1, sigma)

        sig_int = sig[1:N_S]  # at interior nodes, shape (M,)

        # Coefficients
        alpha = 0.5 * sig_int ** 2 / dx ** 2
        mu = r - q - 0.5 * sig_int ** 2
        beta = mu / (2.0 * dx)

        # --- Boundary values (Dirichlet) ---
        S_min = np.exp(x_grid[0])
        S_max = np.exp(x_grid[-1])
        if kind == CALL:
            bc_left = 0.0
            bc_right = max(S_max - K * np.exp(-r * tau), 0.0)
        else:
            bc_left = max(K * np.exp(-r * tau) - S_min, 0.0)
            bc_right = 0.0

        # --- Build tridiagonal coefficients ---
        # The PDE operator L on interior: L V_j = alpha_j (V_{j-1} - 2V_j + V_{j+1})
        #                                       + beta_j (V_{j+1} - V_{j-1}) - r V_j
        # a_coeff = alpha - beta  (coeff of V_{j-1})
        # b_coeff = -2*alpha - r  (coeff of V_j)
        # c_coeff = alpha + beta  (coeff of V_{j+1})

        a_L = alpha - beta   # lower
        b_L = -2.0 * alpha - r  # diag
        c_L = alpha + beta   # upper

        # LHS matrix: I - theta * dt * L  =>  diag = 1 - theta*dt*b_L, etc.
        a_lhs = -theta * dt * a_L
        b_lhs = 1.0 - theta * dt * b_L
        c_lhs = -theta * dt * c_L

        # RHS: (I + (1-theta)*dt*L) V^{old}  for interior points
        e = (1.0 - theta) * dt
        V_int = V[1:N_S]

        rhs = (1.0 + e * b_L) * V_int
        # Contribution from V_{j-1}
        rhs[1:] += e * a_L[1:] * V[1:N_S - 1]
        rhs[0] += e * a_L[0] * V[0]
        # Contribution from V_{j+1}
        rhs[:-1] += e * c_L[:-1] * V[2:N_S]
        rhs[-1] += e * c_L[-1] * V[N_S]

        # Move boundary terms from LHS to RHS
        # First interior eqn (j=1): LHS has a_lhs[0]*V[0], move to RHS
        rhs[0] += theta * dt * a_L[0] * bc_left
        # Last interior eqn (j=N_S-1): LHS has c_lhs[-1]*V[N_S], move to RHS
        rhs[-1] += theta * dt * c_L[-1] * bc_right

        # Solve
        V_new_int = _thomas_solve(a_lhs, b_lhs, c_lhs, rhs)

        # Assemble full vector
        V_new = np.empty(N_S + 1)
        V_new[0] = bc_left
        V_new[1:N_S] = V_new_int
        V_new[N_S] = bc_right

        # American early-exercise projection
        if american:
            intrinsic = _payoff(x_grid, K, kind)
            V_new = np.maximum(V_new, intrinsic)

        # Barrier enforcement
        if barrier_nodes is not None:
            V_new[barrier_nodes] = barrier_value

        # Store for Greeks
        if return_two_layers and n == 1:
            V_at_dt = V_new.copy()

        V = V_new

    if return_two_layers:
        return V, V_at_dt if V_at_dt is not None else V
    return V


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fd_price(
    opt: OptionSpec,
    kind: Literal["call", "put"] = CALL,
    *,
    N_S: int = 200,
    N_t: int = 200,
    theta: float = 0.5,
    S_max_mult: float = 4.0,
    american: bool = False,
) -> float:
    """Price a European or American vanilla option via finite differences.

    Uses the θ-scheme on a uniform log-spot grid.

    Parameters
    ----------
    opt : OptionSpec
        Option specification.
    kind : ``"call"`` or ``"put"``
    N_S : int
        Number of spatial intervals (default 200).
    N_t : int
        Number of time steps (default 200).
    theta : float
        Scheme parameter: 0 = explicit, 0.5 = Crank-Nicolson, 1 = implicit.
    S_max_mult : float
        Grid range as a multiple of σ√T.
    american : bool
        Enable early-exercise (default False).

    Returns
    -------
    float
        Option price.
    """
    x_grid, dx, dt = _build_grid(opt.S0, opt.T, opt.sigma, N_S, N_t, S_max_mult)
    V = _fd_solve(
        x_grid, dx, dt, N_t, opt.K, opt.r, opt.q, opt.sigma,
        kind, theta, american,
    )
    return float(np.interp(np.log(opt.S0), x_grid, V))


def fd_price_barrier(
    opt: OptionSpec,
    kind: Literal["call", "put"] = CALL,
    barrier: float = 0.0,
    barrier_type: Literal[
        "up-and-out", "down-and-out", "up-and-in", "down-and-in"
    ] = "up-and-out",
    *,
    rebate: float = 0.0,
    N_S: int = 200,
    N_t: int = 200,
    theta: float = 0.5,
    S_max_mult: float = 4.0,
) -> float:
    """Price a European barrier option via finite differences.

    Knock-out barriers use Dirichlet boundary conditions at the barrier
    level.  Knock-in prices use in/out parity:  ``V_in = V_vanilla − V_out``.

    Parameters
    ----------
    opt : OptionSpec
    kind : ``"call"`` or ``"put"``
    barrier : float
        Barrier level.
    barrier_type : str
        One of ``"up-and-out"``, ``"down-and-out"``, ``"up-and-in"``,
        ``"down-and-in"``.
    rebate : float
        Rebate on knock-out (default 0).

    Returns
    -------
    float
    """
    grid_kw = dict(N_S=N_S, N_t=N_t, theta=theta, S_max_mult=S_max_mult)

    if barrier_type.endswith("in"):
        out_type = barrier_type.replace("in", "out")
        vanilla = fd_price(opt, kind, **grid_kw)
        knock_out = fd_price_barrier(opt, kind, barrier, out_type,
                                     rebate=rebate, **grid_kw)
        return vanilla - knock_out

    x_grid, dx, dt = _build_grid(opt.S0, opt.T, opt.sigma, N_S, N_t, S_max_mult)

    # Find grid nodes at or beyond the barrier
    x_barrier = np.log(barrier)
    if barrier_type.startswith("up"):
        nodes = np.where(x_grid >= x_barrier)[0]
    else:
        nodes = np.where(x_grid <= x_barrier)[0]

    V = _fd_solve(
        x_grid, dx, dt, N_t, opt.K, opt.r, opt.q, opt.sigma,
        kind, theta, False,
        barrier_nodes=nodes, barrier_value=rebate,
    )
    return float(np.interp(np.log(opt.S0), x_grid, V))


def fd_greeks(
    opt: OptionSpec,
    kind: Literal["call", "put"] = CALL,
    **kwargs,
) -> dict[str, float]:
    """Extract delta, gamma, and theta from the FD grid.

    Delta and gamma are computed from the spatial derivatives at
    ``x = ln(S0)`` via central differences.  Theta is the difference
    between the first two time layers.

    Returns
    -------
    dict[str, float]
        Keys: ``delta``, ``gamma``, ``theta``.
    """
    N_S = kwargs.pop("N_S", 200)
    N_t = kwargs.pop("N_t", 200)
    theta_scheme = kwargs.pop("theta", 0.5)
    S_max_mult = kwargs.pop("S_max_mult", 4.0)
    american = kwargs.pop("american", False)

    x_grid, dx, dt = _build_grid(opt.S0, opt.T, opt.sigma, N_S, N_t, S_max_mult)

    V_0, V_dt = _fd_solve(
        x_grid, dx, dt, N_t, opt.K, opt.r, opt.q, opt.sigma,
        kind, theta_scheme, american,
        return_two_layers=True,
    )

    x0 = np.log(opt.S0)
    j = int(np.searchsorted(x_grid, x0))
    j = max(1, min(j, len(x_grid) - 2))

    S0 = opt.S0

    # dV/dx and d²V/dx² via central differences
    dVdx = (V_0[j + 1] - V_0[j - 1]) / (2.0 * dx)
    d2Vdx2 = (V_0[j + 1] - 2.0 * V_0[j] + V_0[j - 1]) / dx ** 2

    # Chain rule:  delta = (1/S) dV/dx
    #              gamma = (1/S²)(d²V/dx² − dV/dx)
    delta = dVdx / S0
    gamma = (d2Vdx2 - dVdx) / S0 ** 2

    # Theta ≈ -(V(t=0) - V(t=dt)) / dt
    V0_val = float(np.interp(x0, x_grid, V_0))
    Vdt_val = float(np.interp(x0, x_grid, V_dt))
    theta_val = -(V0_val - Vdt_val) / dt

    return {"delta": float(delta), "gamma": float(gamma), "theta": float(theta_val)}


def fd_price_local_vol(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma_func: Callable[[np.ndarray, float], np.ndarray],
    kind: Literal["call", "put"] = CALL,
    *,
    N_S: int = 200,
    N_t: int = 200,
    theta: float = 0.5,
    S_max_mult: float = 4.0,
    ref_vol: float = 0.3,
) -> float:
    """Price with local volatility σ(S, t) on the FD grid.

    At each time step the node-dependent diffusion coefficient is
    obtained by evaluating ``sigma_func(S_array, t)``.

    Parameters
    ----------
    S0, K, T, r, q : float
        Market and instrument parameters.
    sigma_func : callable
        ``sigma_func(S_array, t) -> sigma_array`` .
    kind : str
    ref_vol : float
        Reference vol used only for grid construction (default 0.3).

    Returns
    -------
    float
    """
    x_grid, dx, dt = _build_grid(S0, T, ref_vol, N_S, N_t, S_max_mult)
    N_t_actual = round(T / dt)

    V = _fd_solve(
        x_grid, dx, dt, N_t_actual, K, r, q, 0.0,
        kind, theta, False, sigma_func=sigma_func,
    )
    return float(np.interp(np.log(S0), x_grid, V))
