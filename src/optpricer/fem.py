"""Finite-element (Galerkin) solver for the Black-Scholes PDE.

Uses piecewise-linear (hat) basis functions on a uniform log-spot grid
``x = ln(S)``.  The semi-discrete system

.. math::

    M \\dot{V} + K V = 0

is integrated in time with a θ-scheme (Crank-Nicolson by default),
yielding a tridiagonal system at each step.

This module demonstrates the FEM approach for option pricing.  For
production use the finite-difference solver in :mod:`optpricer.pde` is
typically preferred owing to simpler implementation and equivalent
accuracy in 1-D.

References
----------
- Topper, J. *Financial Engineering with Finite Elements* (Wiley, 2005).
- Duffy, D.J. *Finite Difference Methods in Financial Engineering*
  (Wiley, 2006), Appendix A2.
"""

from __future__ import annotations

import numpy as np
from typing import Literal

from .core import OptionSpec, CALL, PUT

__all__ = ["fem_price"]


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
    """Uniform log-spot grid.  Identical to the FDM grid helper."""
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
    """Tridiagonal solver (Thomas algorithm), O(N)."""
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


def _assemble_mass_stiffness(
    h: float,
    r: float,
    q: float,
    sigma: float,
    M_int: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """Assemble consistent mass (M) and stiffness (K) tridiagonal matrices.

    The bilinear form for the BS PDE in log-spot ``x`` is:

    .. math::

        a(V, \\phi) = \\int \\tfrac{\\sigma^2}{2}\\,V'\\phi'
                      - \\mu\\,V'\\phi + r\\,V\\phi \\, dx

    where ``\\mu = r - q - \\sigma^2/2``.

    With linear hat functions on a uniform grid of spacing ``h``:

    * **Mass matrix** (consistent):
      M_main = 2h/3,  M_off = h/6

    * **Stiffness matrix** K = K_diff + K_conv + K_react:
      - Diffusion:  K_diff_main = σ²/h, K_diff_off = -σ²/(2h)
      - Convection:  K_conv[i,i±1] = ∓μ/2   (central, skew-symmetric)
      - Reaction:   K_react_main = 2rh/3, K_react_off = rh/6

    Returns three arrays each for M and K: ``(sub, main, sup)`` of length
    ``M_int`` (number of interior nodes).
    """
    s2 = sigma ** 2
    mu = r - q - 0.5 * s2

    # Mass matrix
    M_main = np.full(M_int, 2.0 * h / 3.0)
    M_off = np.full(M_int, h / 6.0)  # sub and super

    # Stiffness: diffusion
    Kd_main = np.full(M_int, s2 / h)
    Kd_off = np.full(M_int, -s2 / (2.0 * h))

    # Stiffness: convection (skew-symmetric on uniform grid)
    Kc_sub = np.full(M_int, mu / 2.0)
    Kc_sup = np.full(M_int, -mu / 2.0)
    Kc_main = np.zeros(M_int)

    # Stiffness: reaction
    Kr_main = np.full(M_int, 2.0 * r * h / 3.0)
    Kr_off = np.full(M_int, r * h / 6.0)

    # Combine
    K_sub = Kd_off + Kc_sub + Kr_off
    K_main = Kd_main + Kc_main + Kr_main
    K_sup = Kd_off + Kc_sup + Kr_off

    return M_off, M_main, M_off, K_sub, K_main, K_sup


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fem_price(
    opt: OptionSpec,
    kind: Literal["call", "put"] = CALL,
    *,
    N_S: int = 200,
    N_t: int = 200,
    theta: float = 0.5,
    S_max_mult: float = 4.0,
) -> float:
    """Price a European vanilla option via 1-D Galerkin FEM.

    Parameters
    ----------
    opt : OptionSpec
    kind : ``"call"`` or ``"put"``
    N_S : int
        Number of spatial intervals.
    N_t : int
        Number of time steps.
    theta : float
        Time-stepping parameter (0.5 = Crank-Nicolson).
    S_max_mult : float
        Grid range as multiple of σ√T.

    Returns
    -------
    float
        Option price.
    """
    x_grid, dx, dt = _build_grid(opt.S0, opt.T, opt.sigma, N_S, N_t, S_max_mult)
    h = dx
    M_int = N_S - 1  # number of interior nodes

    # Assemble matrices
    (M_sub, M_main, M_sup,
     K_sub, K_main, K_sup) = _assemble_mass_stiffness(
        h, opt.r, opt.q, opt.sigma, M_int,
    )

    # LHS = M + theta*dt*K
    L_sub = M_sub + theta * dt * K_sub
    L_main = M_main + theta * dt * K_main
    L_sup = M_sup + theta * dt * K_sup

    # RHS_matrix = M - (1-theta)*dt*K
    e = (1.0 - theta) * dt
    R_sub = M_sub - e * K_sub
    R_main = M_main - e * K_main
    R_sup = M_sup - e * K_sup

    # Terminal condition
    S_grid = np.exp(x_grid)
    if kind == CALL:
        V_full = np.maximum(S_grid - opt.K, 0.0)
    else:
        V_full = np.maximum(opt.K - S_grid, 0.0)

    # Backward march
    for n in range(N_t - 1, -1, -1):
        tau = (N_t - n) * dt

        # Boundary values
        S_min_val = S_grid[0]
        S_max_val = S_grid[-1]
        if kind == CALL:
            bc_left = 0.0
            bc_right = max(S_max_val - opt.K * np.exp(-opt.r * tau), 0.0)
        else:
            bc_left = max(opt.K * np.exp(-opt.r * tau) - S_min_val, 0.0)
            bc_right = 0.0

        V_int = V_full[1:N_S]  # interior values

        # RHS = R_matrix @ V_int  (tridiagonal multiply)
        rhs = R_main * V_int
        rhs[1:] += R_sub[1:] * V_int[:-1]
        rhs[:-1] += R_sup[:-1] * V_int[1:]

        # Old boundary contributions from explicit part
        rhs[0] += R_sub[0] * V_full[0]
        rhs[-1] += R_sup[-1] * V_full[N_S]

        # New boundary contributions (move to RHS from LHS)
        rhs[0] -= L_sub[0] * bc_left
        rhs[-1] -= L_sup[-1] * bc_right

        # Solve tridiagonal system
        V_new_int = _thomas_solve(L_sub, L_main, L_sup, rhs)

        V_full[0] = bc_left
        V_full[1:N_S] = V_new_int
        V_full[N_S] = bc_right

    return float(np.interp(np.log(opt.S0), x_grid, V_full))
