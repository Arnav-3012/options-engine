"""
models/gbm.py
-------------
Vectorised Geometric Brownian Motion (GBM) path simulator.

Discretised risk-neutral GBM formula applied at each time step:
    S(t+dt) = S(t) * exp[(r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z]
    where Z ~ N(0,1)

Drift uses the risk-free rate r (risk-neutral measure), not the
historical mu — this is required for no-arbitrage option pricing.

All operations are fully vectorised with NumPy (no Python for-loops).
"""

import numpy as np


def simulate_gbm(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate full GBM price paths from t=0 to t=T.

    Parameters
    ----------
    S0      : float — initial stock price
    r       : float — annualised risk-free rate (risk-neutral drift)
    sigma   : float — annualised volatility
    T       : float — time horizon in years (e.g. 30/365 for 30 days)
    n_steps : int   — number of time steps (e.g. 30 for daily steps)
    n_paths : int   — number of simulation paths (e.g. 10_000)
    seed    : int, optional — random seed for reproducibility

    Returns
    -------
    np.ndarray, shape (n_steps + 1, n_paths)
        Simulated price paths. Row 0 is S0 for all paths.
        paths[i, j] is the price of path j at time step i.
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps

    # Z: (n_steps, n_paths) standard normal draws — one per step per path
    Z = np.random.standard_normal((n_steps, n_paths))

    # Log-return increment at each step: (n_steps, n_paths)
    increments = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z

    # np.cumsum collapses the sequential chain S_t = S_{t-1} * exp(i_t)
    # into S_t = S0 * exp(i_1 + i_2 + ... + i_t) — same result, no loop.
    # Prepend zeros so row 0 gives S0 * exp(0) = S0.
    log_paths = np.vstack([np.zeros(n_paths), np.cumsum(increments, axis=0)])

    # Convert to prices: S0 * exp(cumulative log-return)
    return S0 * np.exp(log_paths)


def terminal_prices(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate only the terminal price S_T for each path (single-step GBM).

    More efficient than simulate_gbm when full paths are not needed,
    e.g. for European option pricing with no path dependency.

    Parameters
    ----------
    S0      : float — initial stock price
    r       : float — annualised risk-free rate
    sigma   : float — annualised volatility
    T       : float — time horizon in years
    n_paths : int   — number of simulation paths
    seed    : int, optional — random seed

    Returns
    -------
    np.ndarray, shape (n_paths,)
        Terminal stock prices S_T.
    """
    if seed is not None:
        np.random.seed(seed)

    Z = np.random.standard_normal(n_paths)
    return S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
