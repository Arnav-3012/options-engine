"""
analysis/implied_vol.py
-----------------------
Bisection-based Implied Volatility (IV) back-solver.

Implied volatility is the sigma that makes the Black-Scholes price equal
to the market-observed option price. There is no closed-form inverse, so
we find it numerically via bisection search.

Algorithm:
    Given market_price, find sigma* such that BS(sigma*) = market_price
    - Bracket: [sigma_low, sigma_high] = [0.001, 5.0]
    - Bisect until interval < tolerance (1e-6)
    - Return NaN if market price is outside arbitrage bounds
"""

import numpy as np
from models.black_scholes import black_scholes

SIGMA_LOW  = 0.001   # lower bound for IV search (0.1% vol)
SIGMA_HIGH = 5.0     # upper bound for IV search (500% vol)
TOLERANCE  = 1e-6    # convergence threshold on sigma interval
MAX_ITER   = 200     # safety cap on bisection iterations


def implied_vol(
    market_price: float,
    S0: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
) -> float:
    """
    Solve for implied volatility via bisection search.

    Finds sigma_implied such that:
        Black-Scholes(S0, K, T, r, sigma_implied, option_type) == market_price

    Parameters
    ----------
    market_price : float — observed market price of the option
    S0           : float — current stock price
    K            : float — strike price
    T            : float — time to expiry in years
    r            : float — annualised risk-free rate
    option_type  : str   — 'call' or 'put'

    Returns
    -------
    float
        Implied volatility (annualised). Returns np.nan if:
        - market_price is below intrinsic value (violates no-arbitrage)
        - market_price exceeds theoretical maximum (S0 for call, K*e^-rT for put)
        - T <= 0 or market_price <= 0
    """
    option_type = option_type.lower()

    # ── Guard: degenerate inputs ───────────────────────────────────────────
    if T <= 0 or market_price <= 0:
        return np.nan

    # ── Guard: arbitrage bounds ────────────────────────────────────────────
    discount = np.exp(-r * T)
    if option_type == "call":
        intrinsic = max(S0 - K * discount, 0.0)
        upper_bound = S0
    else:
        intrinsic = max(K * discount - S0, 0.0)
        upper_bound = K * discount

    if market_price < intrinsic or market_price > upper_bound:
        return np.nan

    # ── Bisection search ───────────────────────────────────────────────────
    lo, hi = SIGMA_LOW, SIGMA_HIGH

    for _ in range(MAX_ITER):
        mid = (lo + hi) / 2.0
        bs_mid = black_scholes(S0, K, T, r, mid, option_type)["price"]

        if bs_mid < market_price:
            lo = mid
        else:
            hi = mid

        if (hi - lo) < TOLERANCE:
            break

    return (lo + hi) / 2.0
