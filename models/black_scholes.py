"""
models/black_scholes.py
-----------------------
Analytical Black-Scholes pricing for European call and put options.

Formulas:
    d1 = [ln(S0/K) + (r + sigma^2/2) * T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    Call = S0 * N(d1) - K * exp(-r*T) * N(d2)
    Put  = K * exp(-r*T) * N(-d2) - S0 * N(-d1)

where N(·) is the cumulative standard normal distribution.
"""

import numpy as np
from scipy.stats import norm


def black_scholes(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> dict:
    """
    Compute the Black-Scholes price and intermediates for a European option.

    Parameters
    ----------
    S0          : float — current stock price
    K           : float — strike price
    T           : float — time to expiry in years (e.g. 30/365)
    r           : float — annualised risk-free rate (e.g. 0.065)
    sigma       : float — annualised volatility (e.g. 0.2107)
    option_type : str   — 'call' or 'put' (case-insensitive)

    Returns
    -------
    dict with keys:
        price  (float) — option price
        d1     (float) — d1 intermediate
        d2     (float) — d2 intermediate
        delta  (float) — N(d1) for call, N(d1)-1 for put
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
    if T <= 0:
        raise ValueError("T must be positive (time to expiry in years).")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:  # put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0

    return {"price": price, "d1": d1, "d2": d2, "delta": delta}
