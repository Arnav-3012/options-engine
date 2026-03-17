"""
models/greeks.py
----------------
Numerical Greeks via bump-and-reprice method.

Each Greek is estimated by perturbing one input to the Black-Scholes
pricer and measuring the resulting price change. This is the standard
approach used in production systems — it works for any pricer, not just
Black-Scholes.

Greeks computed:
    Delta (Δ) — dPrice/dS       sensitivity to stock price
    Gamma (Γ) — d²Price/dS²     rate of change of Delta
    Vega  (ν) — dPrice/dσ       sensitivity to volatility (per 1% move)
    Theta (Θ) — dPrice/dT       time decay per calendar day (negative)
"""

import numpy as np
import pandas as pd
from models.black_scholes import black_scholes

# Bump sizes for finite difference
BUMP_S     = 1.0        # ₹1 stock price bump
BUMP_SIGMA = 0.01       # 1% volatility bump
BUMP_T     = 1 / 365    # 1 calendar day time bump


def greeks(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> dict:
    """
    Compute Delta, Gamma, Vega, and Theta via bump-and-reprice.

    Parameters
    ----------
    S0          : float — current stock price
    K           : float — strike price
    T           : float — time to expiry in years
    r           : float — annualised risk-free rate
    sigma       : float — annualised volatility
    option_type : str   — 'call' or 'put'

    Returns
    -------
    dict with keys:
        price  (float) — current BS option price
        delta  (float) — ΔPrice per ₹1 move in S0
        gamma  (float) — ΔDelta per ₹1 move in S0
        vega   (float) — ΔPrice per 1 percentage-point move in sigma (annualised)
        theta  (float) — ΔPrice per year of time passing (negative for long options);
                         divide by 365 for per-day theta
    """
    def price(s=S0, sig=sigma, t=T):
        return black_scholes(s, K, t, r, sig, option_type)["price"]

    p0      = price()
    p_s_up  = price(s=S0 + BUMP_S)
    p_s_dn  = price(s=S0 - BUMP_S)
    p_sig   = price(sig=sigma + BUMP_SIGMA)
    p_t_dn  = price(t=max(T - BUMP_T, 1e-6))   # guard against T going negative

    delta = (p_s_up - p0) / BUMP_S
    gamma = (p_s_up - 2 * p0 + p_s_dn) / (BUMP_S ** 2)
    vega  = (p_sig  - p0) / BUMP_SIGMA          # per 1 percentage point move
    theta = (p_t_dn - p0) / BUMP_T              # per year (negative = option loses value as time passes)

    return {
        "price": p0,
        "delta": delta,
        "gamma": gamma,
        "vega":  vega,
        "theta": theta,
    }


def greeks_vs_spot(
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    n_points: int = 100,
) -> pd.DataFrame:
    """
    Compute all Greeks across a range of stock prices (0.7×K to 1.3×K).

    Used to generate the Greeks dashboard charts showing how each Greek
    evolves as the stock price moves relative to the strike.

    Parameters
    ----------
    K           : float — strike price
    T           : float — time to expiry in years
    r           : float — annualised risk-free rate
    sigma       : float — annualised volatility
    option_type : str   — 'call' or 'put'
    n_points    : int   — number of S0 values in the range (default: 100)

    Returns
    -------
    pd.DataFrame with columns: spot, price, delta, gamma, vega, theta
    """
    spot_range = np.linspace(0.7 * K, 1.3 * K, n_points)

    rows = [
        {"spot": s, **greeks(s, K, T, r, sigma, option_type)}
        for s in spot_range
    ]

    return pd.DataFrame(rows)


if __name__ == "__main__":
    S0    = 1380.70
    K     = 1400.0
    T     = 30 / 365
    r     = 0.065
    sigma = 0.2107

    print("=" * 42)
    print("  Greeks — RELIANCE.NS Call Option")
    print("=" * 42)
    print(f"  S0={S0}, K={K}, T=30d, σ={sigma}, r={r}")
    print()

    for opt in ("call", "put"):
        g = greeks(S0, K, T, r, sigma, opt)
        print(f"  {opt.upper()}")
        print(f"    Price : ₹{g['price']:.4f}")
        print(f"    Delta :  {g['delta']:.4f}")
        print(f"    Gamma :  {g['gamma']:.6f}")
        print(f"    Vega  :  {g['vega']:.4f}  (per 1% vol move)")
        print(f"    Theta :  {g['theta']:.4f}  (per year)  →  ₹{g['theta']/365:.4f}/day")
        print()
