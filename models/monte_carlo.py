"""
models/monte_carlo.py
---------------------
Standard Monte Carlo pricer for European options.

Pricing steps:
    1. Simulate N terminal prices S_T using risk-neutral GBM
    2. Compute payoff per path:  max(S_T - K, 0)  for call
                                  max(K - S_T, 0)  for put
    3. Average payoffs:          mean_payoff = (1/N) * sum(payoffs)
    4. Discount to present:      price = exp(-r*T) * mean_payoff

Result converges to the Black-Scholes price as N -> infinity.
"""

import numpy as np
from models.gbm import terminal_prices
from models.black_scholes import black_scholes


def mc_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int = 10_000,
    option_type: str = "call",
    seed: int | None = None,
) -> dict:
    """
    Price a European option via standard Monte Carlo simulation.

    Parameters
    ----------
    S0          : float — current stock price
    K           : float — strike price
    T           : float — time to expiry in years (e.g. 30/365)
    r           : float — annualised risk-free rate
    sigma       : float — annualised volatility
    n_paths     : int   — number of simulation paths (default: 10_000)
    option_type : str   — 'call' or 'put' (case-insensitive)
    seed        : int, optional — random seed for reproducibility

    Returns
    -------
    dict with keys:
        price      (float) — MC option price
        std_error  (float) — standard error of the MC estimate
        payoffs    (np.ndarray) — individual path payoffs (for diagnostics)
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    S_T = terminal_prices(S0, r, sigma, T, n_paths, seed=seed)

    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    else:  # put
        payoffs = np.maximum(K - S_T, 0.0)

    discount = np.exp(-r * T)
    price = discount * payoffs.mean()
    std_error = discount * payoffs.std(ddof=1) / np.sqrt(n_paths)

    return {"price": price, "std_error": std_error, "payoffs": payoffs}


def compare_mc_bs(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int = 10_000,
    option_type: str = "call",
    seed: int | None = None,
) -> dict:
    """
    Run MC pricer and compare against the analytical Black-Scholes price.

    Parameters
    ----------
    S0, K, T, r, sigma, n_paths, option_type, seed — same as mc_price()

    Returns
    -------
    dict with keys:
        bs_price    (float) — analytical Black-Scholes price
        mc_price    (float) — Monte Carlo price
        abs_error   (float) — |MC - BS|
        pct_error   (float) — |MC - BS| / BS * 100  (percentage)
        std_error   (float) — MC standard error
    """
    bs = black_scholes(S0, K, T, r, sigma, option_type)["price"]
    mc = mc_price(S0, K, T, r, sigma, n_paths, option_type, seed)

    abs_error = abs(mc["price"] - bs)
    pct_error = abs_error / bs * 100

    return {
        "bs_price":  bs,
        "mc_price":  mc["price"],
        "abs_error": abs_error,
        "pct_error": pct_error,
        "std_error": mc["std_error"],
    }


def mc_antithetic(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int = 10_000,
    option_type: str = "call",
    seed: int | None = None,
) -> dict:
    """
    Price a European option via antithetic variance reduction.

    For every random draw Z ~ N(0,1), also simulate its mirror -Z.
    Because Z and -Z are negatively correlated, their payoff errors
    partially cancel when averaged, reducing estimator variance by ~1.5-2x.

    Implementation:
        paths_a: S_T using +Z
        paths_b: S_T using -Z
        payoff per pair = (payoff(+Z) + payoff(-Z)) / 2
        price = exp(-rT) * mean(paired payoffs)

    Parameters
    ----------
    S0, K, T, r, sigma, n_paths, option_type, seed — same as mc_price()
        n_paths is the number of Z draws; total simulations = 2 * n_paths.

    Returns
    -------
    dict with keys:
        price      (float)      — antithetic MC option price
        std_error  (float)      — standard error of the estimate
        payoffs    (np.ndarray) — paired average payoffs, shape (n_paths,)
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    if seed is not None:
        np.random.seed(seed)

    Z = np.random.standard_normal(n_paths)

    log_term = (r - 0.5 * sigma ** 2) * T
    S_T_pos = S0 * np.exp(log_term + sigma * np.sqrt(T) *  Z)   # +Z paths
    S_T_neg = S0 * np.exp(log_term + sigma * np.sqrt(T) * -Z)   # -Z paths (antithetic)

    if option_type == "call":
        payoffs = (np.maximum(S_T_pos - K, 0.0) + np.maximum(S_T_neg - K, 0.0)) / 2
    else:
        payoffs = (np.maximum(K - S_T_pos, 0.0) + np.maximum(K - S_T_neg, 0.0)) / 2

    discount = np.exp(-r * T)
    price = discount * payoffs.mean()
    std_error = discount * payoffs.std(ddof=1) / np.sqrt(n_paths)

    return {"price": price, "std_error": std_error, "payoffs": payoffs}


if __name__ == "__main__":
    # Test with calibrated RELIANCE.NS parameters
    S0    = 1380.70
    K     = 1400.0
    T     = 30 / 365
    r     = 0.065
    sigma = 0.2107
    N     = 10_000

    print("=" * 45)
    print("  Monte Carlo vs Black-Scholes Comparison")
    print("=" * 45)
    print(f"  S0={S0}, K={K}, T=30d, σ={sigma}, r={r}, N={N:,}")
    print()

    for opt in ("call", "put"):
        result = compare_mc_bs(S0, K, T, r, sigma, N, opt, seed=42)
        print(f"  {opt.upper()}")
        print(f"    BS  price : ₹{result['bs_price']:.4f}")
        print(f"    MC  price : ₹{result['mc_price']:.4f}")
        print(f"    Abs error : ₹{result['abs_error']:.4f}")
        print(f"    Pct error : {result['pct_error']:.3f}%")
        print(f"    Std error : ₹{result['std_error']:.4f}")
        status = "PASS" if result["pct_error"] < 1.0 else "FAIL"
        print(f"    Target <1%: {status}")
        print()
