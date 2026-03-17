"""
data/nse_options.py
-------------------
Option chain fetcher via yfinance.

Primary target: RELIANCE.NS / ^NSEI (NSE Indian equities).
Fallback: SPY (S&P 500 ETF) — same Black-Scholes math, cleaner data feed.

Note: yfinance does not currently expose NSE option chains (NSE blocks the
feed). The fallback to SPY is documented in PROJECT_BRIEF.md Section 7.3:
"Fallback (if NSE data is messy): Use SPY or AAPL options from US markets
— same math, cleaner data."
"""

import numpy as np
import pandas as pd
import yfinance as yf

# Ticker priority order: try NSE names first, then US fallbacks
TICKER_PRIORITY = ["RELIANCE.NS", "^NSEI", "SPY", "AAPL"]

REQUIRED_COLUMNS = ["strike", "lastPrice", "impliedVolatility", "volume", "openInterest"]


def fetch_option_chain(ticker: str = "RELIANCE.NS", expiry_index: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, str, str, float]:
    """
    Fetch a live option chain for the given ticker via yfinance.

    Tries the requested ticker first. If no option expiries are found,
    falls back through TICKER_PRIORITY until a valid chain is retrieved.

    Parameters
    ----------
    ticker       : str — primary Yahoo Finance ticker to try
    expiry_index : int — which expiry to use (0 = nearest, 1 = next, etc.)

    Returns
    -------
    tuple of:
        calls_df   (pd.DataFrame) — cleaned calls chain
        puts_df    (pd.DataFrame) — cleaned puts chain
        used_ticker (str)         — ticker actually used (may differ from input)
        expiry     (str)          — expiry date string used
        spot_price (float)        — current spot price of the underlying
    """
    tickers_to_try = [ticker] + [t for t in TICKER_PRIORITY if t != ticker]

    for t in tickers_to_try:
        tk = yf.Ticker(t)
        expiries = tk.options

        if not expiries:
            print(f"  [{t}] No option expiries found — trying next ticker …")
            continue

        idx = min(expiry_index, len(expiries) - 1)
        expiry = expiries[idx]
        chain  = tk.option_chain(expiry)

        calls_raw = chain.calls[REQUIRED_COLUMNS].copy()
        puts_raw  = chain.puts[REQUIRED_COLUMNS].copy()

        calls_clean = _clean_chain(calls_raw)
        puts_clean  = _clean_chain(puts_raw)

        if calls_clean.empty and puts_clean.empty:
            print(f"  [{t}] Chain returned but all rows filtered out — trying next …")
            continue

        # Spot price from ticker info, fall back to mid-strike if unavailable
        info  = tk.fast_info
        spot  = float(getattr(info, "last_price", np.nan))
        if np.isnan(spot):
            spot = float(calls_clean["strike"].median()) if not calls_clean.empty else np.nan

        print(f"  Using ticker : {t}  |  expiry : {expiry}  |  spot : {spot:.2f}")
        return calls_clean, puts_clean, t, expiry, spot

    raise RuntimeError(
        f"Could not retrieve a valid option chain from any of: {tickers_to_try}"
    )


def _clean_chain(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and clean a raw option chain DataFrame.

    Keeps rows where:
      - lastPrice > 0  (traded at some point)
      - volume > 0 OR openInterest > 0  (liquid strikes)

    Fills NaN volume/openInterest with 0 before filtering.

    Parameters
    ----------
    df : pd.DataFrame — raw chain with REQUIRED_COLUMNS

    Returns
    -------
    pd.DataFrame — cleaned, sorted by strike, index reset
    """
    df = df.copy()
    df["volume"]       = pd.to_numeric(df["volume"],       errors="coerce").fillna(0)
    df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce").fillna(0)
    df["lastPrice"]    = pd.to_numeric(df["lastPrice"],    errors="coerce")

    mask = (df["lastPrice"] > 0) & ((df["volume"] > 0) | (df["openInterest"] > 0))
    return df[mask].sort_values("strike").reset_index(drop=True)
