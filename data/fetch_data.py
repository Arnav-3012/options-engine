"""
data/fetch_data.py
------------------
yfinance data pipeline for the Options Pricing Engine.

Fetches historical daily close prices for NSE-listed stocks and calibrates
the two key GBM parameters:
  - sigma (σ): annualised volatility = std(log returns) * sqrt(252)
  - mu    (μ): annualised drift      = mean(log returns) * 252

Usage:
    python data/fetch_data.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, timedelta

# Project-wide constants
RISK_FREE_RATE = 0.065      # RBI repo rate proxy
TRADING_DAYS   = 252        # NSE annual trading days


def fetch_price_history(ticker: str, period_years: int = 2, end_date: date | None = None) -> pd.Series:
    """
    Download daily adjusted close prices for a given ticker via yfinance.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (e.g. 'RELIANCE.NS', '^NSEI').
    period_years : int
        Number of calendar years of history to fetch (default: 2).
    end_date : date, optional
        End date for the download window. Defaults to today.

    Returns
    -------
    pd.Series
        Daily adjusted closing prices indexed by date, sorted ascending.
    """
    if end_date is None:
        end_date = date.today()

    start_date = end_date - timedelta(days=period_years * 365)

    raw = yf.download(
        ticker,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        auto_adjust=True,
        progress=False,
        multi_level_index=False,   # flat columns — fixes yfinance 0.2.x MultiIndex issue
    )

    if raw.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check symbol and date range.")

    closes = raw["Close"]
    closes.name = "Close"
    closes.index.name = "Date"
    return closes.sort_index()


def calibrate_params(prices: pd.Series) -> dict:
    """
    Calibrate annualised GBM parameters from a price series.

    Uses log returns (continuously compounded), consistent with the
    GBM discretisation:  S(t+dt) = S(t) * exp[(mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z]

    Parameters
    ----------
    prices : pd.Series
        Daily closing price series (must have >= 2 observations).

    Returns
    -------
    dict with keys:
        sigma      (float) — annualised volatility
        mu         (float) — annualised drift
        last_price (float) — most recent closing price (S0 for simulation)
        n_days     (int)   — number of trading days used
    """
    prices = prices.dropna()   # remove incomplete trailing rows (e.g. today's partial data)

    if len(prices) < 2:
        raise ValueError("Need at least 2 price observations to calibrate.")

    log_returns = np.log(prices.values[1:] / prices.values[:-1])   # vectorised, no loops

    sigma = float(log_returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
    mu    = float(log_returns.mean()      * TRADING_DAYS)

    return {
        "sigma":      sigma,
        "mu":         mu,
        "last_price": float(prices.iloc[-1]),
        "n_days":     len(log_returns),
    }


def get_reliance_params() -> dict:
    """
    Convenience wrapper: fetch 2-year RELIANCE.NS history and return
    calibrated GBM parameters, printing a summary to stdout.

    Returns
    -------
    dict — same as calibrate_params(), plus 'ticker' key.
    """
    ticker = "RELIANCE.NS"
    print(f"Fetching 2-year daily close history for {ticker} …")

    prices = fetch_price_history(ticker, period_years=2)
    params = calibrate_params(prices)
    params["ticker"] = ticker

    print("\n--- Calibrated Parameters ---")
    print(f"  Ticker          : {ticker}")
    print(f"  Data points     : {params['n_days']} trading days")
    print(f"  Last close (S0) : ₹{params['last_price']:,.2f}")
    print(f"  Annualised σ    : {params['sigma']:.4f}  ({params['sigma']*100:.2f}%)")
    print(f"  Annualised μ    : {params['mu']:.4f}  ({params['mu']*100:.2f}%)")
    print(f"  Risk-free rate  : {RISK_FREE_RATE:.4f}  ({RISK_FREE_RATE*100:.2f}%)")
    print("-----------------------------\n")

    return params


if __name__ == "__main__":
    params = get_reliance_params()
