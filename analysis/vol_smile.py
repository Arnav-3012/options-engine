"""
analysis/vol_smile.py
---------------------
Implied volatility smile generator.

For each strike in a live option chain, back-solves for the implied
volatility using bisection search (analysis/implied_vol.py). Plots
implied vol vs strike, revealing the volatility smile — the empirical
phenomenon where out-of-the-money options trade at higher implied vol
than at-the-money options, contradicting the flat-vol assumption of
standard GBM / Black-Scholes.

Key finding: OTM puts typically trade 4–6 percentage points higher IV
than ATM options (crash risk premium).
"""

import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go

from data.nse_options import fetch_option_chain
from analysis.implied_vol import implied_vol

# ── Project constants ─────────────────────────────────────────────────────────
RISK_FREE_RATE  = 0.065
CALIBRATED_SIGMA = 0.2107   # flat vol from RELIANCE.NS calibration


def compute_vol_smile(
    chain_df: pd.DataFrame,
    S0: float,
    r: float,
    T: float,
    option_type: str = "call",
) -> pd.DataFrame:
    """
    Compute implied volatility for each strike in an option chain.

    Parameters
    ----------
    chain_df    : pd.DataFrame — cleaned option chain (from nse_options.py)
                  Must have columns: strike, lastPrice, volume, openInterest
    S0          : float — current spot price
    r           : float — annualised risk-free rate
    T           : float — time to expiry in years
    option_type : str   — 'call' or 'put'

    Returns
    -------
    pd.DataFrame with columns:
        strike        — strike price
        market_price  — last traded option price
        implied_vol   — back-solved IV (NaN if outside arbitrage bounds)
        moneyness     — strike / S0  (1.0 = ATM, >1 = OTM call, <1 = OTM put)
        volume        — trading volume
        openInterest  — open interest
    """
    rows = []
    for _, row in chain_df.iterrows():
        iv = implied_vol(
            market_price=float(row["lastPrice"]),
            S0=S0,
            K=float(row["strike"]),
            T=T,
            r=r,
            option_type=option_type,
        )
        rows.append({
            "strike":       float(row["strike"]),
            "market_price": float(row["lastPrice"]),
            "implied_vol":  iv,
            "moneyness":    float(row["strike"]) / S0,
            "volume":       float(row["volume"]),
            "openInterest": float(row["openInterest"]),
        })

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["implied_vol"])
    df = df[df["implied_vol"] > 0.001]   # remove solver failures near zero
    return df.sort_values("strike").reset_index(drop=True)


def plot_vol_smile(
    calls_smile: pd.DataFrame,
    puts_smile: pd.DataFrame,
    S0: float,
    ticker: str,
    expiry: str,
    output_path: str = "outputs/vol_smile.html",
) -> None:
    """
    Plot the volatility smile — implied vol vs strike — using Plotly.

    Shows calls and puts on the same chart, with:
      - Horizontal dashed line: calibrated flat sigma (GBM assumption)
      - Vertical dashed line: current spot price S0
      - Smile shape demonstrates OTM options priced at higher IV

    Parameters
    ----------
    calls_smile : pd.DataFrame — output of compute_vol_smile() for calls
    puts_smile  : pd.DataFrame — output of compute_vol_smile() for puts
    S0          : float        — current spot price
    ticker      : str          — ticker label for chart title
    expiry      : str          — expiry date string for chart title
    output_path : str          — file path for the HTML output
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig = go.Figure()

    # ── Calls ─────────────────────────────────────────────────────────────────
    if not calls_smile.empty:
        fig.add_trace(go.Scatter(
            x=calls_smile["strike"],
            y=calls_smile["implied_vol"] * 100,
            mode="lines+markers",
            name="Calls IV",
            line=dict(color="#636EFA", width=2),
            marker=dict(size=7, symbol="circle"),
            hovertemplate=(
                "Strike: %{x:.2f}<br>"
                "IV: %{y:.2f}%<br>"
                "Moneyness: %{customdata:.3f}<extra></extra>"
            ),
            customdata=calls_smile["moneyness"],
        ))

    # ── Puts ──────────────────────────────────────────────────────────────────
    if not puts_smile.empty:
        fig.add_trace(go.Scatter(
            x=puts_smile["strike"],
            y=puts_smile["implied_vol"] * 100,
            mode="lines+markers",
            name="Puts IV",
            line=dict(color="#EF553B", width=2),
            marker=dict(size=7, symbol="diamond"),
            hovertemplate=(
                "Strike: %{x:.2f}<br>"
                "IV: %{y:.2f}%<br>"
                "Moneyness: %{customdata:.3f}<extra></extra>"
            ),
            customdata=puts_smile["moneyness"],
        ))

    # ── Flat vol reference (GBM assumption) ───────────────────────────────────
    all_strikes = pd.concat([
        calls_smile["strike"] if not calls_smile.empty else pd.Series(dtype=float),
        puts_smile["strike"]  if not puts_smile.empty  else pd.Series(dtype=float),
    ])
    x_min, x_max = all_strikes.min(), all_strikes.max()

    fig.add_shape(type="line",
        x0=x_min, x1=x_max,
        y0=CALIBRATED_SIGMA * 100, y1=CALIBRATED_SIGMA * 100,
        line=dict(color="#00CC96", width=2, dash="dash"),
    )
    fig.add_annotation(
        x=x_max, y=CALIBRATED_SIGMA * 100,
        text=f"  Flat σ = {CALIBRATED_SIGMA*100:.1f}% (GBM)",
        showarrow=False, xanchor="left",
        font=dict(color="#00CC96", size=11),
    )

    # ── Spot price vertical line ───────────────────────────────────────────────
    fig.add_vline(
        x=S0, line_dash="dot", line_color="white", line_width=1.5,
        annotation_text=f"  S0 = {S0:.2f}",
        annotation_position="top right",
        annotation_font=dict(color="white", size=11),
    )

    fig.update_layout(
        title=dict(
            text=(
                f"Volatility Smile — {ticker} Options"
                f"<br><sup>Expiry: {expiry}  |  Spot: {S0:.2f}  |  "
                f"Flat σ = {CALIBRATED_SIGMA*100:.1f}%</sup>"
            ),
            x=0.5,
        ),
        xaxis=dict(title="Strike Price"),
        yaxis=dict(title="Implied Volatility (%)"),
        legend=dict(x=0.02, y=0.98),
        template="plotly_dark",
        width=950, height=580,
    )

    fig.write_html(output_path)
    print(f"Chart saved → {output_path}")


def _print_summary(smile_df: pd.DataFrame, label: str, n: int = 10) -> None:
    """Print a table of the n most liquid strikes."""
    if smile_df.empty:
        print(f"  No valid {label} strikes.")
        return

    top = smile_df.nlargest(n, "openInterest")
    print(f"\n  {label} — top {len(top)} strikes by open interest:")
    print(f"  {'Strike':>10}  {'Mkt Price':>10}  {'IV (%)':>8}  {'Moneyness':>10}  {'OI':>8}")
    print("  " + "-" * 56)
    for _, r in top.iterrows():
        print(
            f"  {r['strike']:>10.2f}  "
            f"{r['market_price']:>10.4f}  "
            f"{r['implied_vol']*100:>7.2f}%  "
            f"{r['moneyness']:>10.4f}  "
            f"{int(r['openInterest']):>8,}"
        )


if __name__ == "__main__":
    print("Fetching option chain …")
    calls_raw, puts_raw, ticker, expiry, S0 = fetch_option_chain("RELIANCE.NS", expiry_index=0)

    # Compute T from expiry string
    days_to_expiry = (datetime.strptime(expiry, "%Y-%m-%d") - datetime.today()).days
    T = max(days_to_expiry, 1) / 365
    print(f"  Days to expiry : {days_to_expiry}  (T = {T:.4f} years)")

    print("\nComputing implied volatilities …")
    calls_smile = compute_vol_smile(calls_raw, S0, RISK_FREE_RATE, T, "call")
    puts_smile  = compute_vol_smile(puts_raw,  S0, RISK_FREE_RATE, T, "put")

    print(f"  Calls with valid IV : {len(calls_smile)}")
    print(f"  Puts  with valid IV : {len(puts_smile)}")

    _print_summary(calls_smile, "CALLS")
    _print_summary(puts_smile,  "PUTS")

    # ── Smile check: OTM vs ATM ────────────────────────────────────────────────
    for label, smile_df, otm_mask in [
        ("CALLS", calls_smile, calls_smile["moneyness"] > 1.05),
        ("PUTS",  puts_smile,  puts_smile["moneyness"]  < 0.95),
    ]:
        if smile_df.empty:
            continue
        atm = smile_df[(smile_df["moneyness"] > 0.97) & (smile_df["moneyness"] < 1.03)]
        otm = smile_df[otm_mask]
        if not atm.empty and not otm.empty:
            atm_iv = atm["implied_vol"].mean() * 100
            otm_iv = otm["implied_vol"].mean() * 100
            print(f"\n  {label} smile: ATM IV = {atm_iv:.2f}%  |  OTM IV = {otm_iv:.2f}%  "
                  f"|  Spread = {otm_iv - atm_iv:+.2f}pp")

    plot_vol_smile(calls_smile, puts_smile, S0, ticker, expiry)
