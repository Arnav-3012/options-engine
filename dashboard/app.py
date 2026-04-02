"""
dashboard/app.py
----------------
Full four-tab Streamlit dashboard for the Options Pricing Engine.
Includes live RELIANCE.NS price feed, intraday chart, and auto-refresh.

Tabs:
    1. Pricing & Greeks  — summary table, metric cards, Greek charts, payoff diagram
    2. Simulation        — GBM path fan chart + terminal price histogram
    3. Convergence       — Standard MC vs antithetic convergence analysis
    4. Volatility Smile  — IV smile, model limits, IV lookup tool

Run with:
    streamlit run dashboard/app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.stats import lognorm
from datetime import datetime

from models.black_scholes import black_scholes
from models.greeks import greeks, greeks_vs_spot
from models.gbm import simulate_gbm, terminal_prices
from models.monte_carlo import mc_price, mc_antithetic
from analysis.convergence import run_convergence, variance_reduction_ratio, PATH_COUNTS
from data.nse_options import fetch_option_chain
from data.fetch_data import fetch_price_history, calibrate_params
from analysis.vol_smile import compute_vol_smile

# ── Constants ──────────────────────────────────────────────────────────────────
CAL_SIGMA    = 0.2107
CAL_S0       = 1380.70
CAL_MU       = -0.0176
CAL_R        = 0.065
VAR_RED      = 2.19
ATM_IV       = 22.50
OTM_PUT_IV   = 91.20
SMILE_SPREAD = 68.70

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Options Pricing Engine", page_icon="📈", layout="wide")
st.title("📈 Options Pricing Engine — GBM + Monte Carlo")
st.caption("RELIANCE.NS · STPA College Project · Antithetic Variance Reduction + Volatility Smile")

# ══════════════════════════════════════════════════════════════════════════════
# LIVE FEED CACHED FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_price() -> dict:
    """
    Fetch current RELIANCE.NS price from yfinance (15-min delay for free feed).
    Cached for 60 seconds. Falls back to CAL_S0 silently on any error.

    Returns dict with keys: price, prev_close, change, change_pct, timestamp, success.
    """
    try:
        info       = yf.Ticker("RELIANCE.NS").fast_info
        price      = float(info.last_price)
        prev_close = float(info.previous_close)
        change     = price - prev_close
        change_pct = change / prev_close * 100 if prev_close else 0.0
        return {
            "price":      price,
            "prev_close": prev_close,
            "change":     change,
            "change_pct": change_pct,
            "timestamp":  datetime.now().strftime("%H:%M:%S"),
            "success":    True,
        }
    except Exception:
        return {
            "price":      CAL_S0,
            "prev_close": CAL_S0,
            "change":     0.0,
            "change_pct": 0.0,
            "timestamp":  "—",
            "success":    False,
        }


@st.cache_data(ttl=300, show_spinner=False)
def fetch_intraday() -> pd.DataFrame:
    """
    Fetch 1-minute intraday close prices for RELIANCE.NS for today.
    Cached for 5 minutes. Returns empty DataFrame on failure.
    """
    try:
        df = yf.download(
            "RELIANCE.NS", period="1d", interval="1m",
            progress=False, multi_level_index=False,
        )
        if df.empty:
            return pd.DataFrame()
        return df[["Close"]].dropna()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_sigma() -> dict:
    """
    Recalibrate sigma and mu from fresh 2-year RELIANCE.NS data.
    Cached for 1 hour — sigma doesn't change minute to minute.

    Returns dict with keys: sigma, mu, last_price, n_days, success.
    """
    try:
        prices = fetch_price_history("RELIANCE.NS", period_years=2)
        params = calibrate_params(prices)
        params["success"] = True
        return params
    except Exception:
        return {
            "sigma":      CAL_SIGMA,
            "mu":         CAL_MU,
            "last_price": CAL_S0,
            "n_days":     0,
            "success":    False,
        }


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════
if "live_price_data"   not in st.session_state:
    st.session_state.live_price_data   = None
if "live_sigma_data"   not in st.session_state:
    st.session_state.live_sigma_data   = None
if "last_auto_refresh" not in st.session_state:
    st.session_state.last_auto_refresh = time.time()
if "s0_override"       not in st.session_state:
    st.session_state.s0_override       = None
if "sigma_override"    not in st.session_state:
    st.session_state.sigma_override    = None

# ── Convenience aliases ────────────────────────────────────────────────────────
live_price_data = st.session_state.live_price_data
live_sigma_data = st.session_state.live_sigma_data

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Live Feed section ──────────────────────────────────────────────────────
    st.header("📡 Live Feed")
    auto_refresh = st.toggle("Auto-refresh price", value=False)

    refresh_interval = 60
    if auto_refresh:
        refresh_interval = st.selectbox(
            "Refresh interval (seconds)", [30, 60, 120, 300], index=1
        )
        elapsed   = time.time() - st.session_state.last_auto_refresh
        remaining = max(0, int(refresh_interval - elapsed))
        st.caption(f"Next refresh in {remaining}s")

    st.divider()

    # ── Parameter sliders ──────────────────────────────────────────────────────
    st.header("Parameters")

    s0_default    = max(500,  min(2500, int(st.session_state.s0_override)))    \
                    if st.session_state.s0_override    is not None else 1380
    sigma_default = max(5,    min(100,  int(st.session_state.sigma_override))) \
                    if st.session_state.sigma_override is not None else 21

    S0        = st.slider("Stock Price S0 (₹)",      500,  2500, s0_default,    10)
    K         = st.slider("Strike Price K (₹)",       500,  2500, 1400,          10)
    T_days    = st.slider("Time to Expiry (days)",      1,   365,   30,           1)
    sigma_pct = st.slider("Volatility σ (%)",           5,   100, sigma_default,  1)
    r_pct     = st.slider("Risk-free Rate r (%)",       1,    15,    7,           1)
    N_paths   = st.slider("Simulation Paths N",      1000, 20000, 10000,        500)
    option_type = st.radio("Option Type", ["call", "put"], index=0)

    st.divider()

    # ── Calibration stats ──────────────────────────────────────────────────────
    st.markdown("**Calibrated (RELIANCE.NS)**")
    if live_sigma_data and live_sigma_data["success"]:
        st.caption(f"σ = {live_sigma_data['sigma']*100:.2f}%  ⬅ LIVE")
        st.caption(f"μ = {live_sigma_data['mu']*100:.2f}%  ⬅ LIVE")
        st.caption(f"S0 = ₹{live_sigma_data['last_price']:,.2f}  ⬅ LIVE")
        st.caption(f"({live_sigma_data['n_days']} trading days)")
    else:
        st.caption(f"σ = {CAL_SIGMA*100:.2f}%  (cached)")
        st.caption(f"S0 = ₹{CAL_S0:,.2f}  (cached)")
        st.caption(f"μ = {CAL_MU*100:.2f}%  (cached)")
    st.caption(f"Variance reduction = {VAR_RED}x")
    st.caption(f"ATM implied vol = {ATM_IV}%")

# ── Derived inputs ─────────────────────────────────────────────────────────────
T     = T_days / 365
sigma = sigma_pct / 100
r     = r_pct / 100

# active_sigma: use live-recalibrated value only if user clicked "Recalibrate σ"
active_sigma = (
    live_sigma_data["sigma"]
    if (live_sigma_data and live_sigma_data["success"] and st.session_state.sigma_override is not None)
    else sigma
)

# ── Pre-compute prices + Greeks (uses active_sigma) ────────────────────────────
call_bs   = black_scholes(S0, K, T, r, active_sigma, "call")
put_bs    = black_scholes(S0, K, T, r, active_sigma, "put")
g         = greeks(S0, K, T, r, active_sigma, option_type)
mc_result = mc_price(S0, K, T, r, active_sigma, N_paths, option_type, seed=42)
pct_err   = abs(mc_result["price"] - call_bs["price"]) / call_bs["price"] * 100

# ── Cached heavy computations ──────────────────────────────────────────────────
@st.cache_data(show_spinner="Running convergence analysis …")
def cached_convergence():
    results = run_convergence(n_repeats=30)
    ratio   = variance_reduction_ratio(results)
    return results, ratio

@st.cache_data(show_spinner="Fetching option chain …")
def cached_smile():
    try:
        # Try expiry indices 0, 1, 2 — skip any expiry that is today or
        # has fewer than 3 calendar days remaining (IV solver breaks near T=0)
        MIN_DAYS = 3
        for idx in range(3):
            calls_raw, puts_raw, ticker, expiry, spot = fetch_option_chain("RELIANCE.NS", idx)
            days = (datetime.strptime(expiry, "%Y-%m-%d") - datetime.today()).days
            if days >= MIN_DAYS:
                break
        T_exp = max(days, MIN_DAYS) / 365
        calls_smile = compute_vol_smile(calls_raw, spot, CAL_R, T_exp, "call")
        puts_smile  = compute_vol_smile(puts_raw,  spot, CAL_R, T_exp, "put")
        # If both come back empty, the smile failed — return None to show warning
        if calls_smile.empty and puts_smile.empty:
            return None, None, "N/A", "N/A", CAL_S0, 30/365
        return calls_smile, puts_smile, ticker, expiry, spot, T_exp
    except Exception:
        return None, None, "N/A", "N/A", CAL_S0, 30/365

# ══════════════════════════════════════════════════════════════════════════════
# TICKER BAR
# ══════════════════════════════════════════════════════════════════════════════
tb1, tb2, tb3, tb4, tb5 = st.columns([1.2, 1.4, 1.6, 2, 1.8])

with tb1:
    if st.button("🔄 Fetch Live Price", use_container_width=True):
        fetch_live_price.clear()
        fetch_intraday.clear()
        result = fetch_live_price()
        st.session_state.live_price_data = result
        st.session_state.last_auto_refresh = time.time()
        st.rerun()

with tb2:
    if st.button("📐 Sync S0 to Live Price", use_container_width=True):
        if live_price_data and live_price_data["success"]:
            st.session_state.s0_override = int(live_price_data["price"])
            st.rerun()
        else:
            st.toast("Fetch live price first.", icon="⚠️")

with tb3:
    if st.button("📊 Recalibrate σ (live data)", use_container_width=True):
        fetch_live_sigma.clear()
        result = fetch_live_sigma()
        st.session_state.live_sigma_data = result
        live_sigma_data = result
        if result["success"]:
            st.session_state.sigma_override = int(result["sigma"] * 100)
        st.rerun()

with tb4:
    if live_price_data and live_price_data["success"]:
        price  = live_price_data["price"]
        chg    = live_price_data["change"]
        chg_p  = live_price_data["change_pct"]
        colour = "#00CC96" if chg >= 0 else "#EF553B"
        arrow  = "▲" if chg >= 0 else "▼"
        st.markdown(
            f"""<div style="padding:6px 0;">
                <span style="font-size:22px; font-weight:700; color:{colour};">
                    ₹{price:,.2f}
                </span>&nbsp;&nbsp;
                <span style="font-size:14px; color:{colour};">
                    {arrow} ₹{abs(chg):.2f} ({chg_p:+.2f}%)
                </span>
                <div style="font-size:11px; color:#888;">RELIANCE.NS · 15-min delay</div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""<div style="padding:6px 0;">
                <span style="font-size:18px; color:#888;">₹{CAL_S0:,.2f}</span>
                <div style="font-size:11px; color:#888;">No live data — using calibrated S0</div>
            </div>""",
            unsafe_allow_html=True,
        )

with tb5:
    if live_price_data and live_price_data["success"]:
        st.markdown(
            f"""<div style="padding:6px 0; font-size:12px; color:#aaa;">
                🟢 Live (15-min delay)<br>
                Last fetch: {live_price_data['timestamp']}
            </div>""",
            unsafe_allow_html=True,
        )
    elif live_price_data and not live_price_data["success"]:
        st.markdown(
            """<div style="padding:6px 0; font-size:12px; color:#EF553B;">
                ⚠ Fetch failed<br>Using fallback S0
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """<div style="padding:6px 0; font-size:12px; color:#888;">
                — Not fetched yet<br>Click 🔄 to load
            </div>""",
            unsafe_allow_html=True,
        )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# INTRADAY CHART EXPANDER
# ══════════════════════════════════════════════════════════════════════════════
if live_price_data and live_price_data["success"]:
    with st.expander("📈 Today's Intraday Price Chart"):
        intraday_df = fetch_intraday()
        if intraday_df.empty:
            st.info("Intraday data unavailable — market may be closed or outside 9:15 AM – 3:30 PM IST.")
        else:
            prev_close = live_price_data["prev_close"]
            last_price = float(intraday_df["Close"].iloc[-1])
            line_colour = "#00CC96" if last_price >= prev_close else "#EF553B"

            fig_intra = go.Figure()
            fig_intra.add_trace(go.Scatter(
                x=intraday_df.index,
                y=intraday_df["Close"],
                mode="lines",
                line=dict(color=line_colour, width=1.8),
                name="RELIANCE.NS",
            ))
            fig_intra.add_hline(
                y=prev_close,
                line_dash="dot", line_color="white", line_width=1,
                annotation_text=f"  Prev close ₹{prev_close:,.2f}",
                annotation_position="right",
                annotation_font=dict(color="white", size=10),
            )
            fig_intra.update_layout(
                title=dict(text="RELIANCE.NS — Today (1-min intervals)", x=0.5),
                xaxis_title="Time", yaxis_title="Price (₹)",
                template="plotly_dark", height=300,
                margin=dict(l=50, r=20, t=50, b=40),
                showlegend=False,
            )
            st.plotly_chart(fig_intra, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Pricing & Greeks",
    "🎲 Simulation",
    "📉 Convergence",
    "😊 Volatility Smile",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Pricing & Greeks
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── Summary table ──────────────────────────────────────────────────────────
    st.subheader("Project Summary")
    summary = {
        "Ticker":                "RELIANCE.NS (SPY for options)",
        "Calibration period":    "2 years daily closes",
        "S0 (last close)":       f"₹{CAL_S0:,.2f}",
        "Annualised σ":          f"{CAL_SIGMA*100:.2f}%",
        "Annualised μ":          f"{CAL_MU*100:.2f}%",
        "Risk-free rate r":      f"{CAL_R*100:.1f}%",
        "BS call price":         f"₹{call_bs['price']:.4f}",
        "MC call price":         f"₹{mc_result['price']:.4f}",
        "MC vs BS error":        f"{pct_err:.3f}%",
        "Variance reduction":    f"{VAR_RED}x",
        "ATM implied vol":       f"{ATM_IV}%",
        "OTM put implied vol":   f"{OTM_PUT_IV}%",
        "Smile spread":          f"+{SMILE_SPREAD:.1f}pp",
    }
    col_a, col_b = st.columns(2)
    items = list(summary.items())
    half  = len(items) // 2 + len(items) % 2
    with col_a:
        st.table(pd.DataFrame(items[:half],  columns=["Metric", "Value"]).set_index("Metric"))
    with col_b:
        st.table(pd.DataFrame(items[half:], columns=["Metric", "Value"]).set_index("Metric"))

    st.divider()

    # ── Price + Greeks metric cards ────────────────────────────────────────────
    st.subheader("Current Prices")
    p1, p2 = st.columns(2)
    p1.metric("Call Price",  f"₹{call_bs['price']:.2f}")
    p2.metric("Put Price",   f"₹{put_bs['price']:.2f}")

    st.subheader(f"Greeks  ({option_type.upper()})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Delta (Δ)", f"{g['delta']:.4f}",
              help="₹ change per ₹1 move in stock")
    c2.metric("Gamma (Γ)", f"{g['gamma']:.6f}",
              help="Rate of change of Delta per ₹1 move")
    c3.metric("Vega (ν)",  f"{g['vega']:.2f}",
              help="₹ change per 1pp vol move")
    c4.metric("Theta (Θ)", f"₹{g['theta']/365:.4f}/day",
              help="₹ change per calendar day (negative = time decay)")

    st.divider()

    # ── Greeks charts ──────────────────────────────────────────────────────────
    st.subheader(f"Greeks vs Stock Price  ({option_type.upper()}, K=₹{K}, T={T_days}d, σ={sigma_pct}%)")
    gdf = greeks_vs_spot(K, T, r, active_sigma, option_type)

    GREEK_CFG = [
        ("delta", "Delta (Δ)", "₹ per ₹1 move",   "#636EFA"),
        ("gamma", "Gamma (Γ)", "ΔDelta per ₹1",    "#EF553B"),
        ("vega",  "Vega (ν)",  "₹ per 1% σ move",  "#00CC96"),
        ("theta", "Theta (Θ)", "₹ per day",         "#AB63FA"),
    ]
    gl, gr = st.columns(2)
    for i, (col_name, title, y_label, colour) in enumerate(GREEK_CFG):
        y_vals = gdf[col_name] if col_name != "theta" else gdf["theta"] / 365
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=gdf["spot"], y=y_vals, mode="lines",
                                 line=dict(color=colour, width=2.5)))
        fig.add_vline(x=S0, line_dash="dot", line_color="white", line_width=1.5,
                      annotation_text=f"  S0=₹{S0}", annotation_position="top right",
                      annotation_font=dict(color="white", size=10))
        fig.update_layout(title=dict(text=title, x=0.5),
                          xaxis_title="Stock Price (₹)", yaxis_title=y_label,
                          template="plotly_dark", height=340,
                          margin=dict(l=50, r=20, t=50, b=40), showlegend=False)
        (gl if i % 2 == 0 else gr).plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Payoff diagram ─────────────────────────────────────────────────────────
    st.subheader("Payoff at Expiry")
    call_prem  = call_bs["price"]
    put_prem   = put_bs["price"]
    s_range    = np.linspace(0.6 * K, 1.4 * K, 300)
    call_pnl   = np.maximum(s_range - K, 0) - call_prem
    put_pnl    = np.maximum(K - s_range, 0) - put_prem
    call_be    = K + call_prem
    put_be     = K - put_prem

    fig_pay = go.Figure()
    fig_pay.add_trace(go.Scatter(x=s_range, y=call_pnl, mode="lines",
                                 name="Call P&L", line=dict(color="#636EFA", width=2.5)))
    fig_pay.add_trace(go.Scatter(x=s_range, y=put_pnl,  mode="lines",
                                 name="Put P&L",  line=dict(color="#EF553B", width=2.5)))
    fig_pay.add_hline(y=0, line_color="white", line_width=1, line_dash="dash")
    fig_pay.add_vline(x=S0, line_dash="dot", line_color="white", line_width=1.5,
                      annotation_text=f"  S0=₹{S0}", annotation_position="top right",
                      annotation_font=dict(color="white", size=10))
    fig_pay.add_vline(x=call_be, line_dash="dot", line_color="#636EFA", line_width=1,
                      annotation_text=f"  Call BE ₹{call_be:.0f}",
                      annotation_position="bottom right",
                      annotation_font=dict(color="#636EFA", size=10))
    fig_pay.add_vline(x=put_be, line_dash="dot", line_color="#EF553B", line_width=1,
                      annotation_text=f"Put BE ₹{put_be:.0f}  ",
                      annotation_position="bottom left",
                      annotation_font=dict(color="#EF553B", size=10))
    fig_pay.update_layout(
        title=dict(text=f"Profit / Loss at Expiry  (K=₹{K})", x=0.5),
        xaxis_title="Stock Price at Expiry (₹)", yaxis_title="P&L (₹)",
        template="plotly_dark", height=420, legend=dict(x=0.02, y=0.98))
    st.plotly_chart(fig_pay, use_container_width=True)

    st.caption(
        f"d₁={call_bs['d1']:.4f}  |  d₂={call_bs['d2']:.4f}  |  "
        f"T={T:.4f}yr  |  σ={active_sigma:.4f}  |  r={r:.4f}"
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Simulation
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    # ── Path fan chart ─────────────────────────────────────────────────────────
    st.subheader("GBM Price Path Simulation")
    n_steps   = T_days
    all_paths = simulate_gbm(S0, r, active_sigma, T, n_steps, N_paths, seed=7)

    time_axis = np.arange(n_steps + 1)
    p5  = np.percentile(all_paths, 5,  axis=1)
    p95 = np.percentile(all_paths, 95, axis=1)

    fig_fan = go.Figure()

    # Shaded band
    fig_fan.add_trace(go.Scatter(
        x=np.concatenate([time_axis, time_axis[::-1]]),
        y=np.concatenate([p95, p5[::-1]]),
        fill="toself", fillcolor="rgba(99,110,250,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="5th–95th percentile", showlegend=True,
    ))

    # 200 sample paths
    for j in range(min(200, N_paths)):
        fig_fan.add_trace(go.Scatter(
            x=time_axis, y=all_paths[:, j],
            mode="lines", line=dict(color="rgba(99,110,250,0.15)", width=0.6),
            showlegend=False, hoverinfo="skip",
        ))

    fig_fan.add_hline(y=K,  line_color="#EF553B", line_dash="dash", line_width=1.5,
                      annotation_text=f"  Strike K=₹{K}", annotation_position="right",
                      annotation_font=dict(color="#EF553B"))
    fig_fan.add_hline(y=S0, line_color="#00CC96", line_dash="dash", line_width=1.5,
                      annotation_text=f"  S0=₹{S0}", annotation_position="right",
                      annotation_font=dict(color="#00CC96"))

    fig_fan.update_layout(
        title=dict(text="Simulated RELIANCE.NS Price Paths — GBM Monte Carlo", x=0.5),
        xaxis_title="Days", yaxis_title="Stock Price (₹)",
        template="plotly_dark", height=480,
        legend=dict(x=0.02, y=0.98),
    )
    st.plotly_chart(fig_fan, use_container_width=True)

    st.divider()

    # ── Terminal price histogram ───────────────────────────────────────────────
    st.subheader("Distribution of Terminal Prices S_T")

    S_T = all_paths[-1, :]
    prob_itm_call = float(np.mean(S_T > K))
    prob_itm_put  = float(np.mean(S_T < K))

    # Log-normal PDF overlay
    mu_ln  = np.log(S0) + (r - 0.5 * active_sigma ** 2) * T
    sig_ln = active_sigma * np.sqrt(T)
    x_pdf  = np.linspace(S_T.min(), S_T.max(), 300)
    y_pdf  = lognorm.pdf(x_pdf, s=sig_ln, scale=np.exp(mu_ln))
    scale_factor = len(S_T) * (S_T.max() - S_T.min()) / 80

    fig_hist = go.Figure()

    # Green area (call ITM: S_T > K)
    x_green = x_pdf[x_pdf >= K]
    y_green = lognorm.pdf(x_green, s=sig_ln, scale=np.exp(mu_ln)) * scale_factor
    fig_hist.add_trace(go.Scatter(
        x=np.concatenate([[x_green[0]], x_green, [x_green[-1]]]),
        y=np.concatenate([[0], y_green, [0]]),
        fill="toself", fillcolor="rgba(0,204,150,0.25)",
        line=dict(color="rgba(0,0,0,0)"), name="Call ITM region", showlegend=True,
    ))

    # Red area (put ITM: S_T < K)
    x_red  = x_pdf[x_pdf <= K]
    y_red  = lognorm.pdf(x_red, s=sig_ln, scale=np.exp(mu_ln)) * scale_factor
    fig_hist.add_trace(go.Scatter(
        x=np.concatenate([[x_red[0]], x_red, [x_red[-1]]]),
        y=np.concatenate([[0], y_red, [0]]),
        fill="toself", fillcolor="rgba(239,85,59,0.20)",
        line=dict(color="rgba(0,0,0,0)"), name="Put ITM region", showlegend=True,
    ))

    fig_hist.add_trace(go.Histogram(
        x=S_T, nbinsx=80, name="Simulated S_T",
        marker=dict(color="rgba(99,110,250,0.6)", line=dict(color="rgba(99,110,250,0.9)", width=0.3)),
    ))

    fig_hist.add_trace(go.Scatter(
        x=x_pdf, y=y_pdf * scale_factor, mode="lines",
        name="Log-normal PDF", line=dict(color="white", width=2, dash="dot"),
    ))

    fig_hist.add_vline(x=K, line_color="#EF553B", line_dash="dash", line_width=2,
                       annotation_text=f"  K=₹{K}", annotation_position="top right",
                       annotation_font=dict(color="#EF553B"))

    fig_hist.add_annotation(
        x=K * 1.06, y=max(y_pdf * scale_factor) * 0.85,
        text=f"P(call ITM) = {prob_itm_call:.1%}",
        showarrow=False, font=dict(color="#00CC96", size=12),
        bgcolor="rgba(0,0,0,0.5)",
    )
    fig_hist.add_annotation(
        x=K * 0.91, y=max(y_pdf * scale_factor) * 0.85,
        text=f"P(put ITM) = {prob_itm_put:.1%}",
        showarrow=False, font=dict(color="#EF553B", size=12),
        bgcolor="rgba(0,0,0,0.5)",
    )

    fig_hist.update_layout(
        title=dict(text=f"Distribution of S_T  (N={N_paths:,}, T={T_days}d)", x=0.5),
        xaxis_title="Terminal Price S_T (₹)", yaxis_title="Count",
        template="plotly_dark", height=440, barmode="overlay",
        legend=dict(x=0.02, y=0.98),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Convergence
# ══════════════════════════════════════════════════════════════════════════════
with tab3:

    st.subheader("Antithetic Variance Reduction — Convergence Analysis")

    conv_results, conv_ratio = cached_convergence()
    bs_true = black_scholes(CAL_S0, 1400, 30/365, CAL_R, CAL_SIGMA, "call")["price"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Variance Reduction Ratio", f"{conv_ratio:.2f}x",
              help="Var(standard MC) / Var(antithetic MC) at N=10,000")
    m2.metric("Black-Scholes True Price", f"₹{bs_true:.4f}")
    m3.metric("Equivalent path saving",
              f"~{int(10000 * (1 - 1/conv_ratio)):,} paths",
              help="Antithetic achieves same accuracy with fewer paths")

    st.info(
        "**What this chart proves:** Standard Monte Carlo is noisy — with the same number of paths "
        "you get a different answer each run. Antithetic variance reduction mirrors every random draw Z "
        "with its negative −Z. Because the two paths are negatively correlated, their errors partially "
        "cancel. The result: the antithetic estimator converges faster and with less noise. "
        f"At N=10,000 paths we achieve a **{conv_ratio:.1f}x variance reduction** — meaning you need "
        f"only 1/{conv_ratio:.1f} as many paths to get the same accuracy."
    )

    ns        = conv_results["path_counts"]
    std_mean  = conv_results["std_prices"].mean(axis=1)
    std_std   = conv_results["std_prices"].std(axis=1, ddof=1)
    anti_mean = conv_results["anti_prices"].mean(axis=1)
    anti_std  = conv_results["anti_prices"].std(axis=1, ddof=1)

    fig_conv = go.Figure()

    # Standard MC band
    fig_conv.add_trace(go.Scatter(
        x=ns + ns[::-1],
        y=list(std_mean + std_std) + list((std_mean - std_std)[::-1]),
        fill="toself", fillcolor="rgba(99,110,250,0.15)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ))
    fig_conv.add_trace(go.Scatter(
        x=ns, y=std_mean, mode="lines+markers",
        name="Standard MC", line=dict(color="#636EFA", width=2), marker=dict(size=7),
    ))

    # Antithetic band
    fig_conv.add_trace(go.Scatter(
        x=ns + ns[::-1],
        y=list(anti_mean + anti_std) + list((anti_mean - anti_std)[::-1]),
        fill="toself", fillcolor="rgba(0,204,150,0.15)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ))
    fig_conv.add_trace(go.Scatter(
        x=ns, y=anti_mean, mode="lines+markers",
        name="Antithetic MC", line=dict(color="#00CC96", width=2), marker=dict(size=7),
    ))

    fig_conv.add_trace(go.Scatter(
        x=ns, y=[bs_true] * len(ns), mode="lines",
        name=f"Black-Scholes (₹{bs_true:.2f})",
        line=dict(color="#EF553B", width=2, dash="dash"),
    ))

    fig_conv.update_layout(
        title=dict(
            text=(f"Standard MC vs Antithetic Convergence — RELIANCE.NS Call Option"
                  f"<br><sup>Variance reduction ratio: {conv_ratio:.2f}x  |  "
                  f"S0=₹{CAL_S0}, K=₹1400, T=30d, σ={CAL_SIGMA*100:.2f}%</sup>"),
            x=0.5),
        xaxis=dict(title="Number of Paths (N)", type="log"),
        yaxis=dict(title="Estimated Call Price (₹)"),
        template="plotly_dark", height=480, legend=dict(x=0.75, y=0.05),
    )
    st.plotly_chart(fig_conv, use_container_width=True)

    # Convergence table
    st.subheader("Convergence Table")
    ratio_per_n = (conv_results["std_prices"].std(axis=1, ddof=1) ** 2 /
                   conv_results["anti_prices"].std(axis=1, ddof=1) ** 2)
    conv_table = pd.DataFrame({
        "N Paths":      ns,
        "Std MC Mean":  [f"₹{v:.4f}" for v in std_mean],
        "Anti MC Mean": [f"₹{v:.4f}" for v in anti_mean],
        "Std MC σ":     [f"{v:.4f}" for v in std_std],
        "Anti MC σ":    [f"{v:.4f}" for v in anti_std],
        "Var Reduction": [f"{v:.2f}x" for v in ratio_per_n],
    })
    st.dataframe(conv_table, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Volatility Smile
# ══════════════════════════════════════════════════════════════════════════════
with tab4:

    st.subheader("Volatility Smile — Implied Vol vs Strike")

    calls_smile, puts_smile, smile_ticker, smile_expiry, smile_spot, smile_T = cached_smile()

    if calls_smile is None:
        st.warning("Could not fetch option chain data.")
    else:
        # Metric cards
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Your flat σ assumption",  f"{sigma_pct}%",
                  help="From the sidebar slider — updates live")
        s2.metric("ATM market implied vol",  f"{ATM_IV}%")
        s3.metric("OTM put implied vol",      f"{OTM_PUT_IV}%")
        s4.metric("Smile spread",            f"+{SMILE_SPREAD:.1f}pp",
                  help="OTM put IV minus ATM IV — the crash risk premium")

        # Vol smile chart
        all_strikes = pd.concat([
            calls_smile["strike"] if not calls_smile.empty else pd.Series(dtype=float),
            puts_smile["strike"]  if not puts_smile.empty  else pd.Series(dtype=float),
        ])
        x_min, x_max = float(all_strikes.min()), float(all_strikes.max())

        fig_smile = go.Figure()

        if not calls_smile.empty:
            fig_smile.add_trace(go.Scatter(
                x=calls_smile["strike"], y=calls_smile["implied_vol"] * 100,
                mode="lines+markers", name="Calls IV",
                line=dict(color="#636EFA", width=2), marker=dict(size=6),
                hovertemplate="Strike: %{x:.2f}<br>IV: %{y:.2f}%<extra></extra>",
            ))

        if not puts_smile.empty:
            fig_smile.add_trace(go.Scatter(
                x=puts_smile["strike"], y=puts_smile["implied_vol"] * 100,
                mode="lines+markers", name="Puts IV",
                line=dict(color="#EF553B", width=2), marker=dict(size=6, symbol="diamond"),
                hovertemplate="Strike: %{x:.2f}<br>IV: %{y:.2f}%<extra></extra>",
            ))

        # Flat sigma line — updates with slider
        fig_smile.add_shape(type="line",
            x0=x_min, x1=x_max,
            y0=sigma_pct, y1=sigma_pct,
            line=dict(color="#00CC96", width=2, dash="dash"),
        )
        fig_smile.add_annotation(
            x=x_max, y=sigma_pct,
            text=f"  Flat σ = {sigma_pct}% (your model)",
            showarrow=False, xanchor="left",
            font=dict(color="#00CC96", size=11),
        )

        fig_smile.add_vline(x=smile_spot, line_dash="dot", line_color="white",
                            line_width=1.5,
                            annotation_text=f"  S0={smile_spot:.2f}",
                            annotation_position="top right",
                            annotation_font=dict(color="white", size=11))

        fig_smile.update_layout(
            title=dict(
                text=(f"Volatility Smile — {smile_ticker} Options"
                      f"<br><sup>Expiry: {smile_expiry}  |  "
                      f"Spot: {smile_spot:.2f}  |  Flat σ = {sigma_pct}%</sup>"),
                x=0.5,
            ),
            xaxis_title="Strike Price", yaxis_title="Implied Volatility (%)",
            template="plotly_dark", height=480, legend=dict(x=0.02, y=0.98),
        )
        st.plotly_chart(fig_smile, use_container_width=True)

        st.divider()

        # ── IV Lookup Tool ─────────────────────────────────────────────────────
        st.subheader("Implied Vol Lookup Tool")
        st.caption("Enter any strike — see what the market implies vs what your flat model assumes.")

        col_in, col_out = st.columns([1, 2])
        with col_in:
            k_input = st.number_input("Strike K_input", min_value=100.0,
                                      max_value=float(x_max) * 1.5,
                                      value=float(smile_spot), step=1.0)

        # Interpolate market IV at k_input
        combined_smile = pd.concat([calls_smile, puts_smile]).sort_values("strike")
        if len(combined_smile) >= 2:
            market_iv_pct = float(np.interp(
                k_input,
                combined_smile["strike"].values,
                combined_smile["implied_vol"].values * 100,
            ))
        else:
            market_iv_pct = float("nan")

        your_iv_pct   = sigma_pct
        gap           = market_iv_pct - your_iv_pct

        with col_out:
            lc, rc = st.columns(2)
            lc.metric("Market implied vol",   f"{market_iv_pct:.2f}%")
            rc.metric("Your flat σ",          f"{your_iv_pct:.2f}%",
                      delta=f"{gap:+.2f}pp gap")

        if abs(gap) > 5:
            moneyness = k_input / smile_spot
            direction = "OTM" if (moneyness > 1.05 or moneyness < 0.95) else "near ATM"
            st.warning(
                f"**Gap of {gap:+.2f}pp** — this is a {direction} strike. "
                f"The market prices this option at **{market_iv_pct:.1f}% implied vol** "
                f"but your flat model assumes **{your_iv_pct:.1f}%**. "
                f"Your model {'underprices' if gap > 0 else 'overprices'} this option by roughly "
                f"₹{abs(gap) * 0.01 * smile_spot * np.sqrt(smile_T):.2f} "
                f"(Vega × vol gap)."
            )
        else:
            st.success(f"Gap of {gap:+.2f}pp — your flat model is a reasonable fit at this strike.")

        st.divider()

        # ── Model limitations ──────────────────────────────────────────────────
        st.subheader("Model Limitations and Next Steps")
        st.markdown("""
**Why flat vol works at-the-money but breaks down at extremes:**

Standard Black-Scholes and GBM assume volatility is constant — a single σ that describes
all possible stock price moves equally. This works reasonably well *at-the-money* (near the
current price), where most trading occurs and where our calibrated σ = 21.07% is accurate.

**Why the smile exists:**

At extreme strikes (especially deep out-of-the-money puts), the market charges a premium.
This is the *crash risk premium* — investors pay extra to insure against tail events like
market crashes that GBM says are nearly impossible (it uses a normal distribution for returns)
but which actually happen more often than expected (*fat tails* / *leptokurtosis*).

The 2008 financial crisis, 2020 COVID crash, and 1987 Black Monday are all events that
GBM would assign near-zero probability but which actually occurred.

**What our results show:**

- ATM implied vol ≈ **22.5%** — close to our calibrated 21.07%, confirming the model works ATM
- OTM put implied vol ≈ **91.2%** — the market charges **4× higher vol** for crash protection
- Smile spread ≈ **+68.7pp** — this is the quantified cost of our model's blind spot

**The natural extension — Heston Stochastic Volatility Model:**

The Heston model (1993) solves this by making volatility itself a random process:
```
dS = μS dt + √v · S dW₁
dv = κ(θ − v) dt + ξ√v dW₂        (v = variance, mean-reverts to θ)
```
This produces a natural volatility smile and is the industry standard at derivatives desks
at JP Morgan, Goldman Sachs, and Citadel. It would be the direct next step for this project.
        """)

# ══════════════════════════════════════════════════════════════════════════════
# AUTO-REFRESH COUNTDOWN (only runs when toggle is ON)
# ══════════════════════════════════════════════════════════════════════════════
if auto_refresh:
    elapsed = time.time() - st.session_state.last_auto_refresh
    if elapsed >= refresh_interval:
        fetch_live_price.clear()
        fetch_intraday.clear()
        result = fetch_live_price()
        st.session_state.live_price_data = result
        if result["success"]:
            st.session_state.s0_override = int(result["price"])
        st.session_state.last_auto_refresh = time.time()
        st.rerun()
    else:
        time.sleep(1)
        st.rerun()
