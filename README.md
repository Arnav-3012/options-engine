# Options Pricing Engine
### GBM + Monte Carlo Simulation for Indian Equity Markets

**STPA College Project — Stochastic Processes**
**Presentation: April 9–10, 2026**

---

## What This Project Does

A quantitative options pricing engine built from scratch in Python. Simulates stock price paths using Geometric Brownian Motion, prices European call and put options via Monte Carlo simulation, benchmarks against the analytical Black-Scholes formula, and reveals the Volatility Smile from real option chain data.

**One-line description for your resume:**
> Built a quantitative options pricing engine using Geometric Brownian Motion and Monte Carlo simulation, calibrated on real NSE data. Implemented antithetic variance reduction and an implied volatility back-solver that reveals the volatility smile — the same anomaly that drives volatility arbitrage strategies at firms like Citadel Securities.

---

## Two Key Innovations

### 1. Antithetic Variance Reduction
Standard Monte Carlo is noisy — every run gives a slightly different answer. For every random draw Z ~ N(0,1), we also simulate its mirror using −Z. Because Z and −Z are negatively correlated, their payoff errors partially cancel.

**Result:** ~2.19x variance reduction. Antithetic MC with 5,000 paths matches the accuracy of standard MC with 10,000 paths.

### 2. Implied Volatility Back-Solver + Volatility Smile
Standard Black-Scholes assumes a single flat volatility σ. Implied volatility reverses this — given a market price, we bisection-search for the σ that produces it. Running this across strikes reveals the **Volatility Smile**: out-of-the-money options trade at significantly higher implied vol than at-the-money options.

**Result:** ATM implied vol ≈ 22.5%, OTM put implied vol ≈ 91.2%, smile spread ≈ +68.7pp — direct evidence of the crash risk premium that GBM ignores.

---

## Results

| Metric | Value |
|---|---|
| Primary stock | RELIANCE.NS |
| Calibration period | 2 years daily closes |
| Last close S0 | ₹1,380.70 |
| Annualised σ | 21.07% |
| Annualised μ | −1.76% |
| Risk-free rate r | 6.5% |
| BS call price (K=1400, T=30d) | ₹27.82 |
| MC call price | ₹27.82 |
| MC vs BS error | 0.016% |
| Antithetic variance reduction | 2.19x |
| ATM implied vol (SPY) | 22.5% |
| OTM put implied vol | 91.2% |
| Volatility smile spread | +68.7pp |

---

## Project Structure

```
options_engine/
├── data/
│   ├── fetch_data.py        # yfinance pipeline — pull RELIANCE.NS, calibrate σ and μ
│   └── nse_options.py       # Option chain fetcher (RELIANCE.NS → SPY fallback)
├── models/
│   ├── gbm.py               # Vectorised GBM path simulator (NumPy, no loops)
│   ├── monte_carlo.py       # Standard MC + antithetic variance reduction
│   ├── black_scholes.py     # Analytical Black-Scholes formula
│   └── greeks.py            # Delta, Gamma, Vega, Theta via bump-and-reprice
├── analysis/
│   ├── implied_vol.py       # Bisection IV back-solver
│   ├── vol_smile.py         # Volatility smile generator + Plotly chart
│   └── convergence.py       # MC vs antithetic convergence comparison
├── dashboard/
│   └── app.py               # Full 4-tab Streamlit interactive dashboard
├── outputs/
│   ├── convergence_comparison.html
│   └── vol_smile.html
├── notebooks/
│   └── exploration.ipynb
├── requirements.txt
└── README.md
```

---

## The Mathematics

**GBM SDE:**
```
dS = μS dt + σS dW
```

**Discretised simulation formula** (what we actually compute):
```
S(t+Δt) = S(t) × exp[(r − σ²/2)Δt + σ√Δt × Z]    where Z ~ N(0,1)
```

**Monte Carlo pricing:**
```
Price = e^(−rT) × mean(max(S_T − K, 0))    over N paths
```

**Black-Scholes analytical formula:**
```
Call = S₀·N(d₁) − K·e^(−rT)·N(d₂)
d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ − σ√T
```

**Antithetic variance reduction:**
```python
paths_a = S0 * exp((r - 0.5*σ²)*T + σ*√T * Z)     # standard
paths_b = S0 * exp((r - 0.5*σ²)*T + σ*√T * (-Z))  # antithetic mirror
price   = e^(-rT) * mean((payoff(paths_a) + payoff(paths_b)) / 2)
```

---

## Tech Stack

| Library | Purpose |
|---|---|
| `numpy` | Vectorised GBM simulation — no Python for-loops |
| `scipy` | Normal CDF for Black-Scholes, bisection IV solver |
| `yfinance` | RELIANCE.NS historical data + option chains |
| `pandas` | Option chain processing |
| `plotly` | All interactive charts |
| `streamlit` | Live 4-tab dashboard |

---

## Setup

```bash
# Clone the repo
git clone https://github.com/Arnav-3012/options-engine.git
cd options-engine

# Install dependencies
pip install -r requirements.txt
```

---

## How to Run

```bash
# Phase 1 — Calibrate RELIANCE.NS parameters
python data/fetch_data.py

# Phase 2 — MC vs Black-Scholes comparison
python -m models.monte_carlo

# Phase 3 — Greeks verification
python -m models.greeks

# Phase 4 — Convergence analysis + chart
python -m analysis.convergence

# Phase 5 — Volatility smile + chart
python -m analysis.vol_smile

# Full dashboard (all phases, interactive)
streamlit run dashboard/app.py
```

---

## Dashboard — 4 Tabs

**Tab 1 — Pricing & Greeks**
Summary table of all project results. Live call/put prices and Greeks (Delta, Gamma, Vega, Theta) as metric cards. Four Greek-vs-spot charts. Payoff diagram with breakeven markers.

**Tab 2 — Simulation**
GBM path fan chart: 200 sample paths + 5th–95th percentile band. Terminal price histogram with log-normal PDF overlay, ITM shading, and P(ITM) annotation.

**Tab 3 — Convergence**
Standard MC vs antithetic MC convergence chart. Variance reduction ratio metric card. Full convergence table. Plain English explanation.

**Tab 4 — Volatility Smile**
Implied vol vs strike chart with live flat-σ reference line. IV lookup tool: enter any strike, see market IV vs model IV with gap warning. Model limitations and Heston model explanation.

---

## Key Findings

**Finding 1 — Variance Reduction:**
Our antithetic estimator achieves **2.19x variance reduction** compared to naive Monte Carlo, consistent with the theoretical prediction for payoff functions with negative co-dependence across paired paths.

**Finding 2 — Volatility Smile:**
Implied volatility back-solved from real option chain data reveals a pronounced volatility smile — out-of-the-money put options trade at significantly higher implied vol (+68.7pp) than at-the-money options — directly contradicting the flat volatility assumption of standard GBM and consistent with the crash risk premium documented in academic literature.

---

## Industry Context

- The global options market has a notional value exceeding **$600 trillion**
- India's NSE is the **world's largest derivatives exchange** by contract volume
- The Black-Scholes model won the **Nobel Prize in Economics in 1997**
- Every major bank's derivatives desk (JP Morgan, Goldman Sachs, Citadel) runs a Monte Carlo GBM engine identical in structure to this project

---

