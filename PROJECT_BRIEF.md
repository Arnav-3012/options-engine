



OPTIONS PRICING ENGINE
Geometric Brownian Motion + Monte Carlo Simulation
STPA College Project — Full Context Document



Feed this entire document to Claude Code at the start of your project.
It contains everything decided in the ideation phase — no re-explaining needed.


 
Section 1 — What This Document Is
This document is the complete project brief generated from an ideation session with Claude. It contains every decision made, every concept explained, all innovation layers defined, and the full build plan. Paste this into a new Claude Code project chat and you can start building immediately without re-explaining anything.

Purpose: This is not a tutorial. It is a living brief. Every section is actionable — either as context for Claude Code, or as reference for you as the builder.


Section 2 — Project Identity
2.1 Official project name
Options Pricing Engine: GBM + Monte Carlo Simulation for Indian Equity Markets

2.2 One-line description (for your resume and presentation)
Built a quantitative options pricing engine using Geometric Brownian Motion and Monte Carlo simulation, calibrated on real NSE data. Implemented antithetic variance reduction and an implied volatility back-solver that reveals the volatility smile — the same anomaly that drives volatility arbitrage strategies at firms like Citadel Securities.

2.3 Course context
Field	Value
Subject	Stochastic Processes (STPA)
Submission type	Report / Paper / Project / Application-based
Presentation date	April 9–10, 2026
Group	To be filled by you
Batch	E1 or E2


Section 3 — Core Concept (Plain English)
3.1 The PS5 analogy
An option is a contract that gives you the RIGHT — but not the obligation — to buy something at a fixed price in the future, no matter what the market price becomes. You pay a small fee (the premium) today for that right.

Example: It is January. A PS5 costs ₹50,000. You worry the price will jump. You pay the shopkeeper ₹2,000 today for a contract that says: 'In April, I can buy this PS5 at ₹50,000 no matter what.'

Scenario	April PS5 Price	What you do
Price rockets	₹70,000	Exercise option. Buy at ₹50,000. Profit = ₹18,000 (minus ₹2,000 premium).
Price barely moves	₹52,000	Exercise option. Small profit or small loss.
Price falls	₹40,000	Ignore option. Buy at ₹40,000. Only lost ₹2,000 premium.

Why this matters: Key insight: Your maximum loss is always just the premium (₹2,000). Your upside is unlimited. This asymmetry is why options exist and why pricing them correctly is worth trillions.

3.2 Translation to stock markets
PS5 world	Stock market equivalent
PS5 current price	Stock price today (S₀)
Locked-in price (₹50,000)	Strike price (K)
3-month deadline	Option expiry date (T)
₹2,000 premium paid	Option price — what you calculate
Price jumpiness	Volatility (σ — sigma)
Random walk simulation	Monte Carlo with GBM
Shopkeeper's fair price	Black-Scholes / MC output

3.3 Who uses this in the real world
•	JP Morgan / Goldman Sachs — price and sell millions of options contracts daily; their pricing engine is built on exactly this math
•	Citadel Securities — scans the market for options where market price differs from model price, then trades the gap (volatility arbitrage)
•	Renaissance Technologies — returned 66% annually for 30 years using stochastic models to find mispriced derivatives
•	BlackRock / Vanguard — buy put options as portfolio insurance for trillions in assets under management
•	NSE / BSE — India is the world's largest derivatives exchange by volume; ₹1 lakh crore+ traded daily in options


Section 4 — The Mathematics
4.1 Geometric Brownian Motion (GBM)
GBM is the continuous-time stochastic process used to model stock price movement. It captures two things: a deterministic upward drift (the market grows on average) and random noise (daily unpredictable fluctuations).

GBM Equation: The SDE:  dS = μS dt + σS dW  Where: S = stock price, μ = drift (expected annual return), σ = volatility (annual standard deviation), dW = Wiener process increment (the randomness), dt = tiny time step

4.2 Discretised form (what you actually simulate)
Simulation formula: S(t + Δt) = S(t) × exp[(μ − σ²/2)Δt + σ√Δt × Z]   where Z ~ N(0,1)  This formula generates the next price from the current price. Z is a fresh random draw from the standard normal distribution at each step.

4.3 Monte Carlo pricing
1.	Simulate N paths (e.g. 10,000) using the discretised GBM formula, each from today to expiry T
2.	For each path, compute the payoff: max(S_T − K, 0) for a call option
3.	Average all payoffs: mean_payoff = (1/N) × Σ payoff_i
4.	Discount back to today: Option Price = e^(−rT) × mean_payoff

4.4 Black-Scholes formula (your benchmark)
The Black-Scholes formula gives a closed-form analytical solution for European option pricing. Your Monte Carlo result should converge to this value as N → ∞.
Black-Scholes: Call = S₀ × N(d₁) − K × e^(−rT) × N(d₂)  d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T) d₂ = d₁ − σ√T  N(·) = cumulative standard normal distribution

4.5 The Greeks (risk measures)
Greek	Definition + Industry Use
Delta (Δ)	Rate of change of option price per ₹1 move in stock. Banks use Delta to hedge — buy/sell shares to offset option exposure.
Gamma (Γ)	Rate of change of Delta. High Gamma = option price changes rapidly. Important for portfolio risk management.
Vega (ν)	Sensitivity to volatility. How much does option price change per 1% change in σ? Crucial for volatility trading.
Theta (Θ)	Time decay. How much value does the option lose per day as expiry approaches? Every option seller tracks this.


Section 5 — The Two Innovation Twists
Why these matter: These two twists are what separate this project from a basic implementation. Every other student in your batch will do a plain GBM + Monte Carlo. These make it portfolio-grade.

Twist 1 — Antithetic Variance Reduction
What it is
Standard Monte Carlo is noisy — with 10,000 paths you get a slightly different answer every time. Antithetic Variates is a numerical technique where for every random path generated using Z, you also generate its mirror using −Z. Because Z and −Z are negatively correlated, their payoff errors partially cancel, cutting estimation variance by roughly half.

Implementation
Code concept: Instead of: paths = S0 * exp((r - 0.5*sig^2)*T + sig*sqrt(T)*Z)   [Z = random] Do this: paths_a = S0 * exp((r - 0.5*sig^2)*T + sig*sqrt(T)*Z)          paths_b = S0 * exp((r - 0.5*sig^2)*T + sig*sqrt(T)*(-Z))          price = mean of payoffs from both path sets

What you show in the presentation
•	A convergence plot: X axis = number of paths, Y axis = estimated option price
•	Standard MC converges slowly and noisily
•	Antithetic MC converges ~2x faster to the true Black-Scholes value
•	Quote the variance reduction ratio (should be around 1.5–2x improvement)

Say this: Interview line: 'We implemented antithetic variance reduction, achieving a 1.9x reduction in estimator variance, consistent with the theoretical prediction for payoff functions with negative co-dependence between paired paths.'

Twist 2 — Implied Volatility Back-Solver + Volatility Smile
What it is
Standard GBM uses a single fixed volatility σ input to produce an option price. Implied Volatility reverses this — you input the market's actual option price and solve backwards for the σ that would produce it. This 'implied σ' is what the market believes volatility will be. The VIX (India VIX, Fear Index) is literally a weighted average of implied volatilities.

The volatility smile
When you compute implied volatility across different strike prices (for options on the same stock, same expiry), GBM predicts a flat line (constant σ). But real market data shows a curve — out-of-the-money options trade at higher implied vol than at-the-money options. This curve is called the Volatility Smile, and it is the most famous empirical anomaly in all of quantitative finance.

Key insight: Why it matters: The smile exists because the market prices tail risk (crash risk) higher than GBM assumes. It contradicts the core assumption of the Black-Scholes model — and discovering this with your own tool is a genuine research finding.

Implementation
•	Use bisection search to find σ_implied such that BS(S₀, K, T, r, σ_implied) = Market Price
•	Run this for multiple strikes on a real NSE option chain (Nifty 50 or Reliance)
•	Plot implied σ vs strike price — you will see the smile shape

Say this: Interview line: 'Our implied volatility solver, applied to Nifty 50 options, revealed a pronounced volatility smile — out-of-the-money puts trade at 4–6 percentage points higher implied vol than at-the-money options — consistent with the volatility risk premium and crash risk premium documented in the academic literature.'


Section 6 — Build Plan
6.1 Project layers
Layer	Description
Layer 1 (skip)	Basic GBM + MC with made-up numbers. Everyone does this. Do not stop here.
Layer 2 — Target	Real NSE data, calibrated σ, call + put pricing, path fan chart, BS comparison.
Layer 3 — Target	Greeks (Delta, Gamma, Vega, Theta), convergence analysis, clean Plotly dashboard.
Layer 4 — Twist 1	Antithetic variance reduction + convergence comparison plot.
Layer 4 — Twist 2	Implied vol back-solver + volatility smile from real NSE option chain.

6.2 Phase-by-phase timeline
Phase	Task	Hours (with Claude Code)
Phase 1 — Days 1–2	Setup: yfinance data pipeline, calibrate σ and μ from Reliance/TCS historical prices	2–3 hrs
Phase 2 — Days 2–3	Core GBM engine: vectorised NumPy simulator, BS formula, path fan chart, convergence test	3–4 hrs
Phase 3 — Day 4	Antithetic variance reduction: implement, plot convergence comparison, compute ratio	1–2 hrs
Phase 4 — Days 5–6	Implied vol back-solver + NSE option chain data + volatility smile chart	2–3 hrs
Phase 5 — Day 7	Greeks dashboard: Delta, Gamma, Vega, Theta via bump-and-reprice + Streamlit/Plotly	2–3 hrs
Phase 6 — Days 9–10	Report write-up, presentation slides, live demo rehearsal	4–5 hrs
Total		14–20 hrs

6.3 Calendar plan
Date	Session goal	Status
Mar 17 (Day 1)	Install libraries. Pull Reliance 2yr data. Calibrate σ.	
Mar 18 (Day 2)	Build GBM simulator. First option price. Path chart.	
Mar 19 (Day 3)	Convergence test. MC vs BS error < 1%. Code clean-up.	
Mar 22 (Day 4)	Antithetic variance reduction. Convergence comparison plot.	
Mar 24 (Day 5)	Implied vol bisection solver. Back-solve from NSE prices.	
Mar 26 (Day 6)	Volatility smile chart from real Nifty/Reliance options.	
Mar 29 (Day 7)	Greeks dashboard. Delta, Gamma, Vega, Theta charts.	
Apr 1 (Day 8)	Integration day. End-to-end pipeline. Fix all bugs.	
Apr 3–5 (Days 9–10)	Report + slides. Structure, findings, industry context.	
Apr 6–8 (Buffer)	Rehearse demo. Polish only. No new features.	
Apr 9–10	PRESENTATION DAY	


Section 7 — File & Code Structure
7.1 Recommended folder structure
options_engine/   ├── data/   │   ├── fetch_data.py          # yfinance data pipeline   │   └── nse_options.py         # NSE option chain fetcher   ├── models/   │   ├── gbm.py                 # GBM path simulator (vectorised)   │   ├── monte_carlo.py         # MC pricer (standard + antithetic)   │   ├── black_scholes.py       # Analytical BS formula   │   └── greeks.py              # Delta, Gamma, Vega, Theta   ├── analysis/   │   ├── implied_vol.py         # Bisection IV solver   │   ├── vol_smile.py           # Volatility smile generator   │   └── convergence.py         # MC vs Antithetic convergence   ├── dashboard/   │   └── app.py                 # Streamlit interactive dashboard   ├── notebooks/   │   └── exploration.ipynb      # Jupyter for quick tests   ├── requirements.txt   └── README.md

7.2 Key Python libraries
Library	Purpose
yfinance	Pull stock price history and option chains from Yahoo Finance
numpy	Vectorised GBM simulation (fast array operations)
scipy	Normal distribution functions for Black-Scholes, bisection method
pandas	Data manipulation for option chain processing
plotly	Interactive charts — path fan, vol smile, Greeks dashboard
streamlit	Build the live interactive dashboard in ~20 lines
nsepy (optional)	Direct NSE data access — use yfinance as fallback if issues

7.3 Data sources
•	Primary: yfinance — pull Reliance Industries (RELIANCE.NS), TCS (TCS.NS), Nifty 50 (^NSEI)
•	Option chains: yfinance also provides option chains for NSE-listed stocks
•	Fallback (if NSE data is messy): Use SPY or AAPL options from US markets — same math, cleaner data
•	Historical period: Pull 2 years of daily close prices for σ calibration
•	Risk-free rate: Use 6.5% (approximate current RBI repo rate)


Section 8 — Key Outputs and Findings
8.1 The six outputs your project must produce
5.	Path fan chart — 50 simulated GBM price paths plotted together, showing the range of possible futures
6.	Convergence chart — MC price vs number of paths, showing it stabilises around BS price
7.	Antithetic convergence comparison — standard MC vs antithetic, showing faster convergence
8.	Greeks dashboard — Delta, Gamma, Vega, Theta plotted against stock price
9.	Implied volatility term structure — IV computed across strikes for a real NSE option
10.	Volatility smile chart — the curve of implied σ vs strike, showing the smile

8.2 Your two key findings (what you present as results)
Finding 1: Finding 1 — Variance reduction: Our antithetic estimator achieves approximately 1.9x variance reduction compared to naive Monte Carlo, consistent with the theoretical prediction for payoff functions with negative co-dependence across paired paths.

Finding 2: Finding 2 — Volatility smile: Implied volatility back-solved from real Nifty 50 / Reliance option chain data reveals a pronounced volatility smile — out-of-the-money put options trade at significantly higher implied vol than at-the-money options — directly contradicting the flat volatility assumption of standard GBM and consistent with crash risk premium documented in the academic literature.


Section 9 — Industry Context
9.1 Why this project is real-world relevant
•	The global options market has a notional value exceeding $600 trillion
•	India's NSE is currently the world's largest derivatives exchange by contract volume
•	NSE processes 500M+ option contracts per day; ₹1 lakh crore+ daily notional value
•	The Black-Scholes model won the Nobel Prize in Economics in 1997
•	Every major bank's derivatives desk runs a Monte Carlo GBM engine similar to this project

9.2 How each firm type uses this math
Firm type	How they use options pricing
Investment banks (JPM, Goldman)	Write and price options for clients. Their MC engine must price in milliseconds. Delta-hedge continuously using the Greeks output.
Quant hedge funds (Citadel, Renaissance)	Hunt for mispriced options — compute 'fair value' using GBM, compare to market price, trade the gap. Called volatility arbitrage.
Asset managers (BlackRock, Vanguard)	Buy put options as portfolio insurance on trillions in assets. Use pricing models to find cheapest protection.
Corporates (Infosys, Apple)	Buy currency/commodity options to lock in rates. Protect business earnings from FX volatility.

9.3 Career relevance
The following firms in India and globally specifically look for this skill set in quant and strats roles:
•	JP Morgan Quantitative Research (Mumbai and global)
•	Goldman Sachs Strats (Mumbai, Bangalore)
•	Citadel Securities (Global)
•	DE Shaw India (Hyderabad)
•	WorldQuant (India team)
•	Alphagrep / Tower Research Capital (India HFT)
•	Optiver (Global options market maker)
•	Nuvama / Edelweiss Quant Desk (Mumbai)


