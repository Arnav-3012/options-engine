# Options Pricing Engine — CLAUDE.md

## Project goal
Build an options pricing engine using GBM + Monte Carlo simulation for Indian equity markets (STPA college project). Presentation: April 9–10, 2026.

## Two key innovations
1. Antithetic Variance Reduction — mirror paths using −Z, target ~1.9x variance reduction
2. Implied Volatility back-solver (bisection) + Volatility Smile from real NSE option chain data

## Tech stack
Python, NumPy (vectorised — no for-loops), SciPy, yfinance, Plotly, Streamlit

## Constants
- Risk-free rate: r = 0.065
- Primary stock: RELIANCE.NS
- Benchmark index: ^NSEI (Nifty 50)
- Simulation paths: N = 10,000
- Historical period: 2 years daily closes

## File structure
data/fetch_data.py        — yfinance pipeline, calibrate sigma/mu
data/nse_options.py       — NSE option chain fetcher
models/gbm.py             — GBM path simulator (vectorised NumPy)
models/monte_carlo.py     — MC pricer (standard + antithetic)
models/black_scholes.py   — Analytical Black-Scholes formula
models/greeks.py          — Delta, Gamma, Vega, Theta
analysis/implied_vol.py   — Bisection IV solver
analysis/vol_smile.py     — Volatility smile generator
analysis/convergence.py   — MC vs antithetic convergence comparison
dashboard/app.py          — Streamlit interactive dashboard

## Coding rules
- NumPy vectorised operations only — no Python for-loops in simulations
- Use Plotly for all charts (not matplotlib)
- All functions must have docstrings
- Save chart outputs as HTML files in an outputs/ folder

## Current phase
Phase 1 — Data pipeline: fetch RELIANCE.NS data, calibrate sigma and mu

## Reference document
Full project brief with all math, phase prompts, innovations, and industry context is in PROJECT_BRIEF.md — read it at the start of each session alongside this file.