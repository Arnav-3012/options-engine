"""
analysis/convergence.py
-----------------------
Convergence analysis: Standard MC vs Antithetic Variance Reduction.

Runs both pricers across increasing path counts and records how quickly
each estimate converges to the true Black-Scholes price. Produces a
Plotly chart saved to outputs/convergence_comparison.html.

Key metric: variance reduction ratio = Var(standard MC) / Var(antithetic MC)
Target: ~1.5x to 2x improvement.
"""

import os
import numpy as np
import plotly.graph_objects as go
from models.black_scholes import black_scholes
from models.monte_carlo import mc_price, mc_antithetic

# ── Inputs ────────────────────────────────────────────────────────────────────
S0          = 1380.70
K           = 1400.0
T           = 30 / 365
r           = 0.065
SIGMA       = 0.2107
OPTION_TYPE = "call"
PATH_COUNTS = [100, 250, 500, 1_000, 2_500, 5_000, 10_000]
N_REPEATS   = 30       # repeats per N to estimate variance reliably
# ─────────────────────────────────────────────────────────────────────────────


def run_convergence(n_repeats: int = N_REPEATS) -> dict:
    """
    Run standard MC and antithetic MC across PATH_COUNTS with multiple repeats.

    For each N in PATH_COUNTS, runs n_repeats independent simulations of both
    methods and records all price estimates. This lets us measure the variance
    of each estimator at every path count.

    Parameters
    ----------
    n_repeats : int — number of independent runs per path count

    Returns
    -------
    dict with keys:
        path_counts  (list)       — path counts used
        std_prices   (np.ndarray) — shape (len(PATH_COUNTS), n_repeats), standard MC prices
        anti_prices  (np.ndarray) — shape (len(PATH_COUNTS), n_repeats), antithetic prices
        bs_price     (float)      — analytical Black-Scholes price (constant)
    """
    bs_price = black_scholes(S0, K, T, r, SIGMA, OPTION_TYPE)["price"]

    std_prices  = np.zeros((len(PATH_COUNTS), n_repeats))
    anti_prices = np.zeros((len(PATH_COUNTS), n_repeats))

    for i, n in enumerate(PATH_COUNTS):
        for j in range(n_repeats):
            std_prices[i, j]  = mc_price(S0, K, T, r, SIGMA, n, OPTION_TYPE)["price"]
            anti_prices[i, j] = mc_antithetic(S0, K, T, r, SIGMA, n, OPTION_TYPE)["price"]

    return {
        "path_counts": PATH_COUNTS,
        "std_prices":  std_prices,
        "anti_prices": anti_prices,
        "bs_price":    bs_price,
    }


def variance_reduction_ratio(results: dict) -> float:
    """
    Compute variance reduction ratio at the largest N (10,000 paths).

    Ratio = Var(standard MC) / Var(antithetic MC)
    A ratio of 1.9 means antithetic achieves the same accuracy as standard
    MC with only ~53% of the paths.

    Parameters
    ----------
    results : dict — output of run_convergence()

    Returns
    -------
    float — variance reduction ratio
    """
    var_std  = np.var(results["std_prices"][-1],  ddof=1)
    var_anti = np.var(results["anti_prices"][-1], ddof=1)
    return var_std / var_anti


def plot_convergence(results: dict, output_path: str = "outputs/convergence_comparison.html") -> None:
    """
    Produce a Plotly convergence comparison chart and save as HTML.

    Three lines:
      - Standard MC mean price across repeats (noisy)
      - Antithetic MC mean price across repeats (smoother)
      - True Black-Scholes price (flat horizontal reference)

    Shaded bands show ±1 std dev across repeats, visualising estimator noise.

    Parameters
    ----------
    results     : dict — output of run_convergence()
    output_path : str  — file path for the HTML output
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ns         = results["path_counts"]
    bs_price   = results["bs_price"]
    std_mean   = results["std_prices"].mean(axis=1)
    std_std    = results["std_prices"].std(axis=1, ddof=1)
    anti_mean  = results["anti_prices"].mean(axis=1)
    anti_std   = results["anti_prices"].std(axis=1, ddof=1)

    fig = go.Figure()

    # ── Standard MC band ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=ns + ns[::-1],
        y=list(std_mean + std_std) + list((std_mean - std_std)[::-1]),
        fill="toself", fillcolor="rgba(99,110,250,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=ns, y=std_mean,
        mode="lines+markers",
        name="Standard MC",
        line=dict(color="#636EFA", width=2),
        marker=dict(size=6),
    ))

    # ── Antithetic MC band ────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=ns + ns[::-1],
        y=list(anti_mean + anti_std) + list((anti_mean - anti_std)[::-1]),
        fill="toself", fillcolor="rgba(0,204,150,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=ns, y=anti_mean,
        mode="lines+markers",
        name="Antithetic MC",
        line=dict(color="#00CC96", width=2),
        marker=dict(size=6),
    ))

    # ── Black-Scholes reference ───────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=ns, y=[bs_price] * len(ns),
        mode="lines",
        name=f"Black-Scholes (₹{bs_price:.2f})",
        line=dict(color="#EF553B", width=2, dash="dash"),
    ))

    ratio = variance_reduction_ratio(results)

    fig.update_layout(
        title=dict(
            text=(
                "MC vs Antithetic Convergence — RELIANCE.NS Call Option"
                f"<br><sup>Variance reduction ratio: {ratio:.2f}x  |  "
                f"S0=₹{S0}, K=₹{K}, T=30d, σ={SIGMA}, r={r}</sup>"
            ),
            x=0.5,
        ),
        xaxis=dict(title="Number of Paths (N)", type="log"),
        yaxis=dict(title="Estimated Call Price (₹)"),
        legend=dict(x=0.75, y=0.05),
        template="plotly_dark",
        width=900, height=550,
    )

    fig.write_html(output_path)
    print(f"Chart saved → {output_path}")


if __name__ == "__main__":
    print("Running convergence analysis …")
    print(f"  {N_REPEATS} repeats × {len(PATH_COUNTS)} path counts = "
          f"{N_REPEATS * len(PATH_COUNTS) * 2} total simulations\n")

    results = run_convergence()
    ratio   = variance_reduction_ratio(results)

    bs = results["bs_price"]
    print(f"  Black-Scholes price : ₹{bs:.4f}")
    print()
    print(f"  {'N':>7}  {'Std MC mean':>12}  {'Anti MC mean':>13}  {'Std σ':>8}  {'Anti σ':>8}")
    print("  " + "-" * 58)
    for i, n in enumerate(PATH_COUNTS):
        print(
            f"  {n:>7,}  "
            f"₹{results['std_prices'][i].mean():>10.4f}  "
            f"₹{results['anti_prices'][i].mean():>11.4f}  "
            f"{results['std_prices'][i].std(ddof=1):>8.4f}  "
            f"{results['anti_prices'][i].std(ddof=1):>8.4f}"
        )

    print()
    print(f"  Variance reduction ratio (N=10,000): {ratio:.3f}x")
    target = "PASS" if ratio >= 1.5 else "FAIL (below 1.5x)"
    print(f"  Target ≥1.5x: {target}")

    plot_convergence(results)
