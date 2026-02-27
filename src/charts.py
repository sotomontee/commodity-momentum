"""
charts.py
---------
All visualisations for the backtest results.
Produces publication-quality charts suitable for a GitHub portfolio.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import os

# ── Style ─────────────────────────────────────────────────────────────────────
STRATEGY_COLOR  = "#2563EB"   # blue
BENCHMARK_COLOR = "#F59E0B"   # amber
DRAWDOWN_COLOR  = "#EF4444"   # red
NEUTRAL_COLOR   = "#6B7280"   # grey

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F9FAFB",
    "axes.grid":         True,
    "grid.color":        "#E5E7EB",
    "grid.linewidth":    0.7,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "sans-serif",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
})

OUTPUT_DIR = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_cumulative_returns(
    strategy: pd.Series,
    benchmark: pd.Series,
    save: bool = True,
) -> plt.Figure:
    """Cumulative return comparison: strategy vs equal-weight benchmark."""
    fig, ax = plt.subplots(figsize=(12, 5))

    cum_strat = (1 + strategy).cumprod()
    cum_bench = (1 + benchmark.reindex(strategy.index)).cumprod()

    ax.plot(cum_strat.index, cum_strat, color=STRATEGY_COLOR,  lw=2,   label="Momentum Strategy")
    ax.plot(cum_bench.index, cum_bench, color=BENCHMARK_COLOR, lw=1.5, label="Equal-Weight Benchmark", linestyle="--")
    ax.axhline(1, color=NEUTRAL_COLOR, lw=0.8, linestyle=":")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax.set_title("Cumulative Returns: Commodity Momentum vs Benchmark", fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Growth of $1")
    ax.legend(framealpha=0.9)

    plt.tight_layout()
    if save:
        fig.savefig(f"{OUTPUT_DIR}/01_cumulative_returns.png", dpi=150)
    return fig


def plot_drawdowns(
    strategy: pd.Series,
    benchmark: pd.Series,
    save: bool = True,
) -> plt.Figure:
    """Drawdown chart for strategy and benchmark."""
    from src.metrics import drawdown_series

    fig, ax = plt.subplots(figsize=(12, 4))

    dd_strat = drawdown_series(strategy)
    dd_bench = drawdown_series(benchmark.reindex(strategy.index).dropna())

    ax.fill_between(dd_strat.index, dd_strat, 0, color=DRAWDOWN_COLOR,  alpha=0.4, label="Strategy Drawdown")
    ax.fill_between(dd_bench.index, dd_bench, 0, color=BENCHMARK_COLOR, alpha=0.25, label="Benchmark Drawdown")
    ax.plot(dd_strat.index, dd_strat, color=DRAWDOWN_COLOR, lw=1.2)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_title("Drawdown Analysis", fontweight="bold", pad=12)
    ax.set_ylabel("Drawdown")
    ax.legend(framealpha=0.9)

    plt.tight_layout()
    if save:
        fig.savefig(f"{OUTPUT_DIR}/02_drawdowns.png", dpi=150)
    return fig


def plot_monthly_returns_heatmap(
    returns: pd.Series,
    save: bool = True,
) -> plt.Figure:
    """Classic monthly returns heatmap (year × month)."""
    df = returns.copy()
    df.index = pd.to_datetime(df.index)
    monthly = df.to_frame("ret")
    monthly["year"]  = monthly.index.year
    monthly["month"] = monthly.index.month

    pivot = monthly.pivot(index="year", columns="month", values="ret")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.5 + 1)))

    vmax = pivot.abs().max().max()
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    # Labels
    ax.set_xticks(range(12))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index)

    for i in range(len(pivot)):
        for j in range(12):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1%}", ha="center", va="center",
                        fontsize=8, color="black")

    plt.colorbar(im, ax=ax, format=mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_title("Monthly Returns Heatmap — Commodity Momentum Strategy", fontweight="bold", pad=12)

    plt.tight_layout()
    if save:
        fig.savefig(f"{OUTPUT_DIR}/03_monthly_heatmap.png", dpi=150)
    return fig


def plot_rolling_sharpe(
    returns: pd.Series,
    window: int = 24,
    save: bool = True,
) -> plt.Figure:
    """Rolling 24-month Sharpe ratio."""
    rolling_sr = (
        returns.rolling(window).mean() /
        returns.rolling(window).std()
    ) * np.sqrt(12)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(rolling_sr.index, rolling_sr, color=STRATEGY_COLOR, lw=1.8)
    ax.axhline(0, color=NEUTRAL_COLOR, lw=0.8, linestyle="--")
    ax.fill_between(rolling_sr.index, rolling_sr, 0,
                    where=rolling_sr > 0, color=STRATEGY_COLOR, alpha=0.15)
    ax.fill_between(rolling_sr.index, rolling_sr, 0,
                    where=rolling_sr < 0, color=DRAWDOWN_COLOR,  alpha=0.15)

    ax.set_title(f"Rolling {window}-Month Sharpe Ratio", fontweight="bold", pad=12)
    ax.set_ylabel("Sharpe Ratio")

    plt.tight_layout()
    if save:
        fig.savefig(f"{OUTPUT_DIR}/04_rolling_sharpe.png", dpi=150)
    return fig


def plot_weight_evolution(
    weights: pd.DataFrame,
    save: bool = True,
) -> plt.Figure:
    """Stacked area chart showing portfolio weight evolution over time."""
    long_w  = weights.clip(lower=0)
    short_w = weights.clip(upper=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    long_w.plot.area(ax=ax1, stacked=True, alpha=0.7, legend=True)
    ax1.set_title("Long Leg Weights Over Time", fontweight="bold")
    ax1.set_ylabel("Weight")
    ax1.legend(loc="upper left", fontsize=8, ncol=3)

    short_w.plot.area(ax=ax2, stacked=True, alpha=0.7, legend=False, colormap="Reds")
    ax2.set_title("Short Leg Weights Over Time", fontweight="bold")
    ax2.set_ylabel("Weight")

    plt.tight_layout()
    if save:
        fig.savefig(f"{OUTPUT_DIR}/05_weight_evolution.png", dpi=150)
    return fig


def plot_all(results: dict) -> None:
    """Generates all charts from a results dict (from backtest.run_backtest)."""
    print("Generating charts...")
    plot_cumulative_returns(results["returns"], results["benchmark"])
    plot_drawdowns(results["returns"], results["benchmark"])
    plot_monthly_returns_heatmap(results["returns"])
    plot_rolling_sharpe(results["returns"])
    plot_weight_evolution(results["weights"])
    print(f"Charts saved to /{OUTPUT_DIR}/")
