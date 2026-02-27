"""
run.py
------
Main entry point. Run this to execute the full backtest pipeline:
    1. Download / load price data
    2. Compute momentum signals
    3. Build portfolio weights
    4. Run backtest
    5. Print performance summary
    6. Generate and save charts

Usage:
    python run.py
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.data_loader import download_prices
from src.signals import compute_momentum_signal, build_portfolio_weights
from src.backtest import run_backtest
from src.metrics import summary_table
from src.charts import plot_all


def main():
    print("=" * 60)
    print("  COMMODITY MOMENTUM STRATEGY — BACKTEST")
    print("=" * 60)

    # ── 1. Data ───────────────────────────────────────────────────
    print("\n[1/5] Loading price data...")
    prices = download_prices(start="2012-01-01")
    print(f"      {len(prices)} months × {len(prices.columns)} commodities")
    print(f"      Assets: {', '.join(prices.columns)}")

    # ── 2. Signal ─────────────────────────────────────────────────
    print("\n[2/5] Computing momentum signals (12-1 month)...")
    signal = compute_momentum_signal(prices, lookback=12, skip=1)

    # ── 3. Weights ────────────────────────────────────────────────
    print("\n[3/5] Building portfolio weights (top 3 long / bottom 3 short)...")
    weights = build_portfolio_weights(signal, top_n=3, bottom_n=3)

    # ── 4. Backtest ───────────────────────────────────────────────
    print("\n[4/5] Running backtest (10bps transaction costs)...")
    results = run_backtest(prices, weights, transaction_cost_bps=10)

    # ── 5. Results ────────────────────────────────────────────────
    print("\n[5/5] Performance Summary")
    print("-" * 50)
    table = summary_table(results["returns"], results["benchmark"])
    print(table.to_string())

    # ── 6. Charts ─────────────────────────────────────────────────
    print()
    plot_all(results)

    print("\nDone. ✓")


if __name__ == "__main__":
    main()
