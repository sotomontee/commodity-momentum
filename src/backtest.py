"""
backtest.py
-----------
Runs the portfolio backtest using the momentum weights from signals.py.

Methodology:
    - Monthly rebalancing
    - Equal-weight long/short legs
    - Transaction costs: 10bps per trade (conservative for ETFs)
    - No leverage: each leg is 100% notional (total gross = 200%)
    - Returns computed on next-month prices (no look-ahead bias)
"""

import pandas as pd
import numpy as np


def compute_portfolio_returns(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    transaction_cost_bps: float = 10.0,
) -> pd.Series:
    """
    Computes monthly portfolio returns from weights and prices.

    Parameters
    ----------
    prices                : monthly price DataFrame
    weights               : portfolio weights (from signals.build_portfolio_weights)
    transaction_cost_bps  : one-way transaction cost in basis points

    Returns
    -------
    pd.Series : monthly portfolio returns
    """
    tc = transaction_cost_bps / 10_000  # convert bps to decimal

    # Forward returns: return from t to t+1
    fwd_returns = prices.pct_change().shift(-1)

    # Align weights and returns
    weights_aligned = weights.reindex(fwd_returns.index)

    # Gross portfolio return each month (before costs)
    gross_returns = (weights_aligned * fwd_returns).sum(axis=1)

    # Turnover: sum of absolute weight changes each month
    turnover = weights_aligned.diff().abs().sum(axis=1)

    # Net return after transaction costs
    net_returns = gross_returns - turnover * tc

    # Drop NaN rows (warm-up period)
    net_returns = net_returns.dropna()

    return net_returns


def compute_benchmark_returns(prices: pd.DataFrame) -> pd.Series:
    """
    Computes an equal-weight long-only commodity benchmark for comparison.
    """
    monthly_returns = prices.pct_change()
    benchmark = monthly_returns.mean(axis=1).dropna()
    benchmark.name = "Equal-Weight Benchmark"
    return benchmark


def run_backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    transaction_cost_bps: float = 10.0,
) -> dict:
    """
    Runs the full backtest and returns a results dictionary.

    Returns
    -------
    dict with keys:
        - returns        : pd.Series of monthly net returns
        - cum_returns    : pd.Series of cumulative returns
        - benchmark      : pd.Series of benchmark monthly returns
        - cum_benchmark  : pd.Series of benchmark cumulative returns
        - weights        : pd.DataFrame of portfolio weights over time
    """
    returns = compute_portfolio_returns(prices, weights, transaction_cost_bps)
    benchmark = compute_benchmark_returns(prices)

    # Align benchmark to strategy period
    benchmark = benchmark.reindex(returns.index)

    cum_returns = (1 + returns).cumprod()
    cum_benchmark = (1 + benchmark).cumprod()

    return {
        "returns": returns,
        "cum_returns": cum_returns,
        "benchmark": benchmark,
        "cum_benchmark": cum_benchmark,
        "weights": weights,
    }


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.data_loader import download_prices
    from src.signals import compute_momentum_signal, build_portfolio_weights

    prices = download_prices()
    signal = compute_momentum_signal(prices)
    weights = build_portfolio_weights(signal)
    results = run_backtest(prices, weights)

    print("Monthly returns (last 5):")
    print(results["returns"].tail())
    print(f"\nTotal return: {results['cum_returns'].iloc[-1] - 1:.2%}")
