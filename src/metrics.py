"""
metrics.py
----------
Performance and risk metrics for the backtest results.
These are the standard metrics used in quantitative research and fund reporting.
"""

import pandas as pd
import numpy as np


def annualised_return(returns: pd.Series, periods_per_year: int = 12) -> float:
    """Compound annualised growth rate (CAGR)."""
    n = len(returns)
    total = (1 + returns).prod()
    return total ** (periods_per_year / n) - 1


def annualised_volatility(returns: pd.Series, periods_per_year: int = 12) -> float:
    """Annualised standard deviation of returns."""
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 12,
) -> float:
    """
    Annualised Sharpe ratio.
    risk_free_rate: annual rate (default 2%)
    """
    monthly_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - monthly_rf
    if excess.std() == 0:
        return np.nan
    return (excess.mean() / excess.std()) * np.sqrt(periods_per_year)


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown."""
    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    return drawdown.min()


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Full drawdown time series for charting."""
    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    return (cum - rolling_max) / rolling_max


def calmar_ratio(returns: pd.Series, periods_per_year: int = 12) -> float:
    """CAGR / |Max Drawdown|. Measures return per unit of drawdown risk."""
    cagr = annualised_return(returns, periods_per_year)
    mdd = abs(max_drawdown(returns))
    return cagr / mdd if mdd != 0 else np.nan


def win_rate(returns: pd.Series) -> float:
    """Fraction of months with positive returns."""
    return (returns > 0).sum() / len(returns)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 12,
) -> float:
    """
    Sortino ratio — like Sharpe but only penalises downside volatility.
    Better metric for asymmetric return distributions.
    """
    monthly_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - monthly_rf
    downside = excess[excess < 0].std()
    if downside == 0:
        return np.nan
    return (excess.mean() / downside) * np.sqrt(periods_per_year)


def summary_table(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series = None,
    risk_free_rate: float = 0.02,
) -> pd.DataFrame:
    """
    Produces a clean summary table comparing strategy vs benchmark.

    Returns
    -------
    pd.DataFrame : metrics as rows, strategy/benchmark as columns
    """
    def _metrics(r):
        return {
            "Annualised Return":    f"{annualised_return(r):.2%}",
            "Annualised Volatility":f"{annualised_volatility(r):.2%}",
            "Sharpe Ratio":         f"{sharpe_ratio(r, risk_free_rate):.2f}",
            "Sortino Ratio":        f"{sortino_ratio(r, risk_free_rate):.2f}",
            "Max Drawdown":         f"{max_drawdown(r):.2%}",
            "Calmar Ratio":         f"{calmar_ratio(r):.2f}",
            "Win Rate (Monthly)":   f"{win_rate(r):.2%}",
            "Total Return":         f"{(1 + r).prod() - 1:.2%}",
        }

    data = {"Strategy": _metrics(strategy_returns)}

    if benchmark_returns is not None:
        bench_aligned = benchmark_returns.reindex(strategy_returns.index).dropna()
        data["Benchmark"] = _metrics(bench_aligned)

    return pd.DataFrame(data)


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.data_loader import download_prices
    from src.signals import compute_momentum_signal, build_portfolio_weights
    from src.backtest import run_backtest

    prices = download_prices()
    signal = compute_momentum_signal(prices)
    weights = build_portfolio_weights(signal)
    results = run_backtest(prices, weights)

    table = summary_table(results["returns"], results["benchmark"])
    print(table.to_string())
