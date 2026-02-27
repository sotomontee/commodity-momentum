"""
signals.py
----------
Constructs the commodity momentum signal.

Strategy:
    - Lookback: 12-month return, skipping the most recent month (12-1 momentum)
    - This is the standard Jegadeesh-Titman (1993) window, widely used in
      cross-sectional momentum research and validated for commodities by
      Gorton, Hayashi & Rouwenhorst (2012) and AQR papers.

Why skip the last month?
    - The most recent month often shows short-term reversal (microstructure noise,
      bid-ask bounce), which would dilute the momentum signal.
"""

import pandas as pd
import numpy as np


def compute_momentum_signal(
    prices: pd.DataFrame,
    lookback: int = 12,
    skip: int = 1,
) -> pd.DataFrame:
    """
    Computes cross-sectional momentum signal for each commodity.

    Parameters
    ----------
    prices   : monthly price DataFrame (rows = dates, cols = commodities)
    lookback : total lookback window in months (default: 12)
    skip     : months to skip at the end (default: 1, avoids reversal effect)

    Returns
    -------
    pd.DataFrame : momentum returns, same shape as prices
    """
    # Return from t-lookback to t-skip
    signal = prices.shift(skip) / prices.shift(lookback) - 1
    return signal


def rank_commodities(signal: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectionally ranks commodities by momentum signal each month.
    Rank 1 = weakest momentum, Rank N = strongest.

    Returns
    -------
    pd.DataFrame : integer ranks (1 to N), NaN where signal is NaN
    """
    return signal.rank(axis=1, na_option="keep")


def build_portfolio_weights(
    signal: pd.DataFrame,
    top_n: int = 3,
    bottom_n: int = 3,
) -> pd.DataFrame:
    """
    Builds a long/short portfolio from the momentum signal.
    - Long: top_n commodities (highest momentum)
    - Short: bottom_n commodities (lowest momentum)
    - Equal weight within each leg
    - Portfolio is dollar-neutral (long $ = short $)

    Parameters
    ----------
    signal   : momentum signal DataFrame
    top_n    : number of commodities to go long
    bottom_n : number of commodities to go short

    Returns
    -------
    pd.DataFrame : weights (-1/bottom_n to +1/top_n), same shape as signal
    """
    weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    ranks = rank_commodities(signal)

    n_assets = ranks.notna().sum(axis=1)  # number of valid assets each month

    for date in signal.index:
        row = ranks.loc[date].dropna()
        if len(row) < top_n + bottom_n:
            continue  # skip if not enough assets

        n = len(row)
        long_threshold = n - top_n       # ranks above this are longs
        short_threshold = bottom_n       # ranks below or equal to this are shorts

        for asset, rank in row.items():
            if rank > long_threshold:
                weights.loc[date, asset] = 1.0 / top_n
            elif rank <= short_threshold:
                weights.loc[date, asset] = -1.0 / bottom_n

    return weights


if __name__ == "__main__":
    # Quick sanity check
    import sys
    sys.path.append(".")
    from src.data_loader import download_prices

    prices = download_prices()
    signal = compute_momentum_signal(prices)
    weights = build_portfolio_weights(signal)

    print("Signal (last 3 rows):")
    print(signal.tail(3).round(3))
    print("\nWeights (last 3 rows):")
    print(weights.tail(3))
