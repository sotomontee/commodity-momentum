"""
data_loader.py
--------------
Downloads adjusted closing prices for commodity ETF proxies via yfinance.
These ETFs are used as liquid, accessible proxies for commodity futures exposure.
"""

import yfinance as yf
import pandas as pd
import os

# Commodity ETF proxies
# Using ETFs instead of raw futures to avoid roll-cost complexity for this backtest
COMMODITY_TICKERS = {
    "Crude Oil":     "USO",   # United States Oil Fund
    "Gold":          "GLD",   # SPDR Gold Shares
    "Silver":        "SLV",   # iShares Silver Trust
    "Copper":        "CPER",  # United States Copper Index Fund
    "Natural Gas":   "UNG",   # United States Natural Gas Fund
    "Wheat":         "WEAT",  # Teucrium Wheat Fund
    "Corn":          "CORN",  # Teucrium Corn Fund
    "Soybeans":      "SOYB",  # Teucrium Soybean Fund
    "Sugar":         "SGG",   # iPath Bloomberg Sugar Subindex
    "Coffee":        "JO",    # iPath Bloomberg Coffee Subindex
}

def download_prices(
    tickers: dict = COMMODITY_TICKERS,
    start: str = "2012-01-01",
    end: str = None,
    cache_path: str = "data/prices.csv",
) -> pd.DataFrame:
    """
    Downloads monthly adjusted close prices for the given tickers.

    Parameters
    ----------
    tickers     : dict mapping commodity name -> ticker symbol
    start       : start date (YYYY-MM-DD)
    end         : end date (YYYY-MM-DD); defaults to today
    cache_path  : if file exists, load from cache instead of re-downloading

    Returns
    -------
    pd.DataFrame : monthly prices, columns = commodity names, index = date
    """
    if os.path.exists(cache_path):
        print(f"Loading prices from cache: {cache_path}")
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return prices

    print("Downloading price data from Yahoo Finance...")
    raw = yf.download(
        list(tickers.values()),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )["Close"]

    # Rename columns from tickers to commodity names
    ticker_to_name = {v: k for k, v in tickers.items()}
    raw.rename(columns=ticker_to_name, inplace=True)

    # Resample to month-end
    monthly = raw.resample("ME").last()

    # Drop columns with too many NaNs (< 60 months of data)
    monthly = monthly.dropna(thresh=60, axis=1)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    monthly.to_csv(cache_path)
    print(f"Saved to {cache_path}")

    return monthly


if __name__ == "__main__":
    df = download_prices()
    print(df.tail())
    print(f"\nShape: {df.shape}")
    print(f"Commodities: {list(df.columns)}")
