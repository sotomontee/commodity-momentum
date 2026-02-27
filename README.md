# Commodity Momentum Strategy

A systematic cross-sectional momentum backtest on a basket of 10 commodity ETFs, built with clean modular Python and an interactive Streamlit dashboard.

---

## Live Demo

🔗 **[Launch Streamlit App →](https://your-app-url.streamlit.app)**  
*(Adjust lookback window, long/short positions, date range — all interactive)*

---

## Strategy Overview

**Momentum** is the tendency of recent winners to keep outperforming recent losers. In commodities, this effect is particularly persistent due to:

- **Slow-moving fundamentals** — supply shocks (weather, geopolitics, CapEx cycles) take months or years to resolve, creating durable trends
- **CTA flow reinforcement** — commodity trading advisors are large trend-following participants whose positioning amplifies momentum
- **Backwardation dynamics** — commodities in backwardation generate positive roll yield, which correlates with momentum signal strength

### Signal: 12-1 Month Cross-Sectional Momentum

Each month, every commodity is ranked by its return from 12 months ago to 1 month ago (skipping the most recent month to avoid short-term reversal noise):

$$\text{Signal}_{i,t} = \frac{P_{i,t-1}}{P_{i,t-12}} - 1$$

The top 3 commodities (strongest momentum) become **long** positions.  
The bottom 3 commodities (weakest momentum) become **short** positions.  
Equal weight within each leg. Portfolio rebalanced monthly.

---

## Universe

| ETF | Commodity | Sector |
|-----|-----------|--------|
| USO | Crude Oil | Energy |
| UNG | Natural Gas | Energy |
| GLD | Gold | Metals |
| SLV | Silver | Metals |
| CPER | Copper | Metals |
| WEAT | Wheat | Agriculture |
| CORN | Corn | Agriculture |
| SOYB | Soybeans | Agriculture |
| SGG | Sugar | Agriculture |
| JO | Coffee | Agriculture |

> ETFs are used as liquid, accessible proxies for commodity futures. A production-grade version would use continuous futures contracts (e.g. via Quandl/Nasdaq Data Link) to avoid ETF management fees and imperfect roll mechanics.

---

## Project Structure

```
commodity-momentum/
│
├── app.py                  # Streamlit interactive dashboard
├── run.py                  # CLI entry point — runs full pipeline
├── requirements.txt
│
├── src/
│   ├── data_loader.py      # Downloads & caches monthly ETF prices (yfinance)
│   ├── signals.py          # Momentum signal construction & portfolio weights
│   ├── backtest.py         # Return computation with transaction costs
│   ├── metrics.py          # Sharpe, Sortino, Calmar, drawdown, win rate
│   └── charts.py           # Matplotlib visualisations
│
└── notebooks/
    └── strategy.ipynb      # Full walkthrough with narrative & charts
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/sotomontee/commodity-momentum.git
cd commodity-momentum

# 2. Install dependencies
pip install -r requirements.txt

# 3a. Run CLI backtest
python run.py

# 3b. Launch interactive dashboard
streamlit run app.py
```

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Lookback window | 12 months | Return window for momentum signal |
| Skip | 1 month | Months excluded at end (reversal filter) |
| Long positions | 3 | Number of top-momentum commodities |
| Short positions | 3 | Number of bottom-momentum commodities |
| Transaction cost | 10 bps | One-way cost per trade |
| Rebalancing | Monthly | Portfolio reconstruction frequency |

All parameters are adjustable interactively in the Streamlit app.

---

## Performance Metrics

The backtest reports the following metrics for both strategy and equal-weight benchmark:

- **Annualised Return** — CAGR over the full period
- **Annualised Volatility** — standard deviation × √12
- **Sharpe Ratio** — excess return per unit of total risk
- **Sortino Ratio** — excess return per unit of *downside* risk only
- **Max Drawdown** — worst peak-to-trough decline
- **Calmar Ratio** — CAGR / |Max Drawdown|
- **Win Rate** — fraction of months with positive returns

---

## Limitations & Caveats

| Limitation | Impact | Potential fix |
|------------|--------|---------------|
| ETF proxies, not raw futures | Embeds management fees and imperfect roll | Use continuous futures (Quandl) |
| Small universe (10 assets) | Higher idiosyncratic risk, noisier signal | Expand to 20–30 commodities |
| Equal weighting | Sub-optimal risk allocation | Volatility-targeted position sizing |
| Lookback chosen in-sample | Risk of overfitting to historical data | Walk-forward parameter optimisation |
| No bid-ask spread modelling | Transaction costs may be understated | Use intraday spread data |

---

## Possible Extensions

- **Volatility scaling** — weight each position by 1/σ to equalise risk contribution
- **Carry combination** — blend momentum with commodity carry (roll yield proxy)
- **Macro regime filter** — condition on PMI, USD strength, or risk appetite
- **Walk-forward analysis** — re-estimate lookback window in expanding windows

---

## References

- Jegadeesh & Titman (1993) — *Returns to Buying Winners and Selling Losers*
- Gorton, Hayashi & Rouwenhorst (2012) — *The Fundamentals of Commodity Futures Returns*
- Asness, Moskowitz & Pedersen (2013) — *Value and Momentum Everywhere* (AQR)

---

## Disclaimer

This project is for educational and research purposes only. It does not constitute financial advice. Past backtest performance does not predict future results.

---

*Built with Python · pandas · matplotlib · Streamlit*
