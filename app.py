"""
app.py
------
Streamlit interactive dashboard for the Commodity Momentum Strategy backtest.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
import sys
import os

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(__file__))

from src.data_loader import download_prices, COMMODITY_TICKERS
from src.signals import compute_momentum_signal, build_portfolio_weights
from src.backtest import run_backtest
from src.metrics import (
    summary_table,
    annualised_return,
    annualised_volatility,
    sharpe_ratio,
    max_drawdown,
    drawdown_series,
    win_rate,
    sortino_ratio,
    calmar_ratio,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Commodity Momentum",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background-color: #0D0F14;
        color: #E8EAF0;
    }

    section[data-testid="stSidebar"] {
        background-color: #13161E;
        border-right: 1px solid #1E2230;
    }

    .metric-card {
        background: #13161E;
        border: 1px solid #1E2230;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 10px;
    }

    .metric-label {
        font-size: 11px;
        font-family: 'DM Mono', monospace;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 4px;
    }

    .metric-value {
        font-size: 26px;
        font-weight: 600;
        font-family: 'DM Mono', monospace;
        color: #E8EAF0;
    }

    .metric-value.positive { color: #34D399; }
    .metric-value.negative { color: #F87171; }

    .metric-delta {
        font-size: 12px;
        color: #6B7280;
        font-family: 'DM Mono', monospace;
        margin-top: 2px;
    }

    .section-header {
        font-size: 13px;
        font-family: 'DM Mono', monospace;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        border-bottom: 1px solid #1E2230;
        padding-bottom: 8px;
        margin: 28px 0 16px 0;
    }

    .pill {
        display: inline-block;
        background: #1E2230;
        border: 1px solid #2A3045;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 11px;
        font-family: 'DM Mono', monospace;
        color: #94A3B8;
        margin: 2px;
    }

    .pill.long  { border-color: #34D399; color: #34D399; background: #0D2B22; }
    .pill.short { border-color: #F87171; color: #F87171; background: #2B0D0D; }

    h1 {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        color: #E8EAF0 !important;
        font-size: 28px !important;
    }

    .stSlider > div > div { background: #1E2230; }
    div[data-testid="stMetric"] { background: transparent; }

    .stSelectbox label, .stSlider label, .stDateInput label {
        color: #94A3B8 !important;
        font-size: 12px !important;
        font-family: 'DM Mono', monospace !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }

    .stButton > button {
        background: #2563EB;
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        width: 100%;
        padding: 10px;
    }
    .stButton > button:hover { background: #1D4ED8; }

    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ─────────────────────────────────────────────────────
CHART_BG    = "#13161E"
CHART_FG    = "#E8EAF0"
GRID_COLOR  = "#1E2230"
BLUE        = "#3B82F6"
AMBER       = "#F59E0B"
RED         = "#F87171"
GREEN       = "#34D399"

plt.rcParams.update({
    "figure.facecolor":     CHART_BG,
    "axes.facecolor":       CHART_BG,
    "axes.edgecolor":       GRID_COLOR,
    "axes.labelcolor":      "#94A3B8",
    "axes.titlecolor":      CHART_FG,
    "xtick.color":          "#6B7280",
    "ytick.color":          "#6B7280",
    "grid.color":           GRID_COLOR,
    "grid.linewidth":       0.6,
    "axes.grid":            True,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.spines.left":     False,
    "font.family":          "sans-serif",
    "legend.facecolor":     "#1A1D27",
    "legend.edgecolor":     GRID_COLOR,
    "legend.labelcolor":    "#94A3B8",
    "legend.fontsize":      9,
    "axes.titlesize":       12,
    "axes.titleweight":     "bold",
})


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(start: str) -> pd.DataFrame:
    return download_prices(start=start)


def color_val(val: float, good: str = "positive") -> str:
    if good == "positive":
        return "positive" if val >= 0 else "negative"
    else:
        return "positive" if val <= 0 else "negative"


def metric_card(label: str, value: str, delta: str = "", css_class: str = "") -> str:
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {css_class}">{value}</div>
        {delta_html}
    </div>
    """


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Parameters")
    st.markdown("---")

    st.markdown('<div class="section-header">Signal</div>', unsafe_allow_html=True)
    lookback = st.slider("Lookback Window (months)", min_value=3, max_value=24, value=12, step=1)
    skip = st.slider("Skip Recent Months", min_value=0, max_value=3, value=1, step=1)

    st.markdown('<div class="section-header">Portfolio</div>', unsafe_allow_html=True)
    top_n    = st.slider("Long Positions (top N)", min_value=1, max_value=5, value=3)
    bottom_n = st.slider("Short Positions (bottom N)", min_value=1, max_value=5, value=3)

    st.markdown('<div class="section-header">Date Range</div>', unsafe_allow_html=True)
    start_year = st.selectbox("Start Year", options=list(range(2012, 2024)), index=0)
    end_year   = st.selectbox("End Year",   options=list(range(2014, 2026)), index=9)

    st.markdown('<div class="section-header">Costs</div>', unsafe_allow_html=True)
    tc_bps = st.slider("Transaction Cost (bps)", min_value=0, max_value=50, value=10, step=5)

    st.markdown("---")
    run_btn = st.button("▶  Run Backtest")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 📈 Commodity Momentum Strategy")
st.markdown(
    '<span class="pill">Cross-Sectional Momentum</span>'
    '<span class="pill">Monthly Rebalancing</span>'
    '<span class="pill">Long / Short</span>'
    '<span class="pill">ETF Proxies</span>',
    unsafe_allow_html=True,
)
st.markdown("")

# ── Main logic ────────────────────────────────────────────────────────────────
if "results" not in st.session_state or run_btn:
    with st.spinner("Running backtest..."):
        start_date = f"{start_year}-01-01"
        end_date   = f"{end_year}-12-31"

        prices = load_data(start_date)
        prices = prices[prices.index.year <= end_year]

        if len(prices) < lookback + skip + 6:
            st.error("Not enough data for the selected parameters. Adjust date range or lookback.")
            st.stop()

        signal  = compute_momentum_signal(prices, lookback=lookback, skip=skip)
        weights = build_portfolio_weights(signal, top_n=top_n, bottom_n=bottom_n)
        results = run_backtest(prices, weights, transaction_cost_bps=tc_bps)

        st.session_state["results"] = results
        st.session_state["prices"]  = prices
        st.session_state["weights"] = weights
        st.session_state["signal"]  = signal

results = st.session_state["results"]
prices  = st.session_state["prices"]
weights = st.session_state["weights"]
signal  = st.session_state["signal"]

r  = results["returns"]
br = results["benchmark"].reindex(r.index).dropna()

# ── KPI Row ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Performance Metrics</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)

ann_ret  = annualised_return(r)
ann_vol  = annualised_volatility(r)
sr       = sharpe_ratio(r)
mdd      = max_drawdown(r)
wr       = win_rate(r)
calmar   = calmar_ratio(r)

bench_sr = sharpe_ratio(br)

with c1:
    st.markdown(metric_card(
        "Ann. Return", f"{ann_ret:.1%}",
        delta=f"Benchmark: {annualised_return(br):.1%}",
        css_class=color_val(ann_ret)
    ), unsafe_allow_html=True)

with c2:
    st.markdown(metric_card(
        "Ann. Volatility", f"{ann_vol:.1%}",
        delta=f"Benchmark: {annualised_volatility(br):.1%}"
    ), unsafe_allow_html=True)

with c3:
    st.markdown(metric_card(
        "Sharpe Ratio", f"{sr:.2f}",
        delta=f"Benchmark: {bench_sr:.2f}",
        css_class=color_val(sr)
    ), unsafe_allow_html=True)

with c4:
    st.markdown(metric_card(
        "Max Drawdown", f"{mdd:.1%}",
        delta=f"Benchmark: {max_drawdown(br):.1%}",
        css_class=color_val(mdd, good="negative")
    ), unsafe_allow_html=True)

with c5:
    st.markdown(metric_card(
        "Win Rate", f"{wr:.1%}",
        css_class=color_val(wr - 0.5)
    ), unsafe_allow_html=True)

with c6:
    st.markdown(metric_card(
        "Calmar Ratio", f"{calmar:.2f}",
        css_class=color_val(calmar)
    ), unsafe_allow_html=True)

# ── Charts Row 1 ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Returns</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([3, 2])

with col_left:
    # Cumulative returns
    fig, ax = plt.subplots(figsize=(10, 4))
    cum_s = results["cum_returns"]
    cum_b = results["cum_benchmark"].reindex(cum_s.index)

    ax.plot(cum_s.index, cum_s, color=BLUE,  lw=2,   label="Momentum Strategy")
    ax.plot(cum_b.index, cum_b, color=AMBER, lw=1.5, label="Equal-Weight Benchmark", linestyle="--", alpha=0.8)
    ax.axhline(1, color=GRID_COLOR, lw=0.8)
    ax.fill_between(cum_s.index, cum_s, 1, where=cum_s >= 1, color=BLUE, alpha=0.07)
    ax.fill_between(cum_s.index, cum_s, 1, where=cum_s <  1, color=RED,  alpha=0.07)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax.set_title("Cumulative Returns")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with col_right:
    # Return distribution
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(r * 100, bins=30, color=BLUE, alpha=0.75, edgecolor=CHART_BG, linewidth=0.5)
    ax.axvline(0,         color=GRID_COLOR, lw=1)
    ax.axvline(r.mean() * 100, color=GREEN, lw=1.5, linestyle="--", label=f"Mean: {r.mean()*100:.2f}%")

    ax.set_title("Monthly Return Distribution")
    ax.set_xlabel("Monthly Return (%)")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── Charts Row 2 ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Risk</div>', unsafe_allow_html=True)

col_left2, col_right2 = st.columns([3, 2])

with col_left2:
    # Drawdown
    fig, ax = plt.subplots(figsize=(10, 3.5))
    dd = drawdown_series(r)
    dd_b = drawdown_series(br)

    ax.fill_between(dd.index, dd * 100,   0, color=RED,   alpha=0.45, label="Strategy")
    ax.fill_between(dd_b.index, dd_b * 100, 0, color=AMBER, alpha=0.2,  label="Benchmark")
    ax.plot(dd.index, dd * 100, color=RED, lw=1)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_title("Drawdown")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with col_right2:
    # Rolling 12m Sharpe
    fig, ax = plt.subplots(figsize=(5, 3.5))
    win = min(24, len(r) // 2)
    rs = (r.rolling(win).mean() / r.rolling(win).std()) * np.sqrt(12)

    ax.plot(rs.index, rs, color=BLUE, lw=1.5)
    ax.axhline(0, color=GRID_COLOR, lw=1)
    ax.fill_between(rs.index, rs, 0, where=rs >= 0, color=BLUE, alpha=0.12)
    ax.fill_between(rs.index, rs, 0, where=rs <  0, color=RED,  alpha=0.12)

    ax.set_title(f"Rolling {win}m Sharpe Ratio")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── Current Positions ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Current Portfolio Positions</div>', unsafe_allow_html=True)

last_weights = weights.iloc[-1]
last_signal  = signal.iloc[-1].dropna().sort_values(ascending=False)

long_assets  = last_weights[last_weights > 0].index.tolist()
short_assets = last_weights[last_weights < 0].index.tolist()

col_l, col_s, col_sig = st.columns([2, 2, 3])

with col_l:
    st.markdown("**LONG**")
    for a in long_assets:
        st.markdown(f'<span class="pill long">▲ {a}</span>', unsafe_allow_html=True)
    if not long_assets:
        st.markdown("*No positions*")

with col_s:
    st.markdown("**SHORT**")
    for a in short_assets:
        st.markdown(f'<span class="pill short">▼ {a}</span>', unsafe_allow_html=True)
    if not short_assets:
        st.markdown("*No positions*")

with col_sig:
    st.markdown("**Momentum Rankings (latest month)**")
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = [GREEN if a in long_assets else RED if a in short_assets else "#4B5563"
              for a in last_signal.index]
    bars = ax.barh(last_signal.index[::-1], last_signal.values[::-1] * 100,
                   color=colors[::-1], height=0.6)
    ax.axvline(0, color=GRID_COLOR, lw=1)
    ax.set_xlabel("12-1M Return (%)")
    ax.set_title("Signal Strength")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── Monthly Heatmap ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Monthly Returns Heatmap</div>', unsafe_allow_html=True)

monthly = r.copy()
monthly.index = pd.to_datetime(monthly.index)
pivot = pd.DataFrame({
    "year":  monthly.index.year,
    "month": monthly.index.month,
    "ret":   monthly.values
}).pivot(index="year", columns="month", values="ret")
pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

fig, ax = plt.subplots(figsize=(14, max(3, len(pivot) * 0.55 + 1)))
vmax = pivot.abs().max().max()
im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

ax.set_xticks(range(12))
ax.set_xticklabels(pivot.columns, color="#94A3B8")
ax.set_yticks(range(len(pivot)))
ax.set_yticklabels(pivot.index, color="#94A3B8")

for i in range(len(pivot)):
    for j in range(12):
        val = pivot.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.1%}", ha="center", va="center",
                    fontsize=8, color="black", fontweight="500")

plt.colorbar(im, ax=ax, format=mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.set_title("Monthly Returns — Commodity Momentum Strategy")
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="color:#4B5563; font-size:11px; font-family: DM Mono, monospace;">'
    'Strategy: Cross-sectional 12-1M momentum on commodity ETFs · Monthly rebalancing · '
    'Data: Yahoo Finance via yfinance · For research purposes only.'
    '</div>',
    unsafe_allow_html=True
)
