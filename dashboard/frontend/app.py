# dashboard/frontend/app.py
#
# AlphaFlow — Live Dashboard
#
# Sections:
#   1. Header + live market ticker
#   2. Portfolio Allocation (pie + bar chart)
#   3. Backtest Performance (vs benchmarks)
#   4. TFT Predictions (signal table)
#   5. Macro Regime Indicator
#   6. Drift Monitor Status
#   7. Pipeline Health
#
# Deploy free on Streamlit Cloud:
#   1. Push to GitHub
#   2. Go to share.streamlit.io
#   3. Connect repo → set main file = dashboard/frontend/app.py
#
# Run locally:
#   streamlit run dashboard/frontend/app.py
# dashboard/frontend/app.py
#
# AlphaFlow — Bloomberg Terminal
# Production-grade institutional portfolio intelligence platform

import json
import os
import sys
import time  # noqa: F401
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px  # noqa: F401
from plotly.subplots import make_subplots  # noqa: F401
import streamlit as st
import yfinance as yf
import requests

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import TICKERS, MULTI_ASSET_TICKERS, LOCAL_DATA_DIR, BENCHMARK_TICKER

REPORTS = Path("reports")
DATA_DIR = Path(LOCAL_DATA_DIR)
ALL_TICKERS = TICKERS + MULTI_ASSET_TICKERS

st.set_page_config(
    page_title="AlphaFlow Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

*, body, .stApp { background-color: #050508 !important; color: #b8bcc8; font-family: 'IBM Plex Mono', monospace; }
.main .block-container { padding: 0.5rem 1rem; max-width: 100%; }

.header-bar {
    background: linear-gradient(90deg, #050508, #0a0a14);
    border-bottom: 1px solid #00e5ff33;
    padding: 8px 0;
    margin-bottom: 12px;
}

.kpi-card {
    background: #08080f;
    border: 1px solid #1a1a2e;
    border-top: 2px solid var(--accent, #00e5ff);
    padding: 10px 14px;
    border-radius: 2px;
}

.kpi-label { font-size: 9px; letter-spacing: 3px; color: #4a5568; text-transform: uppercase; margin-bottom: 4px; }
.kpi-value { font-size: 22px; font-weight: 600; color: var(--accent, #00e5ff); font-family: 'IBM Plex Mono'; }
.kpi-delta { font-size: 10px; color: #4a5568; margin-top: 2px; }

.price-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 5px 8px; border-bottom: 1px solid #0d0d18;
    font-size: 12px; transition: background 0.1s;
}
.price-row:hover { background: #0a0a14; }
.price-ticker { color: #7a8099; letter-spacing: 1px; }
.price-val { color: #b8bcc8; }
.pos { color: #00ff9d !important; }
.neg { color: #ff4444 !important; }

.section-title {
    font-size: 9px; letter-spacing: 4px; color: #2a3050;
    text-transform: uppercase; padding: 6px 0 8px;
    border-bottom: 1px solid #0d0d18; margin-bottom: 8px;
}

.alloc-row {
    display: flex; align-items: center; gap: 8px;
    padding: 4px 0; font-size: 11px;
}
.alloc-ticker { color: #5a6480; width: 55px; }
.alloc-bar-bg { flex: 1; height: 3px; background: #0d0d18; border-radius: 1px; }
.alloc-bar { height: 3px; background: #00e5ff; border-radius: 1px; }
.alloc-pct { color: #00e5ff; width: 40px; text-align: right; }

.regime-badge {
    display: inline-block; padding: 3px 10px;
    border-radius: 2px; font-size: 10px; letter-spacing: 2px;
    font-weight: 600;
}
.regime-bear { background: #ff444415; color: #ff4444; border: 1px solid #ff444433; }
.regime-bull { background: #00ff9d15; color: #00ff9d; border: 1px solid #00ff9d33; }
.regime-volatile { background: #ffd70015; color: #ffd700; border: 1px solid #ffd70033; }
.regime-neutral { background: #7b61ff15; color: #7b61ff; border: 1px solid #7b61ff33; }

.terminal-input .stTextInput input {
    background: #08080f !important;
    border: 1px solid #00e5ff44 !important;
    border-radius: 2px !important;
    color: #00e5ff !important;
    font-family: 'IBM Plex Mono' !important;
    font-size: 12px !important;
}

.stTabs [data-baseweb="tab-list"] { background: #050508; border-bottom: 1px solid #1a1a2e; gap: 0; }
.stTabs [data-baseweb="tab"] {
    background: transparent; color: #2a3050;
    font-size: 10px; letter-spacing: 2px; padding: 8px 16px;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] { color: #00e5ff !important; border-bottom-color: #00e5ff !important; background: transparent !important; }

div[data-testid="stMetricValue"] { color: #00e5ff !important; font-family: 'IBM Plex Mono' !important; }
div[data-testid="stMetricDelta"] svg { display: none; }
.stButton button {
    background: #08080f !important; color: #00e5ff !important;
    border: 1px solid #00e5ff33 !important;
    font-family: 'IBM Plex Mono' !important; font-size: 10px !important;
    letter-spacing: 1px !important; border-radius: 2px !important;
    padding: 4px 10px !important;
}
.stButton button:hover { border-color: #00e5ff88 !important; background: #00e5ff08 !important; }

.stChatMessage { background: #08080f !important; border: 1px solid #1a1a2e !important; border-radius: 2px !important; }
.stChatInput textarea {
    background: #08080f !important; color: #00ff9d !important;
    font-family: 'IBM Plex Mono' !important; font-size: 12px !important;
    border: 1px solid #00e5ff44 !important; border-radius: 2px !important;
}

h1, h2, h3 { color: #00e5ff !important; font-family: 'IBM Plex Mono' !important; font-weight: 500 !important; }
.stSelectbox label, .stSlider label { color: #4a5568 !important; font-size: 10px !important; }
</style>
""",
    unsafe_allow_html=True,
)

CHART_LAYOUT = dict(
    paper_bgcolor="#050508",
    plot_bgcolor="#08080f",
    font=dict(color="#5a6480", family="IBM Plex Mono", size=10),
    margin=dict(l=40, r=20, t=30, b=30),
    xaxis=dict(gridcolor="#0d0d18", zerolinecolor="#0d0d18", showgrid=True),
    yaxis=dict(gridcolor="#0d0d18", zerolinecolor="#0d0d18", showgrid=True),
    legend=dict(bgcolor="#08080f", bordercolor="#1a1a2e", borderwidth=1, font_size=10),
)


# ── DATA LOADERS ──────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_json(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data(ttl=30)
def load_live_prices() -> dict:
    p = DATA_DIR / "streaming" / "live_prices.json"
    if p.exists():
        try:
            with open(p) as f:
                d = json.load(f)
            return d.get("prices", {})
        except Exception:
            pass
    # Fallback: yfinance
    try:
        data = yf.download(
            TICKERS[:10] + [BENCHMARK_TICKER],
            period="2d",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if hasattr(data.columns, "levels"):
            close = data["Close"]
        else:
            close = data[["Close"]]
        prices = {}
        for t in TICKERS[:10] + [BENCHMARK_TICKER]:
            if t in close.columns:
                vals = close[t].dropna()
                if len(vals) >= 2:
                    p_val = float(vals.iloc[-1])
                    prev = float(vals.iloc[-2])
                    pct = (p_val - prev) / prev * 100
                    prices[t] = {"price": round(p_val, 2), "change_pct": round(pct, 3)}
        return prices
    except Exception:
        return {}


@st.cache_data(ttl=3600)
def load_predictions() -> dict:
    ticker_map = {str(i): t for i, t in enumerate(ALL_TICKERS)}
    try:
        df = pd.read_parquet(DATA_DIR / "predictions.parquet")
        latest = df.sort_values("date").groupby("ticker").last().reset_index()
        return {
            ticker_map.get(str(r["ticker"]), str(r["ticker"])): {
                "pred_5d": round(float(r["pred_p50"]) * 100, 3),
                "signal": "BUY"
                if float(r["pred_p50"]) > 0.001
                else "SELL"
                if float(r["pred_p50"]) < -0.001
                else "HOLD",
                "p10": round(float(r.get("pred_p10", 0)) * 100, 3),
                "p90": round(float(r.get("pred_p90", 0)) * 100, 3),
            }
            for _, r in latest.iterrows()
        }
    except Exception:
        return {}


@st.cache_data(ttl=60)
def load_history(ticker: str, days: int = 365) -> pd.DataFrame:
    try:
        from datetime import timedelta

        start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.reset_index()
    except Exception:
        return pd.DataFrame()


# ── LOAD ALL STATE ────────────────────────────────────────────────────
allocation = load_json(str(REPORTS / "allocation.json"))
metrics_list = load_json(str(REPORTS / "metrics.json")) or []
if isinstance(metrics_list, dict):
    metrics_list = [metrics_list]
regime = allocation.get("regime", "unknown").upper()
regime_conf = allocation.get("regime_confidence", 0)
weights = allocation.get("weights", {})
sharpe_mkt = allocation.get("markowitz_sharpe", 0)
af_metrics = next(
    (m for m in metrics_list if "AlphaFlow" in str(m.get("name", ""))), {}
)
bh_metrics = next((m for m in metrics_list if "Buy Hold" in str(m.get("name", ""))), {})
live_prices = load_live_prices()
predictions = load_predictions()
stress_data = load_json(str(REPORTS / "stress_test_report.json"))
regime_class = {
    "BEAR": "regime-bear",
    "BULL": "regime-bull",
    "VOLATILE": "regime-volatile",
}.get(regime, "regime-neutral")


# ── HEADER ────────────────────────────────────────────────────────────
now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d  %H:%M:%S UTC")
st.markdown(
    f"""
<div style="display:flex;justify-content:space-between;align-items:center;
padding:6px 0;border-bottom:1px solid #0d0d18;margin-bottom:12px">
  <div style="display:flex;align-items:center;gap:24px">
    <span style="color:#00e5ff;font-size:16px;font-weight:600;letter-spacing:3px">⚡ ALPHAFLOW</span>
    <span style="color:#2a3050;font-size:10px">INSTITUTIONAL PORTFOLIO INTELLIGENCE</span>
  </div>
  <div style="display:flex;align-items:center;gap:16px">
    <span class="regime-badge {regime_class}">{regime}</span>
    <span style="color:#2a3050;font-size:10px">{now_str}</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── KPI ROW ───────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
kpis = [
    (
        k1,
        "SHARPE RATIO",
        f"{af_metrics.get('sharpe_ratio', sharpe_mkt):.3f}",
        "AlphaFlow TFT",
        "#00e5ff",
    ),
    (
        k2,
        "TOTAL RETURN",
        f"{af_metrics.get('total_return', 0):.1f}%",
        f"vs SPY {bh_metrics.get('total_return', 0):.1f}%",
        "#00ff9d" if af_metrics.get("total_return", 0) > 0 else "#ff4444",
    ),
    (
        k3,
        "MAX DRAWDOWN",
        f"{af_metrics.get('max_drawdown', 0):.1f}%",
        "peak-to-trough",
        "#ffd700",
    ),
    (
        k4,
        "HIT RATE",
        f"{af_metrics.get('hit_rate', 0):.1f}%",
        "directional accuracy",
        "#7b61ff",
    ),
    (
        k5,
        "REGIME",
        regime,
        f"{regime_conf:.0%} confidence",
        "#ff4444" if regime == "BEAR" else "#00ff9d",
    ),
    (k6, "ASSETS", "39", "equity+crypto+bonds+cmdty", "#00e5ff"),
    (
        k7,
        "SORTINO",
        f"{af_metrics.get('sortino_ratio', 0):.3f}",
        "downside-adjusted",
        "#ff6b35",
    ),
]
for col, label, value, delta, color in kpis:
    with col:
        st.markdown(
            f"""
        <div class="kpi-card" style="--accent:{color}">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value" style="color:{color};font-size:20px">{value}</div>
          <div class="kpi-delta">{delta}</div>
        </div>""",
            unsafe_allow_html=True,
        )

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── MAIN TABS ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["PORTFOLIO", "MARKET", "BACKTEST", "CRISIS TEST", "AI ANALYST"]
)


# ══════════════════════════════════════════════════════════════════════
# TAB 1 — PORTFOLIO
# ══════════════════════════════════════════════════════════════════════
with tab1:
    left, mid, right = st.columns([1, 1.2, 1])

    with left:
        st.markdown(
            '<div class="section-title">ALLOCATION</div>', unsafe_allow_html=True
        )
        alloc_html = ""
        for ticker, w in list(weights.items())[:18]:
            bar_w = int(w * 300)
            color = "#00e5ff" if w > 0.10 else "#1a4a5a" if w > 0.02 else "#0d1a20"
            alloc_html += f"""
            <div class="alloc-row">
              <span class="alloc-ticker">{ticker}</span>
              <div class="alloc-bar-bg"><div class="alloc-bar" style="width:{bar_w}px;background:{color}"></div></div>
              <span class="alloc-pct" style="color:{color}">{w * 100:.1f}%</span>
            </div>"""
        st.markdown(alloc_html, unsafe_allow_html=True)

    with mid:
        st.markdown(
            '<div class="section-title">WEIGHT DISTRIBUTION</div>',
            unsafe_allow_html=True,
        )
        if weights:
            top = dict(list(weights.items())[:12])
            colors = [
                "#00e5ff" if v > 0.10 else "#1a4a5a" if v > 0.02 else "#0a1520"
                for v in top.values()
            ]
            fig_pie = go.Figure(
                go.Pie(
                    labels=list(top.keys()),
                    values=[v * 100 for v in top.values()],
                    hole=0.6,
                    marker=dict(colors=colors, line=dict(color="#050508", width=2)),
                    textinfo="label+percent",
                    textfont=dict(size=9, color="#5a6480"),
                    hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
                )
            )
            fig_pie.update_layout(
                **CHART_LAYOUT,
                height=260,
                showlegend=False,
                annotations=[
                    dict(
                        text=f"<b style='color:#00e5ff'>{regime}</b>",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(color="#00e5ff", size=12),
                    )
                ],
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Regime info
        blend = allocation.get("markowitz_blend", 0.5)
        st.markdown(
            f"""
        <div style="background:#08080f;border:1px solid #1a1a2e;padding:10px 14px;border-radius:2px;margin-top:8px">
          <div style="font-size:9px;letter-spacing:3px;color:#2a3050;margin-bottom:8px">REGIME STRATEGY</div>
          <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:6px">
            <span style="color:#5a6480">Markowitz weight</span>
            <span style="color:#00e5ff">{blend:.0%}</span>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:6px">
            <span style="color:#5a6480">RL agent weight</span>
            <span style="color:#7b61ff">{1 - blend:.0%}</span>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:11px">
            <span style="color:#5a6480">Optimizer Sharpe</span>
            <span style="color:#00ff9d">{sharpe_mkt:.3f}</span>
          </div>
        </div>""",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            '<div class="section-title">TFT SIGNALS</div>', unsafe_allow_html=True
        )
        if predictions:
            sorted_preds = sorted(
                predictions.items(), key=lambda x: x[1]["pred_5d"], reverse=True
            )
            for ticker, pred in sorted_preds[:15]:
                p5 = pred["pred_5d"]
                sig = pred["signal"]
                color = (
                    "#00ff9d"
                    if sig == "BUY"
                    else "#ff4444"
                    if sig == "SELL"
                    else "#ffd700"
                )
                st.markdown(
                    f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                padding:4px 8px;border-bottom:1px solid #0d0d18;font-size:11px">
                  <span style="color:#5a6480;width:55px">{ticker}</span>
                  <span style="color:{color};font-size:9px;letter-spacing:1px">{sig}</span>
                  <span style="color:{color}">{p5:+.3f}%</span>
                </div>""",
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — MARKET
# ══════════════════════════════════════════════════════════════════════
with tab2:
    # Price ticker
    if live_prices:
        price_html = (
            '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:12px">'
        )
        for ticker, data in list(live_prices.items())[:20]:
            pct = data.get("change_pct", 0)
            price = data.get("price", 0)
            color = "#00ff9d" if pct >= 0 else "#ff4444"
            price_html += f"""
            <div style="background:#08080f;border:1px solid #1a1a2e;padding:6px 10px;border-radius:2px;min-width:90px">
              <div style="font-size:9px;color:#4a5568;letter-spacing:1px">{ticker}</div>
              <div style="font-size:13px;color:#b8bcc8">${price:,.2f}</div>
              <div style="font-size:10px;color:{color}">{pct:+.2f}%</div>
            </div>"""
        price_html += "</div>"
        st.markdown(price_html, unsafe_allow_html=True)
    else:
        st.info("⚡ Run `python data_pipeline/streaming/producers.py` for live prices")

    col1, col2 = st.columns([2, 1])

    with col1:
        selected = st.multiselect(
            "SELECT TICKERS",
            options=TICKERS[:15] + [BENCHMARK_TICKER],
            default=["AAPL", "MSFT", "GOOGL", "SPY"],
        )
        days = st.select_slider("PERIOD", options=[30, 90, 180, 365], value=180)

        if selected:
            fig = go.Figure()
            for ticker in selected:
                hist = load_history(ticker, days)
                if not hist.empty:
                    close_col = "Close" if "Close" in hist.columns else "close"
                    date_col = "Date" if "Date" in hist.columns else "date"
                    norm = hist[close_col] / hist[close_col].iloc[0] * 100
                    color_map = {
                        "AAPL": "#00e5ff",
                        "MSFT": "#7b61ff",
                        "GOOGL": "#00ff9d",
                        "AMZN": "#ff6b35",
                        "JPM": "#ffd700",
                        "SPY": "#888888",
                    }
                    fig.add_trace(
                        go.Scatter(
                            x=hist[date_col],
                            y=norm,
                            name=ticker,
                            line=dict(
                                color=color_map.get(ticker, "#5a6480"), width=1.5
                            ),
                            hovertemplate=f"{ticker}: %{{y:.1f}}<extra></extra>",
                        )
                    )
            fig.add_hline(y=100, line_dash="dash", line_color="#1a1a2e", opacity=0.5)
            fig.update_layout(
                **CHART_LAYOUT, height=320, yaxis_title="Normalised (base=100)"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(
            '<div class="section-title">LIVE FEED</div>', unsafe_allow_html=True
        )
        for ticker, data in list(live_prices.items())[:20]:
            pct = data.get("change_pct", 0)
            price = data.get("price", 0)
            color = "#00ff9d" if pct >= 0 else "#ff4444"
            arrow = "▲" if pct >= 0 else "▼"
            st.markdown(
                f"""
            <div class="price-row">
              <span class="price-ticker">{ticker}</span>
              <span class="price-val">${price:,.2f}</span>
              <span style="color:{color};font-size:10px">{arrow} {abs(pct):.2f}%</span>
            </div>""",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — BACKTEST
# ══════════════════════════════════════════════════════════════════════
with tab3:
    if metrics_list and len(metrics_list) > 1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                '<div class="section-title">STRATEGY COMPARISON</div>',
                unsafe_allow_html=True,
            )
            df_m = pd.DataFrame(metrics_list)
            cols_show = [
                "name",
                "total_return",
                "annual_return",
                "sharpe_ratio",
                "sortino_ratio",
                "max_drawdown",
                "hit_rate",
            ]
            cols_avail = [c for c in cols_show if c in df_m.columns]
            df_show = df_m[cols_avail].copy()
            df_show.columns = [c.replace("_", " ").upper() for c in cols_avail]

            def highlight_af(row):
                if "AlphaFlow" in str(row.iloc[0]):
                    return ["color: #00e5ff; font-weight: bold"] * len(row)
                return ["color: #5a6480"] * len(row)

            styled = df_show.style.apply(highlight_af, axis=1)
            st.dataframe(styled, use_container_width=True, height=180)

        with col2:
            st.markdown(
                '<div class="section-title">SHARPE COMPARISON</div>',
                unsafe_allow_html=True,
            )
            names = [m.get("name", "") for m in metrics_list]
            sharpes = [m.get("sharpe_ratio", 0) for m in metrics_list]
            colors = ["#00e5ff" if "AlphaFlow" in n else "#1a1a2e" for n in names]
            fig_b = go.Figure(
                go.Bar(
                    x=names,
                    y=sharpes,
                    marker_color=colors,
                    text=[f"{s:.3f}" for s in sharpes],
                    textposition="outside",
                    textfont=dict(color="#5a6480", size=10),
                )
            )
            fig_b.add_hline(
                y=1.0,
                line_dash="dash",
                line_color="#ffd700",
                annotation_text="Sharpe=1.0",
                annotation_font_color="#ffd700",
            )
            fig_b.update_layout(**CHART_LAYOUT, height=220, showlegend=False)
            st.plotly_chart(fig_b, use_container_width=True)

        # Risk vs Return scatter
        st.markdown(
            '<div class="section-title">RISK / RETURN</div>', unsafe_allow_html=True
        )
        fig_s = go.Figure()
        for m in metrics_list:
            is_af = "AlphaFlow" in str(m.get("name", ""))
            fig_s.add_trace(
                go.Scatter(
                    x=[abs(m.get("max_drawdown", 0))],
                    y=[m.get("annual_return", 0)],
                    mode="markers+text",
                    name=m.get("name", ""),
                    text=[m.get("name", "")],
                    textposition="top center",
                    marker=dict(
                        size=18 if is_af else 10,
                        color="#00e5ff" if is_af else "#2a3050",
                        symbol="star" if is_af else "circle",
                        line=dict(color="#050508", width=1),
                    ),
                    showlegend=False,
                )
            )
        fig_s.update_layout(
            **CHART_LAYOUT,
            height=250,
            xaxis_title="Max Drawdown % (lower=better →)",
            yaxis_title="Annual Return %",
        )
        st.plotly_chart(fig_s, use_container_width=True)
    else:
        st.info(
            "Run `python models/forecasting/evaluate.py` to generate backtest results"
        )


# ══════════════════════════════════════════════════════════════════════
# TAB 4 — CRISIS STRESS TEST
# ══════════════════════════════════════════════════════════════════════
with tab4:
    if stress_data:
        # Summary metrics
        valid = {k: v for k, v in stress_data.items() if "error" not in v}
        beat_ct = sum(1 for v in valid.values() if v.get("beat_benchmark"))
        total_ct = len(valid)

        c1, c2, c3 = st.columns(3)
        c1.metric("Crisis Periods Tested", total_ct)
        c2.metric("Beat Benchmark", f"{beat_ct}/{total_ct}")
        c3.metric(
            "Average Alpha",
            f"{np.mean([v.get('alpha_vs_spy', 0) for v in valid.values()]):.1f}%",
        )

        st.divider()

        # Per-crisis cards
        cols = st.columns(3)
        for i, (crisis_name, result) in enumerate(valid.items()):
            with cols[i % 3]:
                strats = result.get("strategies", {})
                af = strats.get("AlphaFlow (Regime-Aware)", {})
                spy = strats.get("SPY Benchmark", {})
                alpha = result.get("alpha_vs_spy", 0)
                beat = result.get("beat_benchmark", False)
                color = result.get("color", "#00e5ff")

                st.markdown(
                    f"""
                <div style="background:#08080f;border:1px solid #1a1a2e;
                border-left:3px solid {color};padding:12px 14px;border-radius:2px;margin-bottom:8px">
                  <div style="font-size:11px;font-weight:600;color:{color};margin-bottom:6px">{crisis_name}</div>
                  <div style="font-size:9px;color:#2a3050;margin-bottom:8px">{result.get("period", "")}</div>
                  <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:10px">
                    <span style="color:#4a5568">AlphaFlow</span>
                    <span style="color:{"#00ff9d" if af.get("total_return", 0) > 0 else "#ff4444"};text-align:right">
                      {af.get("total_return", 0):+.1f}%</span>
                    <span style="color:#4a5568">SPY</span>
                    <span style="color:{"#00ff9d" if spy.get("total_return", 0) > 0 else "#ff4444"};text-align:right">
                      {spy.get("total_return", 0):+.1f}%</span>
                    <span style="color:#4a5568">Alpha</span>
                    <span style="color:{"#00ff9d" if alpha > 0 else "#ff4444"};text-align:right;font-weight:600">
                      {alpha:+.1f}%</span>
                    <span style="color:#4a5568">Max DD</span>
                    <span style="color:#ffd700;text-align:right">{af.get("max_drawdown", 0):.1f}%</span>
                    <span style="color:#4a5568">Sharpe</span>
                    <span style="color:#00e5ff;text-align:right">{af.get("sharpe", 0):.3f}</span>
                  </div>
                  <div style="margin-top:8px;font-size:9px;color:{"#00ff9d" if beat else "#ff4444"}">
                    {"✓ BEAT BENCHMARK" if beat else "✗ UNDERPERFORMED"}</div>
                </div>""",
                    unsafe_allow_html=True,
                )

        # Comparison chart
        st.markdown(
            '<div class="section-title">ALPHAFLOW vs SPY — ALL CRISES</div>',
            unsafe_allow_html=True,
        )
        crisis_names = list(valid.keys())
        af_returns = [
            valid[c]
            .get("strategies", {})
            .get("AlphaFlow (Regime-Aware)", {})
            .get("total_return", 0)
            for c in crisis_names
        ]
        spy_returns = [
            valid[c]
            .get("strategies", {})
            .get("SPY Benchmark", {})
            .get("total_return", 0)
            for c in crisis_names
        ]

        fig_crisis = go.Figure()
        fig_crisis.add_trace(
            go.Bar(
                name="AlphaFlow",
                x=crisis_names,
                y=af_returns,
                marker_color="#00e5ff",
                opacity=0.9,
            )
        )
        fig_crisis.add_trace(
            go.Bar(
                name="SPY",
                x=crisis_names,
                y=spy_returns,
                marker_color="#2a3050",
                opacity=0.9,
            )
        )
        fig_crisis.add_hline(y=0, line_color="#1a1a2e")
        fig_crisis.update_layout(
            **CHART_LAYOUT, height=300, barmode="group", yaxis_title="Total Return %"
        )
        st.plotly_chart(fig_crisis, use_container_width=True)

    else:
        st.info("Run crisis stress test first:")
        st.code("python models/backtesting/crisis_stress_test.py", language="bash")
        if st.button("▶ RUN STRESS TEST NOW"):
            with st.spinner("Running crisis stress test across 6 periods..."):
                import subprocess, sys  # noqa: E401

                result = subprocess.run(
                    [sys.executable, "models/backtesting/crisis_stress_test.py"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    st.success("✓ Stress test complete")
                    st.rerun()
                else:
                    st.error(result.stderr[-500:])


# ══════════════════════════════════════════════════════════════════════
# TAB 5 — AI ANALYST
# ══════════════════════════════════════════════════════════════════════
with tab5:
    col_chat, col_info = st.columns([2, 1])

    with col_info:
        # Ollama status
        ollama_ok = False
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=2)
            ollama_ok = r.status_code == 200
        except Exception:
            pass

        if ollama_ok:
            st.success("🟢 Ollama: Connected")
            try:
                models_data = requests.get(
                    "http://localhost:11434/api/tags", timeout=2
                ).json()
                model_names = [m["name"] for m in models_data.get("models", [])]
                if model_names:
                    sel_model = st.selectbox("Model", model_names)
                else:
                    sel_model = st.text_input("Model", value="llama3.2")
            except Exception:
                sel_model = st.text_input("Model", value="llama3.2")
        else:
            st.warning("🟡 Ollama offline")
            st.code("ollama serve", language="bash")
            sel_model = "llama3.2"

        api_key = st.text_input(
            "Claude API Key (fallback)",
            type="password",
            value=os.getenv("ANTHROPIC_API_KEY", ""),
        )

        st.divider()
        st.markdown(
            '<div class="section-title">QUICK ANALYSIS</div>', unsafe_allow_html=True
        )
        quick_qs = [
            "Why is SILVER overweight?",
            "BEAR regime — key risks?",
            "Best 3 positions now?",
            "Crypto allocation rationale",
            "Compare Sharpe to SPY",
            "What triggers rebalance?",
            "Weakest position right now?",
            "Stress test summary",
        ]
        for q in quick_qs:
            if st.button(q, use_container_width=True):
                st.session_state.pending_q = q

    with col_chat:
        st.markdown(
            '<div class="section-title">AI PORTFOLIO ANALYST — POWERED BY OLLAMA</div>',
            unsafe_allow_html=True,
        )

        if "analyst_msgs" not in st.session_state:
            st.session_state.analyst_msgs = [
                {
                    "role": "assistant",
                    "content": f"**AlphaFlow Terminal** — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
                    f"Regime: **{regime}** | Assets: **39** | Sharpe: **{af_metrics.get('sharpe_ratio', sharpe_mkt):.3f}**\n\n"
                    "I have full access to all system data — allocation, predictions, sentiment, regime, stress tests. Ask anything.",
                }
            ]

        for msg in st.session_state.analyst_msgs:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        pending = getattr(st.session_state, "pending_q", None)
        if pending:
            del st.session_state.pending_q
            user_input = pending
        else:
            user_input = st.chat_input("Ask the analyst...")

        if user_input:
            st.session_state.analyst_msgs.append(
                {"role": "user", "content": user_input}
            )
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                if api_key:
                    os.environ["ANTHROPIC_API_KEY"] = api_key
                sys.path.insert(0, ".")
                try:
                    from dashboard.analyst.llm_analyst import PortfolioAnalyst

                    analyst = PortfolioAnalyst(api_key=api_key, model=sel_model)
                    response = st.write_stream(analyst.ask(user_input))
                except Exception as e:
                    response = f"Analyst error: {e}\n\nEnsure `dashboard/analyst/llm_analyst.py` exists."
                    st.error(response)
            st.session_state.analyst_msgs.append(
                {"role": "assistant", "content": response}
            )
