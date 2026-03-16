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

import streamlit as st
import pandas as pd
import numpy as np  # noqa: F401
import plotly.graph_objects as go
import plotly.express as px  # noqa: F401
from plotly.subplots import make_subplots  # noqa: F401
import requests
import json
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

# ── PAGE CONFIG ───────────────────────────────────────────────────────
st.set_page_config(
    page_title = "AlphaFlow — Portfolio Optimizer",
    page_icon  = "📈",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── STYLING ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #070a0f; }
    .stApp { background-color: #070a0f; color: #c9d1d9; }
    
    .metric-card {
        background: #0d1117;
        border: 1px solid #1e2d40;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 4px 0;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #00e5ff;
        font-family: 'Courier New', monospace;
    }
    
    .metric-label {
        font-size: 11px;
        color: #4a5568;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 4px;
    }
    
    .signal-buy  { color: #00ff9d; font-weight: 700; }
    .signal-sell { color: #ff4444; font-weight: 700; }
    .signal-hold { color: #ffd700; font-weight: 700; }
    
    .regime-card {
        background: linear-gradient(135deg, #0d1117, #1a2332);
        border: 1px solid #00e5ff;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    
    div[data-testid="stMetricValue"] { color: #00e5ff !important; }
    
    .stSelectbox label { color: #c9d1d9 !important; }
    
    h1, h2, h3 { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# ── CONFIG ────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

# Colors
COLORS = {
    "primary"  : "#00e5ff",
    "secondary": "#7b61ff",
    "success"  : "#00ff9d",
    "danger"   : "#ff4444",
    "warning"  : "#ffd700",
    "muted"    : "#4a5568",
    "bg"       : "#070a0f",
    "panel"    : "#0d1117",
}

TICKER_COLORS = {
    "AAPL": "#00e5ff", "MSFT": "#7b61ff", "GOOGL": "#00ff9d",
    "AMZN": "#ff6b35", "JPM" : "#ffd700", "JNJ" : "#ff4db8",
    "XOM" : "#4ecdc4", "BRK-B": "#a8e6cf", "TSLA": "#ff4444",
    "SPY" : "#888888",
}


# ── DATA LOADING ──────────────────────────────────────────────────────

@st.cache_data(ttl=300)  # cache for 5 minutes
def load_allocation():
    try:
        r = requests.get(f"{API_BASE}/api/portfolio/allocation", timeout=5)
        return r.json()
    except Exception:
        # Fallback: load directly from file
        try:
            with open("reports/allocation.json") as f:
                return json.load(f)
        except Exception:
            return None

@st.cache_data(ttl=300)
def load_metrics():
    try:
        r = requests.get(f"{API_BASE}/api/portfolio/metrics", timeout=5)
        return r.json()["strategies"]
    except Exception:
        try:
            with open("reports/metrics.json") as f:
                return json.load(f)
        except Exception:
            return []

@st.cache_data(ttl=60)  # refresh prices every minute
def load_prices():
    try:
        r = requests.get(f"{API_BASE}/api/market/prices", timeout=10)
        return r.json()["prices"]
    except Exception:
        return []

@st.cache_data(ttl=3600)  # predictions change daily
def load_predictions():
    try:
        r = requests.get(f"{API_BASE}/api/market/predictions", timeout=5)
        return r.json()["predictions"]
    except Exception:
        # Load from file directly
        try:
            pred_path = Path("data/local/predictions.parquet")
            if pred_path.exists():
                df = pd.read_parquet(pred_path)
                ticker_map = {
                    "0": "AAPL", "1": "MSFT", "2": "GOOGL", "3": "AMZN",
                    "4": "JPM",  "5": "JNJ",  "6": "SPY",   "7": "BRK-B",
                    "8": "TSLA", "9": "XOM",
                }
                latest = df.sort_values("date").groupby("ticker").last().reset_index()
                results = []
                for _, row in latest.iterrows():
                    ticker   = ticker_map.get(str(row["ticker"]), str(row["ticker"]))
                    pred_p50 = float(row["pred_p50"])
                    signal   = "BUY" if pred_p50 > 0.001 else "SELL" if pred_p50 < -0.001 else "HOLD"
                    results.append({
                        "ticker"    : ticker,
                        "pred_p50"  : round(pred_p50 * 100, 3),
                        "pred_p10"  : round(float(row.get("pred_p10", 0)) * 100, 3),
                        "pred_p90"  : round(float(row.get("pred_p90", 0)) * 100, 3),
                        "signal"    : signal,
                        "confidence": 65.0,
                    })
                return results
        except Exception:
            return []

@st.cache_data(ttl=3600)
def load_drift():
    try:
        r = requests.get(f"{API_BASE}/api/drift/report", timeout=5)
        return r.json()
    except Exception:
        try:
            with open("reports/drift_report.json") as f:
                return json.load(f)
        except Exception:
            return None

@st.cache_data(ttl=60)
def load_history(ticker: str, days: int = 180):
    try:
        r = requests.get(f"{API_BASE}/api/market/history/{ticker}?days={days}", timeout=10)
        return pd.DataFrame(r.json()["data"])
    except Exception:
        import yfinance as yf
        from datetime import timedelta
        df = yf.download(ticker,
                         start=(datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d"),
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        return df


# ── CHART HELPERS ─────────────────────────────────────────────────────

def dark_chart(fig):
    """Apply dark theme to any plotly figure."""
    fig.update_layout(
        paper_bgcolor = COLORS["bg"],
        plot_bgcolor  = COLORS["panel"],
        font          = dict(color="#c9d1d9", family="Courier New"),
        margin        = dict(l=20, r=20, t=40, b=20),
    )
    fig.update_xaxes(gridcolor="#1e2d40", zerolinecolor="#1e2d40")
    fig.update_yaxes(gridcolor="#1e2d40", zerolinecolor="#1e2d40")
    return fig


# ── SIDEBAR ───────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🌊 AlphaFlow")
    st.markdown("*Adaptive Portfolio Optimizer*")
    st.divider()

    st.markdown("### Navigation")
    page = st.radio("", [
        "📊 Portfolio Overview",
        "📈 Market & Predictions",
        "🔬 Backtest Analysis",
        "🛰️ System Health",
    ], label_visibility="collapsed")

    st.divider()
    st.markdown("### Settings")
    auto_refresh = st.toggle("Auto refresh (60s)", value=False)
    show_benchmark = st.toggle("Show SPY benchmark", value=True)

    st.divider()
    st.markdown(f"*Last updated: {datetime.utcnow().strftime('%H:%M:%S')} UTC*")

    if auto_refresh:
        import time
        time.sleep(60)
        st.cache_data.clear()
        st.rerun()


# ════════════════════════════════════════════════════════════════════════
# PAGE: PORTFOLIO OVERVIEW
# ════════════════════════════════════════════════════════════════════════

if page == "📊 Portfolio Overview":
    st.title("📊 Portfolio Overview")
    st.markdown("*AI-driven allocation powered by Temporal Fusion Transformer + Markowitz Optimizer*")

    allocation = load_allocation()
    metrics    = load_metrics()
    prices     = load_prices()

    # ── TOP METRICS ROW ───────────────────────────────────────────────
    if metrics:
        alphaflow = next((m for m in metrics if "AlphaFlow" in m.get("name", "")), {})
        spy_metrics = next((m for m in metrics if "Buy Hold" in m.get("name", "")), {})

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Return", f"{alphaflow.get('total_return', 0):.1f}%",
                      delta=f"{alphaflow.get('total_return', 0) - spy_metrics.get('total_return', 0):.1f}% vs SPY")
        with col2:
            st.metric("Sharpe Ratio", f"{alphaflow.get('sharpe_ratio', 0):.3f}",
                      delta=f"{alphaflow.get('sharpe_ratio', 0) - spy_metrics.get('sharpe_ratio', 0):.3f} vs SPY")
        with col3:
            st.metric("Max Drawdown", f"{alphaflow.get('max_drawdown', 0):.1f}%")
        with col4:
            st.metric("Hit Rate", f"{alphaflow.get('hit_rate', 0):.1f}%")
        with col5:
            if allocation:
                st.metric("Regime", allocation.get("regime_label", "neutral").upper())

    st.divider()

    # ── ALLOCATION CHARTS ─────────────────────────────────────────────
    if allocation and allocation.get("weights"):
        weights = allocation["weights"]
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Current Allocation")

            # Pie chart
            tickers = list(weights.keys())
            values  = [weights[t] * 100 for t in tickers]
            colors  = [TICKER_COLORS.get(t, "#888") for t in tickers]

            fig_pie = go.Figure(go.Pie(
                labels     = tickers,
                values     = values,
                hole       = 0.5,
                marker     = dict(colors=colors, line=dict(color="#070a0f", width=2)),
                textinfo   = "label+percent",
                hovertemplate = "%{label}: %{value:.1f}%<extra></extra>",
            ))
            fig_pie.update_layout(
                showlegend    = False,
                annotations   = [dict(text="AlphaFlow", x=0.5, y=0.5,
                                      font_size=14, showarrow=False,
                                      font_color="#00e5ff")],
            )
            st.plotly_chart(dark_chart(fig_pie), use_container_width=True)

        with col2:
            st.subheader("Weight Distribution")

            # Horizontal bar chart
            sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            t_list = [t for t, _ in sorted_items]
            w_list = [w * 100 for _, w in sorted_items]
            c_list = [TICKER_COLORS.get(t, "#888") for t in t_list]

            fig_bar = go.Figure(go.Bar(
                y           = t_list,
                x           = w_list,
                orientation = "h",
                marker_color= c_list,
                text        = [f"{w:.1f}%" for w in w_list],
                textposition= "outside",
                hovertemplate = "%{y}: %{x:.1f}%<extra></extra>",
            ))
            fig_bar.update_layout(
                xaxis_title = "Weight (%)",
                height      = 350,
            )
            st.plotly_chart(dark_chart(fig_bar), use_container_width=True)

        # Allocation details
        st.subheader("Allocation Details")
        regime = allocation.get("regime_label", "neutral")
        regime_color = {"risk-on": "🟢", "neutral": "🟡", "risk-off": "🔴"}.get(regime, "🟡")

        col1, col2, col3 = st.columns(3)
        col1.metric("Macro Regime", f"{regime_color} {regime.upper()}")
        col2.metric("Markowitz Blend", f"{(1-allocation.get('rl_blend',0.5))*100:.0f}%")
        col3.metric("RL Blend", f"{allocation.get('rl_blend',0.5)*100:.0f}%")

    # ── LIVE PRICES ───────────────────────────────────────────────────
    if prices:
        st.divider()
        st.subheader("Live Market Prices")
        cols = st.columns(min(len(prices), 5))
        for i, price_data in enumerate(prices[:10]):
            col = cols[i % 5]
            delta_color = "normal" if price_data["change"] >= 0 else "inverse"
            with col:
                st.metric(
                    label = price_data["ticker"],
                    value = f"${price_data['price']:,.2f}",
                    delta = f"{price_data['change_pct']:+.2f}%",
                )


# ════════════════════════════════════════════════════════════════════════
# PAGE: MARKET & PREDICTIONS
# ════════════════════════════════════════════════════════════════════════

elif page == "📈 Market & Predictions":
    st.title("📈 Market & TFT Predictions")
    st.markdown("*5-day forward return predictions from the Temporal Fusion Transformer*")

    predictions = load_predictions()

    # ── SIGNAL TABLE ──────────────────────────────────────────────────
    if predictions:
        st.subheader("Model Signals")

        pred_df = pd.DataFrame(predictions)

        def color_signal(val):
            if val == "BUY":  return "color: #00ff9d; font-weight: bold"  # noqa: E701
            if val == "SELL": return "color: #ff4444; font-weight: bold"  # noqa: E701
            return "color: #ffd700; font-weight: bold"

        def color_return(val):
            try:
                v = float(val)
                return f"color: {'#00ff9d' if v > 0 else '#ff4444'}"
            except Exception:
                return ""

        if "pred_p50" in pred_df.columns:
            display_df = pred_df[["ticker", "signal", "pred_p10", "pred_p50", "pred_p90", "confidence"]].copy()
            display_df.columns = ["Ticker", "Signal", "Bear (p10%)", "Base (p50%)", "Bull (p90%)", "Confidence%"]

            styled = display_df.style\
                .applymap(color_signal, subset=["Signal"])\
                .applymap(color_return, subset=["Bear (p10%)", "Base (p50%)", "Bull (p90%)"])\
                .format({"Bear (p10%)": "{:.3f}", "Base (p50%)": "{:.3f}",
                         "Bull (p90%)": "{:.3f}", "Confidence%": "{:.1f}"})

            st.dataframe(styled, use_container_width=True, height=350)

        # Prediction bar chart
        st.subheader("5-Day Return Forecasts")
        if "pred_p50" in pred_df.columns:
            pred_df_sorted = pred_df.sort_values("pred_p50", ascending=False)
            colors = ["#00ff9d" if v > 0 else "#ff4444"
                      for v in pred_df_sorted["pred_p50"]]

            fig_pred = go.Figure()
            fig_pred.add_trace(go.Bar(
                x           = pred_df_sorted["ticker"],
                y           = pred_df_sorted["pred_p50"],
                name        = "p50 (median)",
                marker_color= colors,
                error_y     = dict(
                    type      = "data",
                    array     = (pred_df_sorted["pred_p90"] - pred_df_sorted["pred_p50"]).values,
                    arrayminus= (pred_df_sorted["pred_p50"] - pred_df_sorted["pred_p10"]).values,
                    visible   = True,
                    color     = "#4a5568",
                ),
            ))
            fig_pred.update_layout(
                yaxis_title = "Predicted 5-Day Return (%)",
                xaxis_title = "Ticker",
                height      = 350,
            )
            fig_pred.add_hline(y=0, line_dash="dash", line_color="#4a5568")
            st.plotly_chart(dark_chart(fig_pred), use_container_width=True)

    # ── PRICE CHART ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Price History")

    from config import TICKERS, BENCHMARK_TICKER
    all_tickers = TICKERS + [BENCHMARK_TICKER]

    selected = st.multiselect(
        "Select tickers",
        options  = all_tickers,
        default  = ["AAPL", "MSFT", "SPY"],
    )
    days = st.slider("Days", min_value=30, max_value=365, value=180, step=30)

    if selected:
        fig_price = go.Figure()
        for ticker in selected:
            try:
                hist = load_history(ticker, days)
                if not hist.empty:
                    close_col = "close" if "close" in hist.columns else "Close"
                    # Normalise to 100 for comparison
                    norm = hist[close_col] / hist[close_col].iloc[0] * 100
                    fig_price.add_trace(go.Scatter(
                        x    = hist["date"] if "date" in hist.columns else hist.index,
                        y    = norm,
                        name = ticker,
                        line = dict(color=TICKER_COLORS.get(ticker, "#888"), width=2),
                        hovertemplate = f"{ticker}: %{{y:.1f}}<extra></extra>",
                    ))
            except Exception as e:
                st.warning(f"Could not load {ticker}: {e}")

        fig_price.update_layout(
            yaxis_title = "Normalised Price (base=100)",
            height      = 400,
            legend      = dict(bgcolor="#0d1117", bordercolor="#1e2d40"),
        )
        fig_price.add_hline(y=100, line_dash="dash", line_color="#4a5568", opacity=0.5)
        st.plotly_chart(dark_chart(fig_price), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════
# PAGE: BACKTEST ANALYSIS
# ════════════════════════════════════════════════════════════════════════

elif page == "🔬 Backtest Analysis":
    st.title("🔬 Backtest Analysis")
    st.markdown("*Validation period: Jul 2024 – Mar 2025 | 9 tickers | 65 features*")

    metrics = load_metrics()

    if metrics:
        # ── METRICS TABLE ─────────────────────────────────────────────
        st.subheader("Performance Comparison")

        df = pd.DataFrame(metrics)
        df_display = df[["name", "total_return", "annual_return", "sharpe_ratio",
                          "sortino_ratio", "max_drawdown", "hit_rate"]].copy()
        df_display.columns = ["Strategy", "Total Return %", "Annual Return %",
                               "Sharpe", "Sortino", "Max Drawdown %", "Hit Rate %"]

        def highlight_alphaflow(row):
            if "AlphaFlow" in str(row["Strategy"]):
                return ["background-color: #0d2137; color: #00e5ff"] * len(row)
            return [""] * len(row)

        styled = df_display.style\
            .apply(highlight_alphaflow, axis=1)\
            .format({"Total Return %": "{:.1f}", "Annual Return %": "{:.1f}",
                     "Sharpe": "{:.3f}", "Sortino": "{:.3f}",
                     "Max Drawdown %": "{:.1f}", "Hit Rate %": "{:.1f}"})

        st.dataframe(styled, use_container_width=True)

        # ── CHARTS ────────────────────────────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Sharpe Ratio Comparison")
            bar_colors = ["#00e5ff" if "AlphaFlow" in n else "#333"
                          for n in df["name"]]
            fig_sharpe = go.Figure(go.Bar(
                x             = df["name"],
                y             = df["sharpe_ratio"],
                marker_color  = bar_colors,
                text          = [f"{s:.3f}" for s in df["sharpe_ratio"]],
                textposition  = "outside",
            ))
            fig_sharpe.update_layout(height=300)
            fig_sharpe.add_hline(y=1.0, line_dash="dash",
                                  line_color="#ffd700", opacity=0.7,
                                  annotation_text="Sharpe=1.0 threshold")
            st.plotly_chart(dark_chart(fig_sharpe), use_container_width=True)

        with col2:
            st.subheader("Risk vs Return")
            fig_scatter = go.Figure()
            for _, row in df.iterrows():
                is_af = "AlphaFlow" in str(row["name"])
                fig_scatter.add_trace(go.Scatter(
                    x    = [abs(row["max_drawdown"])],
                    y    = [row["annual_return"]],
                    mode = "markers+text",
                    name = row["name"],
                    text = [row["name"]],
                    textposition = "top center",
                    marker = dict(
                        size  = 20 if is_af else 12,
                        color = "#00e5ff" if is_af else "#555",
                        symbol= "star" if is_af else "circle",
                    ),
                ))
            fig_scatter.update_layout(
                xaxis_title = "Max Drawdown % (lower is better →)",
                yaxis_title = "Annual Return %",
                showlegend  = False,
                height      = 300,
            )
            st.plotly_chart(dark_chart(fig_scatter), use_container_width=True)

        # ── OPEN FULL REPORT ──────────────────────────────────────────
        st.divider()
        report_path = Path("reports/backtest_report.html")
        if report_path.exists():
            st.success("✓ Full interactive backtest report available")
            st.markdown("Run `open reports/backtest_report.html` in terminal to view")


# ════════════════════════════════════════════════════════════════════════
# PAGE: SYSTEM HEALTH
# ════════════════════════════════════════════════════════════════════════

elif page == "🛰️ System Health":
    st.title("🛰️ System Health")
    st.markdown("*Pipeline status, drift monitoring, MLOps overview*")

    # ── PIPELINE STATUS ───────────────────────────────────────────────
    st.subheader("Pipeline Components")

    components = {
        "Market Data"  : Path("data/local/raw/market"),
        "Predictions"  : Path("data/local/predictions.parquet"),
        "Allocation"   : Path("reports/allocation.json"),
        "Metrics"      : Path("reports/metrics.json"),
        "Drift Report" : Path("reports/drift_report.json"),
        "Model Checkpoint": Path("models/checkpoints"),
    }

    cols = st.columns(3)
    for i, (name, path) in enumerate(components.items()):
        exists = path.exists()
        if exists:
            age = (datetime.utcnow().timestamp() - path.stat().st_mtime) / 3600
            status = "✅ Fresh" if age < 48 else f"⚠️ {age:.0f}h old"
        else:
            status = "❌ Missing"
        with cols[i % 3]:
            st.metric(name, status)

    # ── DRIFT REPORT ──────────────────────────────────────────────────
    st.divider()
    st.subheader("Data Drift Monitor")
    drift = load_drift()

    if drift:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Drift Score",  f"{drift.get('drift_score', 0):.3f}")
        col2.metric("Drifted Features", f"{drift.get('n_drifted', 0)}/{drift.get('n_total', 0)}")
        col3.metric("Method", drift.get("method", "psi").upper())
        col4.metric("Action", "🔴 RETRAIN" if drift.get("drift_score", 0) > 0.3 else "🟢 MONITOR")

        # Drift gauge
        drift_score = drift.get("drift_score", 0)
        fig_gauge = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = drift_score,
            title = {"text": "Drift Score", "font": {"color": "#c9d1d9"}},
            gauge = {
                "axis"  : {"range": [0, 1], "tickcolor": "#c9d1d9"},
                "bar"   : {"color": "#00e5ff"},
                "steps" : [
                    {"range": [0, 0.1],  "color": "#0a2a0a"},
                    {"range": [0.1, 0.3],"color": "#2a2a0a"},
                    {"range": [0.3, 1.0],"color": "#2a0a0a"},
                ],
                "threshold": {
                    "line" : {"color": "#ffd700", "width": 4},
                    "thickness": 0.75,
                    "value": 0.3,
                },
            },
            number = {"font": {"color": "#00e5ff"}},
        ))
        fig_gauge.update_layout(height=250, paper_bgcolor=COLORS["bg"],
                                 font=dict(color="#c9d1d9"))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── TECH STACK ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Tech Stack")

    stack = {
        "Data Pipeline"   : "yfinance + SEC EDGAR + Airflow",
        "Feature Store"   : "AWS S3 (Parquet) — 65 features",
        "ML Model"        : "Temporal Fusion Transformer (PyTorch)",
        "Portfolio Opt"   : "Markowitz (cvxpy) + RL (PPO)",
        "MLOps"           : "MLflow + Evidently + GitHub Actions",
        "Infrastructure"  : "AWS S3 + EC2 t2.micro (free tier)",
        "Dashboard"       : "FastAPI + Streamlit",
    }

    for component, tech in stack.items():
        col1, col2 = st.columns([1, 3])
        col1.markdown(f"**{component}**")
        col2.markdown(f"`{tech}`")


if __name__ == "__main__":
    pass