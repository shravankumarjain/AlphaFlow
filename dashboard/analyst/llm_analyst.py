# dashboard/analyst/llm_analyst.py
#
# AlphaFlow — LLM Portfolio Analyst
# Powered by Ollama (local, free, runs offline) with Claude API fallback
#
# Setup Ollama (one time):
#   brew install ollama
#   ollama pull llama3.2          # 2GB, fast on M2
#   ollama pull mistral           # alternative
#   ollama serve                  # starts server at localhost:11434
#
# The analyst has full context:
#   - Current allocation weights
#   - HMM regime + confidence
#   - TFT predictions per ticker
#   - FinBERT sentiment scores
#   - Backtest metrics vs benchmarks
#   - Live streaming prices
#   - Drift report

import json
import logging
import os
import requests
from datetime import datetime
from pathlib import Path
from typing import Generator

import pandas as pd
import streamlit as st

import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import TICKERS, MULTI_ASSET_TICKERS, LOCAL_DATA_DIR

logger = logging.getLogger("llm_analyst")
REPORTS = Path("reports")
DATA_DIR = Path(LOCAL_DATA_DIR)

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")


# ── CONTEXT BUILDER ───────────────────────────────────────────────────


class PortfolioContextBuilder:
    def _load_json(self, path: Path) -> dict:
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return {}

    def load_all(self) -> dict:
        allocation = self._load_json(REPORTS / "allocation.json")
        metrics = self._load_json(REPORTS / "metrics.json") or []
        regime = self._load_json(DATA_DIR / "regime" / "current_regime.json")
        drift = self._load_json(REPORTS / "drift_report.json")
        live_prices = self._load_json(DATA_DIR / "streaming" / "live_prices.json")
        predictions = self._load_predictions()
        sentiment = self._load_sentiment()
        return dict(
            allocation=allocation,
            metrics=metrics,
            regime=regime,
            drift=drift,
            live_prices=live_prices,
            predictions=predictions,
            sentiment=sentiment,
        )

    def _load_predictions(self) -> dict:
        ticker_map = {str(i): t for i, t in enumerate(TICKERS + MULTI_ASSET_TICKERS)}
        try:
            df = pd.read_parquet(DATA_DIR / "predictions.parquet")
            latest = df.sort_values("date").groupby("ticker").last().reset_index()
            out = {}
            for _, row in latest.iterrows():
                t = ticker_map.get(str(row["ticker"]), str(row["ticker"]))
                p50 = float(row["pred_p50"])
                out[t] = {
                    "pred_5d_pct": round(p50 * 100, 3),
                    "signal": "BUY"
                    if p50 > 0.001
                    else "SELL"
                    if p50 < -0.001
                    else "HOLD",
                }
            return out
        except Exception:
            return {}

    def _load_sentiment(self) -> dict:
        try:
            df = pd.read_parquet(
                DATA_DIR / "sentiment" / "finbert_delta_features.parquet"
            )
            latest = df.sort_values("date").groupby("ticker").last().reset_index()
            return {
                row["ticker"]: {
                    "score": round(float(row.get("sentiment_score", 0)), 3),
                    "delta": round(float(row.get("sentiment_delta", 0)), 3),
                    "label": "POSITIVE"
                    if float(row.get("sentiment_score", 0)) > 0.1
                    else "NEGATIVE"
                    if float(row.get("sentiment_score", 0)) < -0.1
                    else "NEUTRAL",
                }
                for _, row in latest.iterrows()
            }
        except Exception:
            return {}

    def build_system_prompt(self) -> str:
        ctx = self.load_all()
        a = ctx["allocation"]
        r = ctx["regime"]
        d = ctx["drift"]
        m = ctx["metrics"]
        p = ctx["predictions"]
        s = ctx["sentiment"]
        lp = ctx.get("live_prices", {}).get("prices", {})

        weights_str = (
            "\n".join(
                f"  {t}: {w * 100:.1f}%"
                for t, w in list(a.get("weights", {}).items())[:15]
            )
            or "  unavailable"
        )

        af = next((x for x in m if "AlphaFlow" in str(x.get("name", ""))), {})
        bh = next((x for x in m if "Buy Hold" in str(x.get("name", ""))), {})
        metrics_str = (
            (
                f"  AlphaFlow: Sharpe={af.get('sharpe_ratio', 0):.3f}, "
                f"Return={af.get('total_return', 0):.1f}%, MaxDD={af.get('max_drawdown', 0):.1f}%\n"
                f"  Buy&Hold:  Sharpe={bh.get('sharpe_ratio', 0):.3f}, "
                f"Return={bh.get('total_return', 0):.1f}%"
            )
            if af
            else "  unavailable"
        )

        top_preds = sorted(p.items(), key=lambda x: x[1]["pred_5d_pct"], reverse=True)[
            :8
        ]
        preds_str = (
            "\n".join(
                f"  {t}: {v['pred_5d_pct']:+.3f}% ({v['signal']})" for t, v in top_preds
            )
            or "  unavailable"
        )

        notable_sent = {t: v for t, v in s.items() if abs(v["delta"]) > 0.05}
        sent_str = (
            "\n".join(
                f"  {t}: {v['label']} score={v['score']:+.3f} delta={v['delta']:+.3f}"
                for t, v in list(notable_sent.items())[:8]
            )
            or "  unavailable"
        )

        live_str = (
            "\n".join(
                f"  {t}: ${v.get('price', 0):.2f} ({v.get('change_pct', 0):+.2f}%)"
                for t, v in list(lp.items())[:10]
            )
            or "  unavailable (start producers.py)"
        )

        return f"""You are AlphaFlow — an institutional-grade AI portfolio analyst.
You have access to all real-time system data. Be specific, cite numbers, be concise.

═══ LIVE PORTFOLIO STATE ═══

REGIME (HMM):
  Current: {r.get("current_regime", "unknown").upper()} | Confidence: {r.get("confidence", 0):.0%}
  Blend: {r.get("blend", {}).get("markowitz", 0.5):.0%} Markowitz + {r.get("blend", {}).get("rl", 0.5):.0%} RL

ALLOCATION (top 15):
{weights_str}

PERFORMANCE vs BENCHMARK:
{metrics_str}

TFT PREDICTIONS (5-day):
{preds_str}

SENTIMENT (FinBERT deltas):
{sent_str}

LIVE PRICES:
{live_str}

DRIFT: score={d.get("drift_score", 0):.3f} | {d.get("n_drifted", 0)}/{d.get("n_total", 0)} features drifted
ASSETS: 39 total (25 equity + 4 crypto + 5 commodity + 5 bonds)
MODEL: TFT on 45,568 samples | Updated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}

Answer questions about allocation, risk, predictions, regime, and portfolio decisions.
Be direct. Cite specific numbers. Use bullet points."""


# ── LLM BACKENDS ─────────────────────────────────────────────────────


def _stream_ollama(system: str, question: str, model: str) -> Generator:
    """Stream from local Ollama server."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "stream": True,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": question},
                ],
                "options": {"temperature": 0.3, "num_predict": 1024},
            },
            stream=True,
            timeout=120,
        )
        for line in resp.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break
    except requests.exceptions.ConnectionError:
        yield "⚠️ Ollama not running. Start it with: `ollama serve`\n\nThen pull a model: `ollama pull llama3.2`"
    except Exception as e:
        yield f"Ollama error: {e}"


def _stream_claude(system: str, question: str, api_key: str) -> Generator:
    """Stream from Claude API as fallback."""
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": question}],
        ) as stream:
            for text in stream.text_stream:
                yield text
    except Exception as e:
        yield f"Claude API error: {e}"


def _mock_stream(question: str) -> Generator:
    """Demo mode — no API needed."""
    import time

    text = f"""**AlphaFlow Analysis** *(Demo — run `ollama serve` for live AI)*

Current regime: **BEAR** (100% confidence, 21/21 recent days)

Regarding: *{question}*

**Key signals:**
- JNJ 25% — lowest beta (0.32), maximum defensive protection in BEAR
- GOOGL 22% — TFT's highest predicted 5-day return among 39 assets
- SILVER 16% — inflation hedge, low correlation to equity drawdowns

**Regime impact:** 80% Markowitz weight means conservative, variance-minimizing allocation.
The model correctly reduces high-vol positions (TSLA, NVDA, BTC) to minimum floor.

**Risk:** Max drawdown -2.3% vs Buy&Hold -5.1% — portfolio protected.

*Start Ollama: `brew install ollama && ollama pull llama3.2 && ollama serve`*"""
    for char in text:
        yield char
        time.sleep(0.008)


class PortfolioAnalyst:
    def __init__(self, api_key: str = "", model: str = OLLAMA_MODEL):
        self.api_key = api_key or ANTHROPIC_KEY
        self.model = model
        self.ctx = PortfolioContextBuilder()

    def _ollama_available(self) -> bool:
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def ask(self, question: str) -> Generator:
        system = self.ctx.build_system_prompt()
        if self._ollama_available():
            yield from _stream_ollama(system, question, self.model)
        elif self.api_key:
            yield from _stream_claude(system, question, self.api_key)
        else:
            yield from _mock_stream(question)


# ── BLOOMBERG-STYLE TERMINAL UI ───────────────────────────────────────


def run_analyst_ui():
    st.set_page_config(
        page_title="AlphaFlow Terminal",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');
    html, body, .stApp { background: #0a0a0a !important; color: #e0e0e0; font-family: 'JetBrains Mono', monospace; }
    .stChatMessage { background: #111 !important; border: 1px solid #222; border-radius: 4px; }
    .stChatInput textarea { background: #111 !important; color: #00ff9d !important; font-family: 'JetBrains Mono', monospace; border: 1px solid #00ff9d !important; }
    .metric-box { background: #111; border: 1px solid #333; padding: 8px 12px; border-radius: 4px; font-size: 12px; }
    .regime-bear { color: #ff4444; font-weight: 600; }
    .regime-bull { color: #00ff9d; font-weight: 600; }
    h1, h2, h3 { color: #00e5ff !important; font-family: 'JetBrains Mono', monospace; }
    .stButton button { background: #111 !important; color: #00e5ff !important; border: 1px solid #00e5ff !important; font-family: 'JetBrains Mono', monospace; font-size: 11px; }
    .stButton button:hover { background: #00e5ff20 !important; }
    div[data-testid="stMetricValue"] { color: #00e5ff !important; font-family: 'JetBrains Mono'; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # ── HEADER BAR ────────────────────────────────────────────────────
    ctx_builder = PortfolioContextBuilder()
    ctx = ctx_builder.load_all()
    regime = ctx["regime"]
    allocation = ctx["allocation"]
    metrics = ctx["metrics"]
    live_prices = ctx.get("live_prices", {}).get("prices", {})
    af_metrics = next((m for m in metrics if "AlphaFlow" in str(m.get("name", ""))), {})

    r_label = regime.get("current_regime", "UNKNOWN").upper()
    r_color = "🔴" if r_label == "BEAR" else "🟢" if r_label == "BULL" else "🟡"

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("⚡ ALPHAFLOW", "LIVE", delta="v2.0")
    col2.metric(
        "REGIME",
        f"{r_color} {r_label}",
        delta=f"{regime.get('confidence', 0):.0%} conf",
    )
    col3.metric(
        "SHARPE", f"{af_metrics.get('sharpe_ratio', 0):.3f}", delta="vs benchmark"
    )
    col4.metric("MAX DD", f"{af_metrics.get('max_drawdown', 0):.1f}%")
    col5.metric("HIT RATE", f"{af_metrics.get('hit_rate', 0):.1f}%")
    col6.metric("ASSETS", "39", delta="equity+crypto+bonds")

    st.divider()

    # ── MAIN LAYOUT ───────────────────────────────────────────────────
    left, right = st.columns([1, 2])

    with left:
        # Live prices ticker
        st.markdown("#### 📡 LIVE PRICES")
        if live_prices:
            for ticker, data in list(live_prices.items())[:15]:
                pct = data.get("change_pct", 0)
                color = "#00ff9d" if pct >= 0 else "#ff4444"
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:3px 8px;border-bottom:1px solid #1a1a1a;font-size:12px">'
                    f'<span style="color:#aaa">{ticker}</span>'
                    f'<span style="color:{color}">{pct:+.2f}%</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("⚠ Start producers.py for live prices")

        st.divider()

        # Portfolio allocation
        st.markdown("#### 💼 ALLOCATION")
        weights = allocation.get("weights", {})
        for ticker, w in list(weights.items())[:12]:
            bar_w = int(w * 200)
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;'
                f'padding:2px 0;font-size:12px">'
                f'<span style="color:#888;width:60px">{ticker}</span>'
                f'<div style="background:#00e5ff;height:4px;width:{bar_w}px;border-radius:2px"></div>'
                f'<span style="color:#00e5ff">{w * 100:.1f}%</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

        st.divider()

        # Quick questions
        st.markdown("#### ⚡ QUICK ANALYSIS")
        questions = [
            "Why is SILVER overweight?",
            "BEAR regime — what to do?",
            "Top TFT predictions today",
            "Risk analysis — weak spots?",
            "Crypto allocation rationale",
            "When will we rebalance?",
        ]
        for q in questions:
            if st.button(q, use_container_width=True):
                st.session_state.pending_q = q

        st.divider()

        # Ollama setup status
        analyst_temp = PortfolioAnalyst()
        if analyst_temp._ollama_available():
            st.success("🟢 Ollama: Connected")
            model_choice = st.selectbox(
                "Model", ["llama3.2", "mistral", "llama3.1", "phi3"]
            )
        else:
            st.warning("🟡 Ollama: Not running")
            st.code("ollama serve", language="bash")
            model_choice = "llama3.2"
            api_key = st.text_input(
                "Claude API Key (fallback)",
                type="password",
                value=os.getenv("ANTHROPIC_API_KEY", ""),
            )
            if api_key:
                os.environ["ANTHROPIC_API_KEY"] = api_key

    with right:
        st.markdown("#### 🧠 AI PORTFOLIO ANALYST")
        st.caption("Ask anything — powered by Ollama (local) with full system context")

        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": f"**AlphaFlow Terminal Online** — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
                    f"Regime: **{r_label}** ({regime.get('confidence', 0):.0%} confidence)\n"
                    f"Assets: **39** across equity, crypto, bonds, commodities\n"
                    f"Model: **TFT** — Sharpe {af_metrics.get('sharpe_ratio', 0):.3f}\n\n"
                    f"I have full access to all system data. Ask me anything about the portfolio.",
                }
            ]

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Handle pending quick question
        pending = getattr(st.session_state, "pending_q", None)
        if pending:
            del st.session_state.pending_q
            user_input = pending
        else:
            user_input = st.chat_input("Ask the analyst...")

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                analyst = PortfolioAnalyst(model=model_choice)
                response = st.write_stream(analyst.ask(user_input))
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    run_analyst_ui()
