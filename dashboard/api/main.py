# dashboard/api/main.py
#
# AlphaFlow — FastAPI Backend
#
# Endpoints:
#   GET  /                          — health check
#   GET  /api/portfolio/allocation  — current portfolio weights
#   GET  /api/portfolio/metrics     — backtest performance metrics
#   GET  /api/market/prices         — latest prices for all tickers
#   GET  /api/market/predictions    — TFT model predictions
#   GET  /api/drift/report          — latest drift report
#   GET  /api/pipeline/status       — pipeline health status
#
# Run locally:
#   uvicorn dashboard.api.main:app --reload --port 8000
#   open http://localhost:8000/docs  (auto-generated Swagger UI)

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional  # noqa: F401
import sys

import boto3
import pandas as pd
import numpy as np  # noqa: F401
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import AWS_REGION, S3_BUCKET, TICKERS, BENCHMARK_TICKER  # noqa: F401

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# ── APP SETUP ─────────────────────────────────────────────────────────
app = FastAPI(
    title       = "AlphaFlow API",
    description = "Adaptive Portfolio Optimizer — powered by TFT + RL",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

s3 = boto3.client("s3", region_name=AWS_REGION)
REPORTS_DIR = Path("reports")
DATA_DIR    = Path("data/local")


# ── RESPONSE MODELS ───────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status    : str
    timestamp : str
    version   : str

class AllocationResponse(BaseModel):
    timestamp      : str
    regime         : str
    regime_label   : str
    weights        : dict
    markowitz_sharpe: float
    rl_blend       : float

class MetricsResponse(BaseModel):
    strategies: list

class PriceData(BaseModel):
    ticker  : str
    price   : float
    change  : float
    change_pct: float

class PredictionData(BaseModel):
    ticker  : str
    pred_p10: float
    pred_p50: float
    pred_p90: float
    signal  : str
    confidence: float

class DriftResponse(BaseModel):
    drift_score  : float
    n_drifted    : int
    n_total      : int
    action       : str
    last_trained : str
    computed_at  : str


# ── HELPERS ───────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{path.name} not found — run pipeline first")
    with open(path) as f:
        return json.load(f)

def get_cache_age_hours(path: Path) -> float:
    if not path.exists():
        return 999
    return (datetime.utcnow().timestamp() - path.stat().st_mtime) / 3600


# ── ROUTES ────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status"   : "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version"  : "1.0.0",
    }


@app.get("/api/portfolio/allocation", response_model=AllocationResponse)
async def get_allocation():
    """
    Returns current portfolio allocation weights.
    Source: reports/allocation.json (updated daily by pipeline)
    """
    data = load_json(REPORTS_DIR / "allocation.json")
    age  = get_cache_age_hours(REPORTS_DIR / "allocation.json")  # noqa: F841

    return {
        "timestamp"       : data.get("timestamp", datetime.utcnow().isoformat()),
        "regime"          : str(data.get("macro_regime", 1)),
        "regime_label"    : data.get("regime_label", "neutral"),
        "weights"         : data.get("weights", {}),
        "markowitz_sharpe": data.get("markowitz_sharpe", 0),
        "rl_blend"        : data.get("rl_blend", 0.5),
    }


@app.get("/api/portfolio/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Returns backtest performance metrics for all strategies.
    Source: reports/metrics.json
    """
    data = load_json(REPORTS_DIR / "metrics.json")
    return {"strategies": data}


@app.get("/api/market/prices")
async def get_prices():
    """
    Returns latest market prices and daily changes for all tickers.
    Fetches live from yfinance — always current.
    """
    results = []
    all_tickers = TICKERS + [BENCHMARK_TICKER]

    try:
        data = yf.download(
            all_tickers,
            period   = "2d",
            interval = "1d",
            auto_adjust = True,
            progress = False,
        )

        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]
        else:
            close = data[["Close"]]

        for ticker in all_tickers:
            try:
                if ticker in close.columns:
                    prices     = close[ticker].dropna()
                    if len(prices) >= 2:
                        price      = float(prices.iloc[-1])
                        prev_price = float(prices.iloc[-2])
                        change     = price - prev_price
                        change_pct = (change / prev_price) * 100
                    else:
                        price = float(prices.iloc[-1]) if len(prices) > 0 else 0
                        change = 0
                        change_pct = 0

                    results.append({
                        "ticker"    : ticker,
                        "price"     : round(price, 2),
                        "change"    : round(change, 2),
                        "change_pct": round(change_pct, 3),
                        "is_benchmark": ticker == BENCHMARK_TICKER,
                    })
            except Exception as e:
                logger.warning(f"Price fetch failed for {ticker}: {e}")

    except Exception as e:
        logger.error(f"Bulk price fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {"prices": results, "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/market/predictions")
async def get_predictions():
    """
    Returns latest TFT model predictions per ticker.
    Source: data/local/predictions.parquet
    """
    pred_path = DATA_DIR / "predictions.parquet"
    if not pred_path.exists():
        raise HTTPException(status_code=404, detail="No predictions found — run evaluate.py first")

    # Ticker encoding map
    ticker_map = {
        "0": "AAPL", "1": "MSFT", "2": "GOOGL", "3": "AMZN",
        "4": "JPM",  "5": "JNJ",  "6": "SPY",   "7": "BRK-B",
        "8": "TSLA", "9": "XOM",
    }

    df = pd.read_parquet(pred_path)
    # Get most recent prediction per ticker
    latest = df.sort_values("date").groupby("ticker").last().reset_index()

    results = []
    for _, row in latest.iterrows():
        ticker    = ticker_map.get(str(row["ticker"]), str(row["ticker"]))
        pred_p50  = float(row["pred_p50"])
        uncertainty = float(row.get("uncertainty", 0.05))

        # Generate signal
        if pred_p50 > 0.001:
            signal = "BUY"
        elif pred_p50 < -0.001:
            signal = "SELL"
        else:
            signal = "HOLD"

        confidence = min(abs(pred_p50) / (uncertainty + 1e-6) * 100, 100)

        results.append({
            "ticker"    : ticker,
            "pred_p10"  : round(float(row.get("pred_p10", 0)) * 100, 3),
            "pred_p50"  : round(pred_p50 * 100, 3),
            "pred_p90"  : round(float(row.get("pred_p90", 0)) * 100, 3),
            "signal"    : signal,
            "confidence": round(confidence, 1),
            "date"      : str(row["date"]),
        })

    return {"predictions": results, "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/drift/report", response_model=DriftResponse)
async def get_drift_report():
    """Returns latest model drift monitoring report."""
    data = load_json(REPORTS_DIR / "drift_report.json")
    return {
        "drift_score" : data.get("drift_score", 0),
        "n_drifted"   : data.get("n_drifted", 0),
        "n_total"     : data.get("n_total", 0),
        "action"      : "RETRAIN" if data.get("drift_score", 0) > 0.3 else "monitor",
        "last_trained": data.get("last_trained", "unknown"),
        "computed_at" : data.get("computed_at", "unknown"),
    }


@app.get("/api/pipeline/status")
async def get_pipeline_status():
    """Returns health status of all pipeline components."""
    files = {
        "market_data"  : DATA_DIR / "raw/market",
        "predictions"  : DATA_DIR / "predictions.parquet",
        "allocation"   : REPORTS_DIR / "allocation.json",
        "metrics"      : REPORTS_DIR / "metrics.json",
        "drift_report" : REPORTS_DIR / "drift_report.json",
    }

    status = {}
    for name, path in files.items():
        exists = path.exists()
        age    = get_cache_age_hours(path) if exists else 999
        status[name] = {
            "exists"    : exists,
            "age_hours" : round(age, 1),
            "fresh"     : age < 48,
            "status"    : "✓" if exists and age < 48 else "⚠" if exists else "✗",
        }

    overall = "healthy" if all(v["exists"] for v in status.values()) else "degraded"
    return {"overall": overall, "components": status, "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/market/history/{ticker}")
async def get_ticker_history(ticker: str, days: int = 180):
    """Returns historical price data for a ticker."""
    ticker = ticker.upper()
    if ticker not in TICKERS + [BENCHMARK_TICKER]:
        raise HTTPException(status_code=400, detail=f"Unknown ticker: {ticker}")

    try:
        df = yf.download(
            ticker,
            start    = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d"),
            end      = datetime.utcnow().strftime("%Y-%m-%d"),
            auto_adjust = True,
            progress = False,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]

        return {
            "ticker": ticker,
            "data"  : [
                {
                    "date"  : str(row["date"])[:10],
                    "open"  : round(float(row["open"]), 2),
                    "high"  : round(float(row["high"]), 2),
                    "low"   : round(float(row["low"]), 2),
                    "close" : round(float(row["close"]), 2),
                    "volume": int(row["volume"]),
                }
                for _, row in df.iterrows()
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dashboard.api.main:app", host="0.0.0.0", port=8000, reload=True)