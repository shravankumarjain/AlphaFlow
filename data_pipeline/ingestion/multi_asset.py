"""
data_pipeline/ingestion/multi_asset.py
AlphaFlow Phase 9 — Multi-Asset Data Ingestion
Crypto (Binance), Bonds (FRED), Metals/Commodities (yfinance)
All normalized into the same feature format as equities.
$0 cost — all free APIs.
"""

import os, logging, time, requests  # noqa: E401
import pandas as pd
import numpy as np
import yfinance as yf
import boto3
from io import BytesIO
from pathlib import Path  # noqa: F401
from datetime import datetime, timedelta  # noqa: F401
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(module)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("multi_asset")

S3_BUCKET  = os.getenv("S3_BUCKET", "alphaflow-data-lake-291572330987")
AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")
FRED_API_KEY = os.getenv("FRED_API_KEY", "e029163e792aea81c8587370c2d25626")  # free at fred.stlouisfed.org

# ── Asset universes ───────────────────────────────────────────────────────────

CRYPTO_ASSETS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "BNB": "BNBUSDT",
}

COMMODITY_TICKERS = {
    "GOLD" : "GC=F",    # Gold futures
    "OIL"  : "CL=F",    # Crude oil futures
    "SILVER": "SI=F",   # Silver futures
    "NATGAS": "NG=F",   # Natural gas
    "WHEAT" : "ZW=F",   # Wheat futures
}

BOND_ETF_TICKERS = {
    "US10Y_ETF" : "IEF",   # iShares 7-10 Year Treasury
    "US30Y_ETF" : "TLT",   # iShares 20+ Year Treasury
    "TIPS_ETF"  : "TIP",   # iShares TIPS Bond
    "HY_ETF"    : "HYG",   # High Yield Corporate Bond
    "IG_ETF"    : "LQD",   # Investment Grade Corporate Bond
}

FRED_SERIES = {
    "fed_funds_rate" : "FEDFUNDS",
    "cpi_yoy"        : "CPIAUCSL",
    "unemployment"   : "UNRATE",
    "gdp_growth"     : "A191RL1Q225SBEA",
    "yield_10y"      : "GS10",
    "yield_2y"       : "GS2",
    "yield_spread"   : "T10Y2Y",
    "vix"            : "VIXCLS",
}


# ── Crypto ingestion via Binance public API ───────────────────────────────────

def fetch_binance_klines(symbol: str, interval: str = "1d",
                          start_date: str = "2020-01-01") -> pd.DataFrame:
    """
    Fetch OHLCV from Binance public API — no API key needed.
    interval: 1d, 4h, 1h
    """
    base_url = "https://api.binance.com/api/v3/klines"
    start_ms  = int(pd.Timestamp(start_date).timestamp() * 1000)
    all_klines = []

    while True:
        params = {
            "symbol"   : symbol,
            "interval" : interval,
            "startTime": start_ms,
            "limit"    : 1000,
        }
        try:
            resp = requests.get(base_url, params=params, timeout=15)
            resp.raise_for_status()
            klines = resp.json()
            if not klines:
                break
            all_klines.extend(klines)
            last_ts = klines[-1][0]
            if last_ts == start_ms:
                break
            start_ms = last_ts + 1
            if len(klines) < 1000:
                break
            time.sleep(0.1)
        except Exception as e:
            logger.warning(f"  ⚠ Binance fetch failed for {symbol}: {e}")
            break

    if not all_klines:
        return pd.DataFrame()

    df = pd.DataFrame(all_klines, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_volume","trades","taker_base","taker_quote","ignore"
    ])
    df["date"]   = pd.to_datetime(df["open_time"], unit="ms").dt.normalize()
    df["open"]   = df["open"].astype(float)
    df["high"]   = df["high"].astype(float)
    df["low"]    = df["low"].astype(float)
    df["close"]  = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df = df[["date","open","high","low","close","volume"]].drop_duplicates("date")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def ingest_crypto(start_date: str = "2020-01-01") -> dict:
    """Fetch all crypto assets and return normalized DataFrames."""
    results = {}
    for name, symbol in CRYPTO_ASSETS.items():
        logger.info(f"  Fetching {name} ({symbol})...")
        df = fetch_binance_klines(symbol, "1d", start_date)
        if df.empty:
            logger.warning(f"  ⚠ No data for {name}")
            continue
        df["asset"]       = name
        df["asset_class"] = "crypto"
        df["return_1d"]   = df["close"].pct_change()
        df["log_return"]  = np.log(df["close"] / df["close"].shift(1))
        df["vol_21d"]     = df["return_1d"].rolling(21).std() * np.sqrt(252)
        df["vol_7d"]      = df["return_1d"].rolling(7).std()  * np.sqrt(252)
        df["momentum_5d"] = df["close"].pct_change(5)
        df["momentum_21d"]= df["close"].pct_change(21)
        df["rsi_14"]      = _compute_rsi(df["close"], 14)
        df = df.dropna().reset_index(drop=True)
        results[name]     = df
        logger.info(f"  ✓ {name}: {len(df)} days | {df['date'].min().date()} → {df['date'].max().date()}")
    return results


# ── Commodity ingestion via yfinance ─────────────────────────────────────────

def ingest_commodities(start_date: str = "2020-01-01") -> dict:
    """Fetch commodity futures via yfinance."""
    results = {}
    for name, ticker in COMMODITY_TICKERS.items():
        logger.info(f"  Fetching {name} ({ticker})...")
        try:
            raw = yf.download(ticker, start=start_date,
                              auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            if raw.empty:
                continue
            df = raw[["Open","High","Low","Close","Volume"]].copy()
            df.columns = ["open","high","low","close","volume"]
            df.index.name = "date"
            df = df.reset_index()
            df["date"]        = pd.to_datetime(df["date"]).dt.normalize()
            df["asset"]       = name
            df["asset_class"] = "commodity"
            df["return_1d"]   = df["close"].pct_change()
            df["vol_21d"]     = df["return_1d"].rolling(21).std() * np.sqrt(252)
            df["momentum_5d"] = df["close"].pct_change(5)
            df["momentum_21d"]= df["close"].pct_change(21)
            df["rsi_14"]      = _compute_rsi(df["close"], 14)
            df = df.dropna().reset_index(drop=True)
            results[name] = df
            logger.info(f"  ✓ {name}: {len(df)} days")
        except Exception as e:
            logger.warning(f"  ⚠ {name} failed: {e}")
    return results


# ── Bond ETF ingestion via yfinance ──────────────────────────────────────────

def ingest_bonds(start_date: str = "2020-01-01") -> dict:
    """Fetch bond ETFs via yfinance as proxy for bond market."""
    results = {}
    for name, ticker in BOND_ETF_TICKERS.items():
        logger.info(f"  Fetching {name} ({ticker})...")
        try:
            raw = yf.download(ticker, start=start_date,
                              auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            if raw.empty:
                continue
            df = raw[["Open","High","Low","Close","Volume"]].copy()
            df.columns = ["open","high","low","close","volume"]
            df.index.name = "date"
            df = df.reset_index()
            df["date"]        = pd.to_datetime(df["date"]).dt.normalize()
            df["asset"]       = name
            df["asset_class"] = "bond"
            df["return_1d"]   = df["close"].pct_change()
            df["vol_21d"]     = df["return_1d"].rolling(21).std() * np.sqrt(252)
            df["momentum_5d"] = df["close"].pct_change(5)
            df["duration_proxy"] = df["close"] / df["close"].rolling(252).mean()
            df = df.dropna().reset_index(drop=True)
            results[name] = df
            logger.info(f"  ✓ {name}: {len(df)} days")
        except Exception as e:
            logger.warning(f"  ⚠ {name} failed: {e}")
    return results


# ── FRED macro data ───────────────────────────────────────────────────────────

def fetch_fred_series(series_id: str, start_date: str = "2020-01-01") -> pd.Series:
    """Fetch a single FRED series. Falls back to yfinance proxy if no API key."""
    if FRED_API_KEY:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id"      : series_id,
            "api_key"        : FRED_API_KEY,
            "file_type"      : "json",
            "observation_start": start_date,
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            obs  = data.get("observations", [])
            if obs:
                s = pd.Series(
                    {pd.Timestamp(o["date"]): float(o["value"])
                     for o in obs if o["value"] != "."},
                    name=series_id
                )
                return s
        except Exception as e:
            logger.warning(f"  ⚠ FRED {series_id} failed: {e}")

    # Free fallback — use yfinance proxies
    proxies = {
        "GS10"    : "^TNX",
        "GS2"     : "^IRX",
        "T10Y2Y"  : None,
        "VIXCLS"  : "^VIX",
    }
    proxy = proxies.get(series_id)
    if proxy:
        try:
            raw = yf.download(proxy, start=start_date,
                              auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            return raw["Close"].rename(series_id)
        except Exception:
            pass
    return pd.Series(name=series_id, dtype=float)


def ingest_macro_factors(start_date: str = "2020-01-01") -> pd.DataFrame:
    """Build daily macro factor DataFrame from FRED + yfinance."""
    logger.info("  Fetching macro factors...")
    series = {}
    for name, fred_id in FRED_SERIES.items():
        s = fetch_fred_series(fred_id, start_date)
        if not s.empty:
            series[name] = s

    if not series:
        return pd.DataFrame()

    macro_df = pd.DataFrame(series)
    macro_df.index = pd.to_datetime(macro_df.index)
    macro_df = macro_df.resample("D").last().ffill()
    macro_df = macro_df.reset_index().rename(columns={"index": "date"})
    macro_df["yield_curve_slope"] = macro_df.get("yield_10y", 0) - macro_df.get("yield_2y", 0)
    macro_df["real_rate"]         = macro_df.get("yield_10y", 0) - macro_df.get("cpi_yoy", 0)
    logger.info(f"  ✓ Macro factors: {len(macro_df)} days | {macro_df.columns.tolist()}")
    return macro_df


# ── Helper ────────────────────────────────────────────────────────────────────

def _compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


# ── S3 upload ─────────────────────────────────────────────────────────────────

def upload_to_s3(df: pd.DataFrame, asset_class: str, asset_name: str, s3_client) -> str:
    key      = f"raw/multi_asset/{asset_class}/{asset_name}/{asset_name}_daily.parquet"
    buf      = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3_client.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())
    return f"s3://{S3_BUCKET}/{key}"


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_multi_asset_ingestion(start_date: str = "2020-01-01") -> dict:
    s3 = boto3.client("s3", region_name=AWS_REGION)

    logger.info("=" * 60)
    logger.info("AlphaFlow Phase 9 — Multi-Asset Ingestion")
    logger.info("=" * 60)
    logger.info(f"  Crypto    : {list(CRYPTO_ASSETS.keys())}")
    logger.info(f"  Commodities: {list(COMMODITY_TICKERS.keys())}")
    logger.info(f"  Bonds     : {list(BOND_ETF_TICKERS.keys())}")
    logger.info(f"  Macro     : {list(FRED_SERIES.keys())}")
    logger.info("")

    results = {"crypto": {}, "commodity": {}, "bond": {}, "macro": None}

    # Crypto
    logger.info("Step 1: Crypto (Binance)...")
    results["crypto"] = ingest_crypto(start_date)
    for name, df in results["crypto"].items():
        uri = upload_to_s3(df, "crypto", name, s3)
        logger.info(f"  ✓ Uploaded {name} → {uri}")

    # Commodities
    logger.info("\nStep 2: Commodities (yfinance)...")
    results["commodity"] = ingest_commodities(start_date)
    for name, df in results["commodity"].items():
        uri = upload_to_s3(df, "commodity", name, s3)
        logger.info(f"  ✓ Uploaded {name} → {uri}")

    # Bonds
    logger.info("\nStep 3: Bonds (yfinance ETFs)...")
    results["bond"] = ingest_bonds(start_date)
    for name, df in results["bond"].items():
        uri = upload_to_s3(df, "bond", name, s3)
        logger.info(f"  ✓ Uploaded {name} → {uri}")

    # Macro
    logger.info("\nStep 4: Macro factors (FRED/yfinance)...")
    results["macro"] = ingest_macro_factors(start_date)
    if results["macro"] is not None and not results["macro"].empty:
        uri = upload_to_s3(results["macro"], "macro", "global_macro", s3)
        logger.info(f"  ✓ Uploaded macro → {uri}")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Multi-Asset Ingestion Complete")
    logger.info("=" * 60)
    logger.info(f"  Crypto assets    : {len(results['crypto'])}")
    logger.info(f"  Commodities      : {len(results['commodity'])}")
    logger.info(f"  Bond ETFs        : {len(results['bond'])}")
    macro_cols = len(results["macro"].columns) if results["macro"] is not None else 0
    logger.info(f"  Macro factors    : {macro_cols}")
    logger.info("")
    logger.info("  All assets in S3: s3://alphaflow-data-lake-.../raw/multi_asset/")
    logger.info("  Next: run feature_pipeline_multi_asset.py to build features")

    return results


if __name__ == "__main__":
    results = run_multi_asset_ingestion(start_date="2020-01-01")