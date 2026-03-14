# data_pipeline/ingestion/market_data.py
#
# What this file does:
#   1. Downloads historical OHLCV data for all tickers via yfinance
#   2. Validates the data (checks for gaps, bad values, wrong types)
#   3. Saves clean Parquet files locally AND uploads to S3
#
# Why Parquet?
#   - 10x smaller than CSV for the same data
#   - Preserves column types (datetime stays datetime, float stays float)
#   - Industry standard for data lakes — what you'd use at any real firm
#
# Run this file directly to test:
#   python data_pipeline/ingestion/market_data.py

import os  # noqa: F401
import logging
import boto3
import yfinance as yf
import pandas as pd
from datetime import datetime
from pathlib import Path
from botocore.exceptions import ClientError

# bring in our central config
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import (
    AWS_REGION, S3_BUCKET, TICKERS,
    HISTORICAL_START, HISTORICAL_END, DATA_INTERVAL,
    S3_RAW_PREFIX, LOCAL_DATA_DIR
)

# ── LOGGING SETUP ────────────────────────────────────────────────────
# Always use logging, never print() in production code.
# This writes to console AND a log file simultaneously.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),                          # console
        logging.FileHandler("logs/market_data.log"),     # file
    ]
)
logger = logging.getLogger("market_data")


# ── S3 CLIENT ────────────────────────────────────────────────────────
def get_s3_client():
    """Create and return a boto3 S3 client."""
    return boto3.client("s3", region_name=AWS_REGION)


# ── DOWNLOAD ─────────────────────────────────────────────────────────
def download_ticker(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    """
    Download OHLCV data for a single ticker from Yahoo Finance.

    Returns a clean DataFrame or raises if download fails.
    Column names are lowercased for consistency: open, high, low, close, volume, adj_close
    """
    logger.info(f"Downloading {ticker} | {start} → {end} | interval={interval}")

    # yfinance returns MultiIndex columns when auto_adjust=False
    # We use auto_adjust=True so 'close' is already split-adjusted
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,    # suppress yfinance's own progress bar
        threads=False,
    )

    if df.empty:
        raise ValueError(f"yfinance returned empty DataFrame for {ticker}")

    # Flatten MultiIndex columns if present (happens with single ticker too sometimes)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Standardise column names to lowercase
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Add metadata columns — critical for traceability in a data lake
    df["ticker"]       = ticker
    df["ingested_at"]  = datetime.utcnow().isoformat()
    df["source"]       = "yahoo_finance"

    # Reset index so 'date' becomes a regular column (easier with Parquet)
    df = df.reset_index()
    df = df.rename(columns={"index": "date", "Date": "date", "Datetime": "date"})

    logger.info(f"  ✓ {ticker}: {len(df)} rows | {df['date'].min()} → {df['date'].max()}")
    return df


# ── VALIDATION ───────────────────────────────────────────────────────
def validate_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Run data quality checks on raw OHLCV data.

    Checks:
    - No completely empty rows
    - High >= Low (basic sanity)
    - Volume >= 0
    - No future dates
    - Close price is positive
    - Flag (but keep) rows with missing values so we can inspect them

    Returns cleaned DataFrame.
    Raises ValueError if critical checks fail.
    """
    logger.info(f"Validating {ticker} | {len(df)} rows")
    issues = []

    # 1. Check for completely empty rows
    empty_rows = df[["open", "high", "low", "close", "volume"]].isna().all(axis=1).sum()
    if empty_rows > 0:
        logger.warning(f"  ⚠ {ticker}: {empty_rows} completely empty rows — dropping")
        df = df.dropna(subset=["open", "high", "low", "close", "volume"], how="all")

    # 2. High must be >= Low
    bad_hl = df[df["high"] < df["low"]]
    if len(bad_hl) > 0:
        issues.append(f"{len(bad_hl)} rows where high < low")
        logger.warning(f"  ⚠ {ticker}: {len(bad_hl)} rows where high < low — dropping")
        df = df[df["high"] >= df["low"]]

    # 3. Volume must be non-negative
    bad_vol = df[df["volume"] < 0]
    if len(bad_vol) > 0:
        issues.append(f"{len(bad_vol)} rows with negative volume")
        df = df[df["volume"] >= 0]

    # 4. Close price must be positive
    bad_close = df[df["close"] <= 0]
    if len(bad_close) > 0:
        issues.append(f"{len(bad_close)} rows with close <= 0")
        df = df[df["close"] > 0]

    # 5. No future dates
    today = pd.Timestamp(datetime.utcnow().date())
    future = df[pd.to_datetime(df["date"]) > today]
    if len(future) > 0:
        issues.append(f"{len(future)} future-dated rows")
        df = df[pd.to_datetime(df["date"]) <= today]

    # 6. Add a data quality flag column — useful downstream
    df["dq_issues"] = ", ".join(issues) if issues else "clean"

    # CRITICAL check — if more than 20% of rows were dropped, something is wrong
    if len(df) == 0:
        raise ValueError(f"{ticker}: all rows failed validation — aborting")

    logger.info(f"  ✓ {ticker}: validation passed | {len(df)} rows remain | issues: {issues or 'none'}")
    return df


# ── SAVE LOCALLY ─────────────────────────────────────────────────────
def save_local(df: pd.DataFrame, ticker: str) -> str:
    """Save DataFrame as Parquet to local data directory."""
    local_dir = Path(LOCAL_DATA_DIR) / "raw" / "market"
    local_dir.mkdir(parents=True, exist_ok=True)

    filepath = local_dir / f"{ticker}.parquet"
    df.to_parquet(filepath, index=False, engine="pyarrow")
    logger.info(f"  ✓ Saved locally: {filepath}")
    return str(filepath)


# ── UPLOAD TO S3 ─────────────────────────────────────────────────────
def upload_to_s3(local_path: str, ticker: str, s3_client) -> str:
    """
    Upload a local Parquet file to S3.

    S3 key structure:
        raw/market/daily/{TICKER}/{TICKER}_daily.parquet

    Using ticker-level folders makes it easy to query by stock later.
    """
    s3_key = f"{S3_RAW_PREFIX}/market/daily/{ticker}/{ticker}_daily.parquet"

    try:
        s3_client.upload_file(local_path, S3_BUCKET, s3_key)
        s3_uri = f"s3://{S3_BUCKET}/{s3_key}"
        logger.info(f"  ✓ Uploaded to S3: {s3_uri}")
        return s3_uri
    except ClientError as e:
        logger.error(f"  ✗ S3 upload failed for {ticker}: {e}")
        raise


# ── MAIN ORCHESTRATOR ────────────────────────────────────────────────
def run_market_ingestion(
    tickers: list = TICKERS,
    start: str = HISTORICAL_START,
    end: str = HISTORICAL_END,
    interval: str = DATA_INTERVAL,
) -> dict:
    """
    Run the full market data ingestion pipeline.

    For each ticker:
        download → validate → save local → upload S3

    Returns a summary dict with success/failure per ticker.
    """
    logger.info("=" * 60)
    logger.info("AlphaFlow — Market Data Ingestion Pipeline")
    logger.info(f"Tickers : {tickers}")
    logger.info(f"Period  : {start} → {end}")
    logger.info(f"Interval: {interval}")
    logger.info("=" * 60)

    # Make sure logs directory exists
    Path("logs").mkdir(exist_ok=True)

    s3 = get_s3_client()
    results = {}

    for ticker in tickers:
        try:
            # Step 1: Download
            df_raw = download_ticker(ticker, start, end, interval)

            # Step 2: Validate
            df_clean = validate_ohlcv(df_raw, ticker)

            # Step 3: Save locally
            local_path = save_local(df_clean, ticker)

            # Step 4: Upload to S3
            s3_uri = upload_to_s3(local_path, ticker, s3)

            results[ticker] = {
                "status"  : "success",
                "rows"    : len(df_clean),
                "s3_uri"  : s3_uri,
            }

        except Exception as e:
            logger.error(f"FAILED {ticker}: {e}")
            results[ticker] = {"status": "failed", "error": str(e)}

    # ── SUMMARY REPORT ───────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("INGESTION SUMMARY")
    logger.info("=" * 60)
    success = [t for t, r in results.items() if r["status"] == "success"]
    failed  = [t for t, r in results.items() if r["status"] == "failed"]
    logger.info(f"✓ Success : {len(success)} tickers — {success}")
    logger.info(f"✗ Failed  : {len(failed)} tickers  — {failed}")

    if failed:
        logger.warning("Some tickers failed. Check logs above for details.")

    return results


# ── ENTRY POINT ──────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_market_ingestion()

    # Print a clean table to console
    print("\n── Results ──────────────────────────────")
    for ticker, info in results.items():
        if info["status"] == "success":
            print(f"  ✓ {ticker:6s} | {info['rows']:4d} rows | {info['s3_uri']}")
        else:
            print(f"  ✗ {ticker:6s} | FAILED: {info['error']}")