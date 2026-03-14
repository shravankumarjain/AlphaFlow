# data_pipeline/ingestion/news_data.py
#
# What this file does:
#   1. Pulls financial news headlines from yfinance for each ticker
#   2. Pulls recent SEC filings metadata from EDGAR (earnings, 10-K, 10-Q)
#   3. Saves raw JSON → local + S3

import os  # noqa: F401
import json
import time
import logging
import requests
import boto3
import yfinance as yf
import pandas as pd  # noqa: F401
from datetime import datetime, timedelta  # noqa: F401
from pathlib import Path
from botocore.exceptions import ClientError

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import (
    AWS_REGION, S3_BUCKET, TICKERS,
    S3_RAW_PREFIX, LOCAL_DATA_DIR
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/news_data.log"),
    ]
)
logger = logging.getLogger("news_data")


# ── YFINANCE NEWS ─────────────────────────────────────────────────────
def fetch_yfinance_news(ticker: str) -> list:
    """
    Fetch news from yfinance — free, no API key, no rate limits.
    Returns list of article dicts with title, summary, source, url.
    """
    try:
        t = yf.Ticker(ticker)
        articles = []
        for item in t.news:
            articles.append({
                "ticker"     : ticker,
                "title"      : item.get("content", {}).get("title", ""),
                "summary"    : item.get("content", {}).get("summary", ""),
                "published"  : item.get("content", {}).get("pubDate", ""),
                "source"     : item.get("content", {}).get("provider", {}).get("displayName", ""),
                "url"        : item.get("content", {}).get("canonicalUrl", {}).get("url", ""),
                "fetched_at" : datetime.utcnow().isoformat(),
                "source_name": "yfinance_news",
            })
        logger.info(f"  ✓ yfinance news {ticker}: {len(articles)} articles")
        return articles
    except Exception as e:
        logger.error(f"  ✗ yfinance news failed for {ticker}: {e}")
        return []


# ── SEC EDGAR ─────────────────────────────────────────────────────────
def fetch_sec_filings(ticker: str, filing_types: list = None) -> list:
    """
    Fetch recent SEC filing metadata for a ticker from EDGAR.
    Completely free, no API key required.
    Filing types: 10-K (annual), 10-Q (quarterly), 8-K (material events).
    """
    if filing_types is None:
        filing_types = ["10-K", "10-Q", "8-K"]

    cik = get_cik_for_ticker(ticker)
    if not cik:
        logger.warning(f"  ⚠ Could not find CIK for {ticker} — skipping SEC")
        return []

    url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    headers = {"User-Agent": "AlphaFlow student-project alphaflow@gmail.com"}

    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json()

        filings_raw = data.get("filings", {}).get("recent", {})
        if not filings_raw:
            logger.warning(f"  ⚠ No filings found for {ticker}")
            return []

        forms        = filings_raw.get("form", [])
        filing_dates = filings_raw.get("filingDate", [])
        accession    = filings_raw.get("accessionNumber", [])
        descriptions = filings_raw.get("primaryDocument", [])

        filings = []
        for i, form in enumerate(forms):
            if form in filing_types:
                filings.append({
                    "ticker"          : ticker,
                    "cik"             : cik,
                    "form_type"       : form,
                    "filing_date"     : filing_dates[i] if i < len(filing_dates) else None,
                    "accession_number": accession[i] if i < len(accession) else None,
                    "primary_document": descriptions[i] if i < len(descriptions) else None,
                    "fetched_at"      : datetime.utcnow().isoformat(),
                    "source_name"     : "sec_edgar",
                })

        logger.info(f"  ✓ SEC EDGAR {ticker}: {len(filings)} filings ({', '.join(filing_types)})")
        time.sleep(0.2)
        return filings

    except Exception as e:
        logger.error(f"  ✗ SEC EDGAR fetch failed for {ticker}: {e}")
        return []


def get_cik_for_ticker(ticker: str) -> str | None:
    """Look up EDGAR CIK number for a stock ticker."""
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        headers = {"User-Agent": "AlphaFlow student-project alphaflow@gmail.com"}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        companies = response.json()

        for _, company in companies.items():
            if company.get("ticker", "").upper() == ticker.upper():
                return str(company["cik_str"])
        return None

    except Exception as e:
        logger.error(f"CIK lookup failed for {ticker}: {e}")
        return None


# ── SAVE & UPLOAD ─────────────────────────────────────────────────────
def save_and_upload_news(articles: list, ticker: str, source: str, s3_client) -> str | None:
    """Save news articles as JSON locally and upload to S3."""
    if not articles:
        return None

    local_dir = Path(LOCAL_DATA_DIR) / "raw" / "news" / source
    local_dir.mkdir(parents=True, exist_ok=True)

    date_str   = datetime.utcnow().strftime("%Y%m%d")
    filename   = f"{ticker}_{date_str}.json"
    local_path = local_dir / filename

    with open(local_path, "w") as f:
        json.dump(articles, f, indent=2, default=str)
    logger.info(f"  ✓ Saved locally: {local_path}")

    s3_key = f"{S3_RAW_PREFIX}/news/{source}/{ticker}/{filename}"
    try:
        s3_client.upload_file(str(local_path), S3_BUCKET, s3_key)
        s3_uri = f"s3://{S3_BUCKET}/{s3_key}"
        logger.info(f"  ✓ Uploaded to S3: {s3_uri}")
        return s3_uri
    except ClientError as e:
        logger.error(f"  ✗ S3 upload failed: {e}")
        return None


# ── MAIN ORCHESTRATOR ─────────────────────────────────────────────────
def run_news_ingestion(tickers: list = TICKERS) -> dict:
    """Run the full news ingestion pipeline for all tickers."""
    logger.info("=" * 60)
    logger.info("AlphaFlow — News & SEC Ingestion Pipeline")
    logger.info(f"Tickers: {tickers}")
    logger.info("=" * 60)

    Path("logs").mkdir(exist_ok=True)
    s3 = boto3.client("s3", region_name=AWS_REGION)
    results = {}

    for ticker in tickers:
        logger.info(f"\nProcessing {ticker}...")
        ticker_results = {}

        # ── yfinance news ──
        articles = fetch_yfinance_news(ticker)
        uri = save_and_upload_news(articles, ticker, "yfinance_news", s3)
        ticker_results["yfinance_news"] = {"count": len(articles), "s3_uri": uri}

        time.sleep(0.5)

        # ── SEC EDGAR ──
        filings = fetch_sec_filings(ticker)
        uri = save_and_upload_news(filings, ticker, "sec_edgar", s3)
        ticker_results["sec_edgar"] = {"count": len(filings), "s3_uri": uri}

        results[ticker] = ticker_results

    logger.info("\n" + "=" * 60)
    logger.info("NEWS INGESTION SUMMARY")
    logger.info("=" * 60)
    for ticker, r in results.items():
        news_count = r.get("yfinance_news", {}).get("count", 0)
        sec_count  = r.get("sec_edgar", {}).get("count", 0)
        logger.info(f"  {ticker:6s} | yfinance news: {news_count:3d} articles | SEC: {sec_count:3d} filings")

    return results


if __name__ == "__main__":
    results = run_news_ingestion()
    print("\n── News Ingestion Complete ──")
    for ticker, r in results.items():
        n = r.get("yfinance_news", {}).get("count", 0)
        s = r.get("sec_edgar", {}).get("count", 0)
        print(f"  {ticker:6s} | {n:3d} news | {s:3d} SEC filings")