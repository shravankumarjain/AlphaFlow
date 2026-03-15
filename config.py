# config.py — Central configuration for AlphaFlow
# All settings in one place. Never hardcode values in pipeline files.

import os
from dotenv import load_dotenv

load_dotenv()  # loads your .env file automatically

# ── AWS ──────────────────────────────────────────────────────────────
AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")
S3_BUCKET = os.getenv("S3_BUCKET", "alphaflow-data-lake-291572330987")

# ── API KEYS ─────────────────────────────────────────────────────────
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
SEC_API_KEY = os.getenv("SEC_API_KEY")  # from sec-api.io (free tier)
FRED_API_KEY = os.getenv("FRED_API_KEY")  # optional macro data

# ── STOCK UNIVERSE ───────────────────────────────────────────────────
# Starting small — 10 large-cap tickers across sectors
# We'll expand this later when the pipeline is stable
TICKERS = [
    "AAPL",  # Apple        — Technology
    "MSFT",  # Microsoft    — Technology
    "GOOGL",  # Alphabet     — Communication
    "AMZN",  # Amazon       — Consumer
    "JPM",  # JPMorgan     — Finance
    "JNJ",  # J&J          — Healthcare
    "XOM",  # ExxonMobil   — Energy
    "BRK-B",  # Berkshire    — Diversified
    "TSLA",  # Tesla        — EV / Tech
    "SPY",  # S&P500 ETF   — Benchmark
]

# ── DATA SETTINGS ────────────────────────────────────────────────────
HISTORICAL_START = "2020-01-01"  # 5 years of training data
HISTORICAL_END = "2025-03-14"  # fixed end for reproducibility
DATA_INTERVAL = "1d"  # daily OHLCV

# ── S3 PATH STRUCTURE ────────────────────────────────────────────────
# We follow a medallion architecture: raw → processed → features
# raw/      = exactly as received from source, never modified
# processed/ = cleaned, validated, typed correctly
# features/  = model-ready engineered features
S3_RAW_PREFIX = "raw"
S3_PROCESSED_PREFIX = "processed"
S3_FEATURES_PREFIX = "features"

# ── LOCAL PATHS ──────────────────────────────────────────────────────
# During dev we also save locally so you can inspect files easily
LOCAL_DATA_DIR = "data/local"
