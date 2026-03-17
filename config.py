# config.py — Central configuration for AlphaFlow
import os
from dotenv import load_dotenv
load_dotenv()

# ── AWS ──────────────────────────────────────────────────────────────
AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")
S3_BUCKET  = os.getenv("S3_BUCKET", "alphaflow-data-lake-291572330987")

# ── API KEYS ─────────────────────────────────────────────────────────
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
SEC_API_KEY  = os.getenv("SEC_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")

# ── STOCK UNIVERSE — 25 tickers across 7 sectors ─────────────────────
TICKERS = [
    # Technology (5)
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Alphabet
    "NVDA",   # Nvidia        ← NEW
    "AMD",    # AMD           ← NEW

    # Consumer / E-commerce (3)
    "AMZN",   # Amazon
    "WMT",    # Walmart       ← NEW
    "COST",   # Costco        ← NEW

    # Finance (4)
    "JPM",    # JPMorgan
    "GS",     # Goldman Sachs ← NEW
    "BAC",    # Bank of America ← NEW
    "V",      # Visa          ← NEW

    # Healthcare (4)
    "JNJ",    # J&J
    "UNH",    # UnitedHealth  ← NEW
    "PFE",    # Pfizer        ← NEW
    "ABBV",   # AbbVie        ← NEW

    # Energy (2)
    "XOM",    # ExxonMobil
    "CVX",    # Chevron       ← NEW

    # Diversified / Defensive (3)
    "BRK-B",  # Berkshire
    "PG",     # Procter & Gamble ← NEW
    "KO",     # Coca-Cola     ← NEW

    # High-growth / Volatile (4)
    "TSLA",   # Tesla
    "META",   # Meta          ← NEW
    "NFLX",   # Netflix       ← NEW
    "CRM",    # Salesforce    ← NEW
]

BENCHMARK_TICKER = "SPY"

# ── DATA SETTINGS ────────────────────────────────────────────────────
HISTORICAL_START = "2020-01-01"
HISTORICAL_END   = "2026-03-16"
DATA_INTERVAL    = "1d"

# ── S3 PATH STRUCTURE ────────────────────────────────────────────────
S3_RAW_PREFIX       = "raw"
S3_PROCESSED_PREFIX = "processed"
S3_FEATURES_PREFIX  = "features"

# ── LOCAL PATHS ──────────────────────────────────────────────────────
LOCAL_DATA_DIR = "data/local"