# feature_engineering/feature_pipeline.py
#
# AlphaFlow — Full Feature Engineering Pipeline
#
# Signal Architecture (inspired by systematic hedge fund design):
#
#   PRIMARY (leading signals)
#   ├── Sentiment velocity        — rate of change in news sentiment
#   ├── SEC filing patterns       — filing delays, 8-K frequency spikes
#   ├── Options implied volatility — market's forward fear gauge
#   └── Macro regime              — Fed direction, yield curve, VIX regime
#
#   SECONDARY (cross-asset / market structure)
#   ├── Cross-asset correlations  — rolling corr vs SPY, bonds, VIX
#   ├── Relative strength         — ticker vs sector vs market
#   └── Volume profile            — institutional accumulation signals
#
#   CONFIRMATION (lagging — used as filters only)
#   ├── RSI, MACD, Bollinger Bands
#   └── Rolling volatility regimes
#
# Output: single flat Parquet per ticker in S3 processed/ layer
#         ready to feed directly into TFT model in Phase 3
#
# Run: python feature_engineering/feature_pipeline.py

import warnings
warnings.filterwarnings("ignore")

import logging  # noqa: E402
import boto3 # noqa: E402
import json # noqa: E402
import pandas as pd # noqa: E402
import numpy as np # noqa: E402
import yfinance as yf # noqa: E402
from datetime import datetime, timedelta # noqa: E402
from pathlib import Path # noqa: E402
from io import BytesIO # noqa: E402

import sys # noqa: E402
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    AWS_REGION, S3_BUCKET, TICKERS,
    S3_RAW_PREFIX, S3_PROCESSED_PREFIX, S3_FEATURES_PREFIX,
    LOCAL_DATA_DIR, HISTORICAL_START, HISTORICAL_END
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/feature_pipeline.log"),
    ]
)
logger = logging.getLogger("feature_pipeline")

s3 = boto3.client("s3", region_name=AWS_REGION)


# ═══════════════════════════════════════════════════════════════════════
# HELPERS — S3 read/write
# ═══════════════════════════════════════════════════════════════════════

def read_parquet_from_s3(s3_key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
    return pd.read_parquet(BytesIO(obj["Body"].read()))


def write_parquet_to_s3(df: pd.DataFrame, s3_key: str) -> str:
    buffer = BytesIO()
    df.to_parquet(buffer, index=False, engine="pyarrow")
    buffer.seek(0)
    s3.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=buffer.getvalue())
    uri = f"s3://{S3_BUCKET}/{s3_key}"
    logger.info(f"  ✓ Written to S3: {uri}")
    return uri


def read_json_from_s3(s3_key: str) -> list:
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
        return json.loads(obj["Body"].read())
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════
# BLOCK 1 — LAGGING CONFIRMATION SIGNALS
# RSI, MACD, Bollinger Bands, rolling volatility
# Used as regime filters, not primary predictors
# ═══════════════════════════════════════════════════════════════════════

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators using pure pandas — no TA-Lib dependency.
    These are CONFIRMATION signals only, not primary drivers.
    """
    close = df["close"]
    high  = df["high"]  # noqa: F841
    low   = df["low"]  # noqa: F841
    vol   = df["volume"]

    # ── RSI (14-day) ──────────────────────────────────────────────────
    delta     = close.diff()
    gain      = delta.clip(lower=0)
    loss      = -delta.clip(upper=0)
    avg_gain  = gain.ewm(com=13, adjust=False).mean()
    avg_loss  = loss.ewm(com=13, adjust=False).mean()
    rs        = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # ── MACD (12, 26, 9) ──────────────────────────────────────────────
    ema12          = close.ewm(span=12, adjust=False).mean()
    ema26          = close.ewm(span=26, adjust=False).mean()
    df["macd"]     = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # ── Bollinger Bands (20-day, 2 std) ───────────────────────────────
    sma20              = close.rolling(20).mean()
    std20              = close.rolling(20).std()
    df["bb_upper"]     = sma20 + 2 * std20
    df["bb_lower"]     = sma20 - 2 * std20
    df["bb_mid"]       = sma20
    # %B — where price sits within the bands (0=lower, 1=upper)
    df["bb_pct"]       = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)
    df["bb_width"]     = (df["bb_upper"] - df["bb_lower"]) / (sma20 + 1e-9)

    # ── Rolling Volatility ────────────────────────────────────────────
    log_ret             = np.log(close / close.shift(1))
    df["vol_5d"]        = log_ret.rolling(5).std()  * np.sqrt(252)
    df["vol_21d"]       = log_ret.rolling(21).std() * np.sqrt(252)
    df["vol_63d"]       = log_ret.rolling(63).std() * np.sqrt(252)
    # Volatility regime: is current vol above its own 63-day average?
    df["vol_regime"]    = (df["vol_21d"] > df["vol_21d"].rolling(63).mean()).astype(int)

    # ── Volume signals ────────────────────────────────────────────────
    df["volume_sma20"]  = vol.rolling(20).mean()
    # Volume ratio > 2 = unusual institutional activity
    df["volume_ratio"]  = vol / (df["volume_sma20"] + 1e-9)

    # ── Price momentum (returns over multiple horizons) ───────────────
    for d in [1, 5, 10, 21, 63]:
        df[f"return_{d}d"] = close.pct_change(d)

    # ── Moving average crossovers (trend direction) ───────────────────
    df["sma_20"]        = close.rolling(20).mean()
    df["sma_50"]        = close.rolling(50).mean()
    df["sma_200"]       = close.rolling(200).mean()
    df["above_sma50"]   = (close > df["sma_50"]).astype(int)
    df["above_sma200"]  = (close > df["sma_200"]).astype(int)
    # Golden cross signal: 50 > 200 SMA
    df["golden_cross"]  = (df["sma_50"] > df["sma_200"]).astype(int)

    logger.info(f"    ✓ Technical indicators: {len([c for c in df.columns if any(x in c for x in ['rsi','macd','bb','vol','return','sma'])])} features")
    return df


# ═══════════════════════════════════════════════════════════════════════
# BLOCK 2 — MACRO REGIME SIGNALS (PRIMARY)
# Fed direction, yield curve, VIX regime
# These tell the model WHAT KIND of market we're in
# Aladdin uses macro regimes to switch between sub-models
# ═══════════════════════════════════════════════════════════════════════

def build_macro_features(start: str, end: str) -> pd.DataFrame:
    """
    Pull macro data via yfinance and compute regime signals.

    Tickers used:
    ^VIX   — CBOE Volatility Index (fear gauge)
    ^TNX   — 10-year Treasury yield
    ^IRX   — 3-month Treasury yield
    TLT    — 20-year Treasury ETF (bond market direction)
    GLD    — Gold (risk-off signal)
    DX-Y.NYB — US Dollar Index
    """
    logger.info("  Building macro regime features...")

    macro_tickers = {
        "^VIX"     : "vix",
        "^TNX"     : "yield_10y",
        "^IRX"     : "yield_3m",
        "TLT"      : "bonds",
        "GLD"      : "gold",
        "DX-Y.NYB" : "dxy",
    }

    frames = {}
    for ticker, name in macro_tickers.items():
        try:
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=True, progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                frames[name] = df["Close"].rename(name)
        except Exception as e:
            logger.warning(f"    ⚠ Could not fetch {ticker}: {e}")

    if not frames:
        logger.warning("    ⚠ No macro data fetched — skipping macro features")
        return pd.DataFrame()

    macro = pd.concat(frames.values(), axis=1)
    macro.index = pd.to_datetime(macro.index)

    # ── Yield curve slope (10y - 3m) ─────────────────────────────────
    # Negative = inverted = recession signal (leading indicator)
    if "yield_10y" in macro.columns and "yield_3m" in macro.columns:
        macro["yield_curve"] = macro["yield_10y"] - macro["yield_3m"]
        macro["yield_curve_inverted"] = (macro["yield_curve"] < 0).astype(int)

    # ── VIX regime ────────────────────────────────────────────────────
    # VIX < 15 = calm, 15-25 = normal, > 25 = fear, > 35 = crisis
    if "vix" in macro.columns:
        macro["vix_regime"] = pd.cut(
            macro["vix"],
            bins=[0, 15, 25, 35, 999],
            labels=[0, 1, 2, 3]   # 0=calm, 1=normal, 2=fear, 3=crisis
        ).astype(float)
        macro["vix_30d_change"] = macro["vix"].pct_change(30)

    # ── Bond trend (risk-on vs risk-off) ─────────────────────────────
    # Rising bonds (TLT up) = risk-off, money fleeing equities
    if "bonds" in macro.columns:
        macro["bonds_trend"] = macro["bonds"].pct_change(21)
        macro["risk_off"]    = (macro["bonds_trend"] > 0.02).astype(int)

    # ── Dollar strength ───────────────────────────────────────────────
    # Strong dollar = headwind for multinationals
    if "dxy" in macro.columns:
        macro["dxy_trend"] = macro["dxy"].pct_change(21)

    # ── Gold signal ───────────────────────────────────────────────────
    if "gold" in macro.columns:
        macro["gold_trend"] = macro["gold"].pct_change(21)

    # ── Overall macro regime score (composite) ────────────────────────
    # Combines multiple signals into a single regime label
    # 0 = risk-on (buy equities), 1 = neutral, 2 = risk-off (defensive)
    regime_score = pd.Series(0, index=macro.index)
    if "yield_curve_inverted" in macro.columns:
        regime_score += macro["yield_curve_inverted"]
    if "vix_regime" in macro.columns:
        regime_score += (macro["vix_regime"] >= 2).astype(int)
    if "risk_off" in macro.columns:
        regime_score += macro["risk_off"]

    macro["macro_regime"] = pd.cut(
        regime_score,
        bins=[-1, 0, 1, 999],
        labels=[0, 1, 2]
    ).astype(float)

    macro = macro.reset_index().rename(columns={"index": "date", "Date": "date"})
    logger.info(f"    ✓ Macro features: {len(macro.columns)} columns | {len(macro)} rows")
    return macro


# ═══════════════════════════════════════════════════════════════════════
# BLOCK 3 — OPTIONS IMPLIED VOLATILITY (PRIMARY)
# IV reflects what the market EXPECTS to happen — forward looking
# ═══════════════════════════════════════════════════════════════════════

def add_options_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Add options-derived features.
    We use the nearest-expiry ATM implied volatility as a signal.
    This is a genuine leading indicator — it reflects trader positioning.

    Note: yfinance only gives current options chain, not historical IV.
    For historical IV we use the VIX-style calculation: 30-day realised vol
    as a proxy, plus put/call ratio from the current chain as a level signal.
    """
    try:
        t = yf.Ticker(ticker)
        expirations = t.options

        if not expirations:
            logger.warning(f"    ⚠ No options data for {ticker}")
            df["iv_atm_proxy"]   = np.nan
            df["put_call_ratio"] = np.nan
            return df

        # Get nearest expiry options chain
        nearest = expirations[0]
        chain   = t.option_chain(nearest)

        calls = chain.calls
        puts  = chain.puts

        # Put/Call open interest ratio — high = bearish positioning
        total_call_oi = calls["openInterest"].sum()
        total_put_oi  = puts["openInterest"].sum()
        pc_ratio = total_put_oi / (total_call_oi + 1e-9)

        # ATM IV — find the strike closest to current price
        current_price = df["close"].iloc[-1]
        atm_call = calls.iloc[(calls["strike"] - current_price).abs().argsort()[:1]]
        atm_iv   = atm_call["impliedVolatility"].values[0] if len(atm_call) > 0 else np.nan

        # Apply as constant signals (current snapshot)
        # In production these would be time-series — we log-note this limitation
        df["iv_atm_proxy"]   = atm_iv
        df["put_call_ratio"] = pc_ratio

        # IV premium: is current IV above realised vol? (vol risk premium)
        df["iv_premium"] = df["iv_atm_proxy"] - df["vol_21d"]

        logger.info(f"    ✓ Options features for {ticker}: IV={atm_iv:.3f}, P/C={pc_ratio:.3f}")

    except Exception as e:
        logger.warning(f"    ⚠ Options features failed for {ticker}: {e}")
        df["iv_atm_proxy"]   = np.nan
        df["put_call_ratio"] = np.nan
        df["iv_premium"]     = np.nan

    return df


# ═══════════════════════════════════════════════════════════════════════
# BLOCK 4 — SEC FILING PATTERN SIGNALS (PRIMARY)
# Filing velocity and 8-K frequency are genuine leading indicators
# ═══════════════════════════════════════════════════════════════════════

def add_sec_filing_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Extract signals from SEC filing metadata stored in S3.

    Key signals:
    1. Filing delay: days between quarter end and 10-Q filing
       → Late filers (>40 days) historically underperform
       → Very early filers tend to have positive surprises

    2. 8-K frequency spike: sudden increase in material event filings
       → Spike = something significant happening (M&A, CEO change, etc.)

    3. Filing recency: days since last 10-K or 10-Q
       → Stale filings = information gap = higher uncertainty
    """
    date_str = datetime.utcnow().strftime("%Y%m%d")
    s3_key   = f"{S3_RAW_PREFIX}/news/sec_edgar/{ticker}/{ticker}_{date_str}.json"
    filings  = read_json_from_s3(s3_key)

    # Default values if no filings available
    df["sec_8k_count_90d"]    = 0
    df["sec_filing_delay"]    = np.nan
    df["sec_days_since_10q"]  = np.nan
    df["sec_filing_spike"]    = 0

    if not filings:
        logger.warning(f"    ⚠ No SEC filings found for {ticker}")
        return df

    filing_df = pd.DataFrame(filings)
    filing_df["filing_date"] = pd.to_datetime(filing_df["filing_date"], errors="coerce")
    filing_df = filing_df.dropna(subset=["filing_date"])

    # ── 8-K frequency in rolling 90-day windows ───────────────────────
    eightk = filing_df[filing_df["form_type"] == "8-K"].copy()
    if len(eightk) > 0:
        eightk = eightk.sort_values("filing_date")
        # Count 8-Ks in trailing 90 days from each date in our price data
        df_dates = pd.to_datetime(df["date"])
        counts = []
        for d in df_dates:
            window_start = d - timedelta(days=90)
            cnt = ((eightk["filing_date"] >= window_start) &
                   (eightk["filing_date"] <= d)).sum()
            counts.append(cnt)
        df["sec_8k_count_90d"] = counts

        # Spike = current 90d count > 1.5x historical average
        mean_8k = df["sec_8k_count_90d"].mean()
        df["sec_filing_spike"] = (df["sec_8k_count_90d"] > mean_8k * 1.5).astype(int)

    # ── 10-Q filing delay ─────────────────────────────────────────────
    # Quarter ends: Mar 31, Jun 30, Sep 30, Dec 31
    # SEC deadline: 40 days after quarter end for large accelerated filers
    tenq = filing_df[filing_df["form_type"] == "10-Q"].copy()
    if len(tenq) > 0:
        tenq = tenq.sort_values("filing_date")
        quarter_ends = pd.to_datetime([
            "2020-03-31","2020-06-30","2020-09-30","2020-12-31",
            "2021-03-31","2021-06-30","2021-09-30","2021-12-31",
            "2022-03-31","2022-06-30","2022-09-30","2022-12-31",
            "2023-03-31","2023-06-30","2023-09-30","2023-12-31",
            "2024-03-31","2024-06-30","2024-09-30","2024-12-31",
        ])
        delays = []
        for qe in quarter_ends:
            # Find first 10-Q filed after this quarter end
            after = tenq[tenq["filing_date"] > qe]
            if len(after) > 0:
                delay = (after.iloc[0]["filing_date"] - qe).days
                if delay < 120:  # ignore if > 4 months (probably wrong quarter)
                    delays.append(delay)
        if delays:
            avg_delay = np.mean(delays)
            df["sec_filing_delay"] = avg_delay
            logger.info(f"    ✓ SEC filing delay for {ticker}: avg {avg_delay:.0f} days")

    # ── Days since last 10-Q ──────────────────────────────────────────
    tenq_dates = filing_df[filing_df["form_type"].isin(["10-Q","10-K"])]["filing_date"]
    if len(tenq_dates) > 0:
        last_filing = tenq_dates.max()
        df_dates    = pd.to_datetime(df["date"])
        df["sec_days_since_10q"] = (df_dates - last_filing).dt.days.clip(lower=0)

    logger.info(f"    ✓ SEC features for {ticker}: {df['sec_8k_count_90d'].max():.0f} max 8-Ks/90d")
    return df


# ═══════════════════════════════════════════════════════════════════════
# BLOCK 5 — NEWS SENTIMENT VELOCITY (PRIMARY)
# Uses FinBERT scores stored from news ingestion
# Velocity (rate of change) matters more than absolute sentiment
# ═══════════════════════════════════════════════════════════════════════

def add_sentiment_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Load pre-computed FinBERT sentiment scores and engineer velocity features.

    Sentiment velocity = how fast sentiment is changing
    This is a leading indicator: accelerating negative sentiment
    precedes price drops by 1-3 days in academic literature.

    Note: FinBERT scoring is run in Phase 3 (models/sentiment).
    Here we load the scores IF they exist, else use neutral defaults.
    This allows the pipeline to run end-to-end before the NLP model is ready.
    """
    # Try to load pre-scored sentiment from S3
    date_str   = datetime.utcnow().strftime("%Y%m%d")
    s3_key     = f"{S3_PROCESSED_PREFIX}/sentiment/{ticker}/{ticker}_{date_str}_sentiment.parquet"

    try:
        sent_df = read_parquet_from_s3(s3_key)
        sent_df["date"] = pd.to_datetime(sent_df["date"])

        # Merge sentiment onto price data
        df["date"] = pd.to_datetime(df["date"])
        df = df.merge(
            sent_df[["date","sentiment_score","sentiment_label"]],
            on="date", how="left"
        )
        df["sentiment_score"] = df["sentiment_score"].fillna(0)

    except Exception:
        # FinBERT not run yet — use neutral placeholder
        # The pipeline is designed to work in stages
        df["sentiment_score"] = 0.0
        df["sentiment_label"] = "neutral"
        logger.info(f"    ℹ Sentiment scores not yet available for {ticker} — using neutral defaults")

    # ── Sentiment velocity features ───────────────────────────────────
    s = df["sentiment_score"]
    df["sentiment_ma3"]       = s.rolling(3).mean()    # 3-day smoothed
    df["sentiment_ma7"]       = s.rolling(7).mean()    # 7-day smoothed
    # Velocity: how much did sentiment change in last 3 days?
    df["sentiment_velocity"]  = s.diff(3)
    # Acceleration: is the velocity itself accelerating?
    df["sentiment_accel"]     = df["sentiment_velocity"].diff(3)
    # Divergence: price going up but sentiment going down = bearish signal
    price_direction     = np.sign(df["return_5d"]) if "return_5d" in df.columns else 0
    sent_direction      = np.sign(df["sentiment_velocity"])
    df["sentiment_divergence"] = (price_direction != sent_direction).astype(float)

    logger.info(f"    ✓ Sentiment features added for {ticker}")
    return df


# ═══════════════════════════════════════════════════════════════════════
# BLOCK 6 — RELATIVE STRENGTH & CROSS-ASSET (SECONDARY)
# How is this stock performing vs the market and its sector?
# ═══════════════════════════════════════════════════════════════════════

def add_relative_features(df: pd.DataFrame, ticker: str,
                           spy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute relative strength vs SPY (market benchmark).

    Relative strength = this stock's return / SPY's return
    RS > 1 = outperforming market (institutional money flowing in)
    RS < 1 = underperforming (distribution)

    Also adds beta (sensitivity to market moves) as a risk feature.
    """
    if spy_df.empty or ticker == "SPY":
        df["rs_vs_spy_21d"] = np.nan
        df["beta_63d"]      = np.nan
        return df

    df   = df.copy()
    spy  = spy_df.copy()

    df["date"]  = pd.to_datetime(df["date"])
    spy["date"] = pd.to_datetime(spy["date"])

    # Merge SPY close onto ticker df
    df = df.merge(
        spy[["date","close"]].rename(columns={"close":"spy_close"}),
        on="date", how="left"
    )

    ticker_ret = df["close"].pct_change()
    spy_ret    = df["spy_close"].pct_change()

    # ── Relative Strength (21-day) ────────────────────────────────────
    ticker_cum = (1 + ticker_ret).rolling(21).apply(lambda x: x.prod(), raw=True)
    spy_cum    = (1 + spy_ret).rolling(21).apply(lambda x: x.prod(), raw=True)
    df["rs_vs_spy_21d"] = ticker_cum / (spy_cum + 1e-9)

    # ── Beta (63-day rolling) ─────────────────────────────────────────
    # Beta > 1 = amplifies market moves (aggressive)
    # Beta < 1 = dampens market moves (defensive)
    def rolling_beta(y, x, window=63):
        betas = []
        for i in range(len(y)):
            if i < window:
                betas.append(np.nan)
            else:
                yi = y.iloc[i-window:i].values
                xi = x.iloc[i-window:i].values
                mask = ~(np.isnan(yi) | np.isnan(xi))
                if mask.sum() < 20:
                    betas.append(np.nan)
                else:
                    cov  = np.cov(yi[mask], xi[mask])[0][1]
                    var  = np.var(xi[mask])
                    betas.append(cov / var if var > 0 else np.nan)
        return pd.Series(betas, index=y.index)

    df["beta_63d"] = rolling_beta(ticker_ret, spy_ret)

    # ── Excess return (alpha proxy) ───────────────────────────────────
    df["excess_return_21d"] = (
        df["close"].pct_change(21) - df["spy_close"].pct_change(21)
    )

    logger.info(f"    ✓ Relative features for {ticker}: beta={df['beta_63d'].dropna().mean():.2f}")
    return df


# ═══════════════════════════════════════════════════════════════════════
# MASTER PIPELINE — wires all blocks together
# ═══════════════════════════════════════════════════════════════════════

def build_features_for_ticker(
    ticker: str,
    macro_df: pd.DataFrame,
    spy_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline for one ticker.

    Order matters:
    1. Load raw OHLCV from S3
    2. Add technical (lagging) indicators
    3. Add options features
    4. Add SEC filing patterns
    5. Add sentiment velocity
    6. Add relative strength vs SPY
    7. Merge macro regime
    8. Final cleanup
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Building features: {ticker}")
    logger.info(f"{'='*50}")

    # ── Step 1: Load raw data ─────────────────────────────────────────
    s3_key = f"{S3_RAW_PREFIX}/market/daily/{ticker}/{ticker}_daily.parquet"
    try:
        df = read_parquet_from_s3(s3_key)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        logger.info(f"  ✓ Loaded {len(df)} rows from S3")
    except Exception as e:
        logger.error(f"  ✗ Could not load {ticker} from S3: {e}")
        return pd.DataFrame()

    # ── Step 2: Technical indicators ─────────────────────────────────
    logger.info("  Block 1: Technical indicators...")
    df = add_technical_indicators(df)

    # ── Step 3: Options features ──────────────────────────────────────
    logger.info("  Block 3: Options features...")
    df = add_options_features(df, ticker)

    # ── Step 4: SEC filing patterns ───────────────────────────────────
    logger.info("  Block 4: SEC filing patterns...")
    df = add_sec_filing_features(df, ticker)

    # ── Step 5: Sentiment velocity ────────────────────────────────────
    logger.info("  Block 5: Sentiment features...")
    df = add_sentiment_features(df, ticker)

    # ── Step 6: Relative strength ─────────────────────────────────────
    logger.info("  Block 6: Relative strength...")
    df = add_relative_features(df, ticker, spy_df)

    # ── Step 7: Merge macro regime ────────────────────────────────────
    logger.info("  Block 7: Macro regime merge...")
    if not macro_df.empty:
        macro_df["date"] = pd.to_datetime(macro_df["date"])
        macro_cols = [c for c in macro_df.columns if c != "date"]
        df = df.merge(macro_df[["date"] + macro_cols], on="date", how="left")
        # Forward fill macro features over weekends/holidays
        df[macro_cols] = df[macro_cols].ffill()
        logger.info(f"    ✓ Merged {len(macro_cols)} macro features")

    # ── Step 8: Final cleanup ─────────────────────────────────────────
    # Drop rows where we don't have enough history for rolling windows
    # 200-day SMA requires 200 rows — before that we have NaN
    df = df.iloc[200:].reset_index(drop=True)

    # Replace inf values (from division) with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Add target variable — what we're predicting
    # Forward 5-day return (will be NaN for last 5 rows — that's correct)
    df["target_return_5d"] = df["close"].pct_change(5).shift(-5)

    feature_cols = [c for c in df.columns if c not in
                    ["date","ticker","ingested_at","source","dq_issues"]]
    logger.info(f"  ✓ Final feature set: {len(feature_cols)} features | {len(df)} rows")

    return df


def run_feature_pipeline(tickers: list = TICKERS) -> dict:
    """Run the full feature pipeline for all tickers."""
    logger.info("=" * 60)
    logger.info("AlphaFlow — Feature Engineering Pipeline")
    logger.info(f"Tickers: {tickers}")
    logger.info("=" * 60)

    Path("logs").mkdir(exist_ok=True)

    # ── Build macro features once (shared across all tickers) ─────────
    logger.info("\nBuilding macro regime features (shared)...")
    macro_df = build_macro_features(HISTORICAL_START, HISTORICAL_END)

    # ── Load SPY once as benchmark ────────────────────────────────────
    logger.info("Loading SPY benchmark...")
    try:
        spy_df = read_parquet_from_s3(
            f"{S3_RAW_PREFIX}/market/daily/SPY/SPY_daily.parquet"
        )
    except Exception:
        spy_df = pd.DataFrame()
        logger.warning("Could not load SPY — relative features will be skipped")

    results = {}
    all_features = []

    for ticker in tickers:
        df = build_features_for_ticker(ticker, macro_df, spy_df)

        if df.empty:
            results[ticker] = {"status": "failed"}
            continue

        # ── Save to S3 features layer ─────────────────────────────────
        s3_key = f"{S3_FEATURES_PREFIX}/market/{ticker}/{ticker}_features.parquet"
        uri    = write_parquet_to_s3(df, s3_key)

        # ── Save locally for inspection ───────────────────────────────
        local_dir = Path(LOCAL_DATA_DIR) / "features"
        local_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(local_dir / f"{ticker}_features.parquet", index=False)

        results[ticker] = {
            "status"  : "success",
            "rows"    : len(df),
            "features": len(df.columns),
            "s3_uri"  : uri,
        }
        all_features.append(df)

    # ── Summary ───────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE PIPELINE SUMMARY")
    logger.info("=" * 60)
    for ticker, r in results.items():
        if r["status"] == "success":
            logger.info(f"  ✓ {ticker:6s} | {r['rows']:4d} rows | {r['features']:3d} features")
        else:
            logger.info(f"  ✗ {ticker:6s} | FAILED")

    return results


if __name__ == "__main__":
    results = run_feature_pipeline()
    print("\n── Feature Engineering Complete ──")
    for ticker, r in results.items():
        if r["status"] == "success":
            print(f"  ✓ {ticker:6s} | {r['rows']:4d} rows | {r['features']:3d} features | {r['s3_uri']}")
        else:
            print(f"  ✗ {ticker:6s} | FAILED")