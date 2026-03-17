"""
feature_engineering/feature_pipeline_multi_asset.py
AlphaFlow Phase 9 — Multi-Asset Feature Engineering
Builds the same feature format as equities for crypto, bonds, commodities.
All assets share the same TFT input schema — true multi-asset model.
"""

import os, logging, warnings  # noqa: E401
warnings.filterwarnings("ignore")

import boto3  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402
from io import BytesIO  # noqa: E402
from pathlib import Path  # noqa: E402, F401
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(module)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("feature_pipeline_multi")

S3_BUCKET  = os.getenv("S3_BUCKET", "alphaflow-data-lake-291572330987")
AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")

# All multi-asset tickers with their class and sector
MULTI_ASSETS = {
    # Crypto
    "BTC"       : ("crypto",    "Crypto"),
    "ETH"       : ("crypto",    "Crypto"),
    "SOL"       : ("crypto",    "Crypto"),
    "BNB"       : ("crypto",    "Crypto"),
    # Commodities
    "GOLD"      : ("commodity", "Commodity"),
    "OIL"       : ("commodity", "Commodity"),
    "SILVER"    : ("commodity", "Commodity"),
    "NATGAS"    : ("commodity", "Commodity"),
    "WHEAT"     : ("commodity", "Commodity"),
    # Bonds
    "US10Y_ETF" : ("bond",      "Bond"),
    "US30Y_ETF" : ("bond",      "Bond"),
    "TIPS_ETF"  : ("bond",      "Bond"),
    "HY_ETF"    : ("bond",      "Bond"),
    "IG_ETF"    : ("bond",      "Bond"),
}

# SPY for relative strength calculation
_spy_cache = None


def _get_spy(start_date="2020-01-01") -> pd.Series:
    global _spy_cache
    if _spy_cache is None:
        spy = yf.download("SPY", start=start_date, auto_adjust=True, progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        _spy_cache = spy["Close"].pct_change().rename("spy_return")
    return _spy_cache


def load_from_s3(asset_class: str, asset_name: str, s3_client) -> pd.DataFrame:
    key = f"raw/multi_asset/{asset_class}/{asset_name}/{asset_name}_daily.parquet"
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        df  = pd.read_parquet(BytesIO(obj["Body"].read()))
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)
    except Exception as e:
        logger.error(f"  ✗ Could not load {asset_name}: {e}")
        return pd.DataFrame()


def load_macro(s3_client) -> pd.DataFrame:
    try:
        obj = s3_client.get_object(
            Bucket=S3_BUCKET,
            Key="raw/multi_asset/macro/global_macro/global_macro_daily.parquet"
        )
        df = pd.read_parquet(BytesIO(obj["Body"].read()))
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)
    except Exception as e:
        logger.warning(f"  ⚠ Macro data not available: {e}")
        return pd.DataFrame()


def build_features(df: pd.DataFrame, asset_name: str,
                   asset_class: str, macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build 60+ features for a single multi-asset instrument.
    Matches the equity feature schema so TFT sees identical input structure.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # ── Block 1: Returns & volatility ────────────────────────────────────────
    df["return_1d"]     = df["close"].pct_change()
    df["return_5d"]     = df["close"].pct_change(5)
    df["return_21d"]    = df["close"].pct_change(21)
    df["log_return"]    = np.log(df["close"] / df["close"].shift(1))
    df["vol_5d"]        = df["return_1d"].rolling(5).std()  * np.sqrt(252)
    df["vol_21d"]       = df["return_1d"].rolling(21).std() * np.sqrt(252)
    df["vol_63d"]       = df["return_1d"].rolling(63).std() * np.sqrt(252)
    df["vol_ratio"]     = df["vol_5d"] / (df["vol_21d"] + 1e-8)
    df["realized_var"]  = df["return_1d"].rolling(21).var() * 252

    # ── Block 2: Technical indicators ────────────────────────────────────────
    # RSI
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi_14"] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # Moving averages
    df["sma_20"]   = df["close"].rolling(20).mean()
    df["sma_50"]   = df["close"].rolling(50).mean()
    df["sma_200"]  = df["close"].rolling(200).mean()
    df["ema_12"]   = df["close"].ewm(span=12).mean()
    df["ema_26"]   = df["close"].ewm(span=26).mean()

    # MACD
    df["macd"]         = df["ema_12"] - df["ema_26"]
    df["macd_signal"]  = df["macd"].ewm(span=9).mean()
    df["macd_hist"]    = df["macd"] - df["macd_signal"]

    # Bollinger Bands
    bb_mid         = df["close"].rolling(20).mean()
    bb_std         = df["close"].rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_pct"]   = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (bb_mid + 1e-10)

    # Price position relative to MAs
    df["price_vs_sma20"]  = (df["close"] - df["sma_20"])  / (df["sma_20"]  + 1e-10)
    df["price_vs_sma50"]  = (df["close"] - df["sma_50"])  / (df["sma_50"]  + 1e-10)
    df["price_vs_sma200"] = (df["close"] - df["sma_200"]) / (df["sma_200"] + 1e-10)

    # Momentum
    df["momentum_5d"]  = df["close"].pct_change(5)
    df["momentum_21d"] = df["close"].pct_change(21)
    df["momentum_63d"] = df["close"].pct_change(63)

    # Volume features (where available)
    if "volume" in df.columns and df["volume"].sum() > 0:
        df["volume_sma20"]  = df["volume"].rolling(20).mean()
        df["volume_ratio"]  = df["volume"] / (df["volume_sma20"] + 1e-10)
        df["dollar_volume"] = df["close"] * df["volume"]
    else:
        df["volume_ratio"]  = 1.0
        df["dollar_volume"] = df["close"]

    # ATR (Average True Range)
    if "high" in df.columns and "low" in df.columns:
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"]  - df["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()
        df["atr_pct"] = df["atr_14"] / (df["close"] + 1e-10)
    else:
        df["atr_14"]  = df["vol_21d"] * df["close"]
        df["atr_pct"] = df["vol_21d"]

    # ── Block 3: Asset-class specific features ────────────────────────────────
    if asset_class == "crypto":
        # Crypto-specific: extreme volatility regimes
        df["vol_regime_crypto"] = (df["vol_21d"] > df["vol_21d"].rolling(252).quantile(0.75)).astype(int)
        df["drawdown_from_ath"]  = df["close"] / df["close"].cummax() - 1
        df["options_iv"]         = 0.0  # placeholder — crypto IV not free
        df["put_call_ratio"]     = 1.0
    elif asset_class == "commodity":
        df["vol_regime_crypto"]  = 0
        df["drawdown_from_ath"]  = df["close"] / df["close"].cummax() - 1
        df["options_iv"]         = df["vol_21d"]  # use realized vol as proxy
        df["put_call_ratio"]     = 1.0
    else:  # bond
        df["vol_regime_crypto"]  = 0
        df["drawdown_from_ath"]  = df["close"] / df["close"].cummax() - 1
        df["options_iv"]         = df["vol_21d"]
        df["put_call_ratio"]     = 1.0

    # ── Block 4: Relative strength vs SPY ─────────────────────────────────────
    spy = _get_spy()
    spy_aligned = spy.reindex(df.set_index("date").index).fillna(0)
    asset_ret   = df.set_index("date")["return_1d"].fillna(0)

    # Beta (rolling 63-day)
    cov_  = asset_ret.rolling(63).cov(spy_aligned)
    var_  = spy_aligned.rolling(63).var()
    beta  = (cov_ / (var_ + 1e-10)).clip(-3, 3)
    df["beta"]            = beta.values
    df["excess_return_1d"] = (asset_ret - spy_aligned).values
    df["relative_strength_21d"] = (
        (1 + asset_ret).rolling(21).apply(np.prod) /
        (1 + spy_aligned).rolling(21).apply(np.prod) - 1
    ).values

    # ── Block 5: Sentiment placeholders (neutral for non-equities) ────────────
    df["sentiment_score"]        = 0.0
    df["sentiment_delta"]        = 0.0
    df["sentiment_delta_2q"]     = 0.0
    df["sentiment_acceleration"] = 0.0
    df["sentiment_vs_ma3"]       = 0.0

    # SEC filing placeholders
    df["sec_filing_count_90d"] = 0
    df["sec_filing_delay_avg"] = 0
    df["sec_8k_spike"]         = 0
    df["sec_info_staleness"]   = 999

    # Options placeholders (for assets without options data)
    if "options_iv" not in df.columns:
        df["options_iv"] = df["vol_21d"]

    # ── Block 6: Macro merge ──────────────────────────────────────────────────
    if macro_df is not None and not macro_df.empty:
        df = df.merge(
            macro_df[["date","yield_spread","vix","yield_curve_slope","real_rate",
                       "fed_funds_rate","cpi_yoy"]],
            on="date", how="left"
        )
        macro_cols = ["yield_spread","vix","yield_curve_slope","real_rate",
                      "fed_funds_rate","cpi_yoy"]
        for col in macro_cols:
            df[col] = df[col].ffill().fillna(0)
    else:
        df["yield_spread"]      = 0.0
        df["vix"]               = 20.0
        df["yield_curve_slope"] = 0.0
        df["real_rate"]         = 0.0
        df["fed_funds_rate"]    = 5.0
        df["cpi_yoy"]           = 3.0

    # Macro regime signal
    df["macro_regime"] = (
        (df["vix"] > 25).astype(int) +
        (df["yield_spread"] < 0).astype(int) * 2
    ).clip(0, 3)

    # ── Block 7: Target variable ──────────────────────────────────────────────
    df["target_return_5d"] = df["close"].pct_change(5).shift(-5)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    df["ticker"] = asset_name
    df["sector"] = MULTI_ASSETS[asset_name][1]

    # Drop raw price columns not needed by TFT
    drop_cols = ["open","high","low","volume","close_time","quote_volume",
                 "trades","taker_base","taker_quote","ignore","asset","asset_class",
                 "ema_12","ema_26","sma_20","sma_50","sma_200","bb_upper","bb_lower"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = df.dropna(subset=["return_1d"]).reset_index(drop=True)

    return df


def upload_features(df: pd.DataFrame, asset_name: str, s3_client) -> str:
    key = f"features/multi_asset/{asset_name}/{asset_name}_features.parquet"
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3_client.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())
    return f"s3://{S3_BUCKET}/{key}"


def run_multi_asset_features() -> dict:
    s3 = boto3.client("s3", region_name=AWS_REGION)

    logger.info("=" * 60)
    logger.info("AlphaFlow — Multi-Asset Feature Pipeline")
    logger.info("=" * 60)

    # Load macro once
    logger.info("Loading macro factors...")
    macro_df = load_macro(s3)

    results = {}
    for asset_name, (asset_class, sector) in MULTI_ASSETS.items():
        logger.info(f"\nBuilding features: {asset_name} ({asset_class})")

        raw_df = load_from_s3(asset_class, asset_name, s3)
        if raw_df.empty:
            continue

        features_df = build_features(raw_df, asset_name, asset_class, macro_df)
        s3_uri      = upload_features(features_df, asset_name, s3)

        results[asset_name] = {
            "rows"    : len(features_df),
            "features": len(features_df.columns),
            "s3_uri"  : s3_uri,
        }
        logger.info(f"  ✓ {asset_name}: {len(features_df)} rows | "
                    f"{len(features_df.columns)} features → {s3_uri}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Multi-Asset Feature Pipeline Complete")
    logger.info("=" * 60)
    logger.info(f"  Assets processed: {len(results)}")
    logger.info(f"  Total assets now: 25 equities + {len(results)} multi-asset = "
                f"{25 + len(results)} assets")
    logger.info("")
    logger.info("Next: update config.py TICKERS to include multi-asset,")
    logger.info("      retrain TFT on full 44-asset universe")

    return results


if __name__ == "__main__":
    results = run_multi_asset_features()