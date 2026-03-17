# models/forecasting/dataset.py
#
# AlphaFlow — TFT Dataset Preparation
#
# What this file does:
#   Loads the 70-feature Parquet files from S3 and converts them into
#   the exact format pytorch-forecasting's TemporalFusionTransformer expects.
#
# The THREE types of inputs TFT needs:
#
#   1. TIME-VARYING KNOWN (future)
#      Things we know in advance for any future date
#      e.g. day_of_week, month, is_earnings_week
#      TFT uses these to "look ahead" in its encoder
#
#   2. TIME-VARYING UNKNOWN (past only)
#      Things we only know after they happen
#      e.g. close price, RSI, sentiment_score, volume
#      TFT learns patterns from these historically
#
#   3. STATIC (per ticker, never changes)
#      e.g. ticker identity, sector
#      TFT uses these to personalise its attention per asset
#
# Why this split matters:
#   A model that doesn't know WHAT it can see in the future vs past
#   will cheat during training (look-ahead bias) and fail in production.
#   This is the #1 mistake in financial ML. We prevent it explicitly here.
#
# Run: python models/forecasting/dataset.py

import warnings

warnings.filterwarnings("ignore")

import logging  # noqa: E402
import boto3  # noqa: E402
import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
from pathlib import Path  # noqa: E402
from io import BytesIO  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402

import sys  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import (  # noqa: E402
    AWS_REGION,
    S3_BUCKET,
    TICKERS,
    S3_FEATURES_PREFIX,  # noqa: F401
    LOCAL_DATA_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/dataset.log"),
    ],
)
logger = logging.getLogger("dataset")

s3 = boto3.client("s3", region_name=AWS_REGION)


# ═══════════════════════════════════════════════════════════════════════
# FEATURE GROUPS
# Explicitly defining what the model can/cannot see about the future
# ═══════════════════════════════════════════════════════════════════════

# Things we know in advance for any future date
TIME_VARYING_KNOWN = [
    "day_of_week",  # 0-4 (Mon-Fri)
    "month",  # 1-12
    "quarter",  # 1-4
    "is_month_end",  # earnings tend to cluster here
    "is_quarter_end",  # big rebalancing happens
    "days_to_month_end",  # countdown signal
]

# Things we only know after they happen — learned from history
TIME_VARYING_UNKNOWN = [
    # Price-derived
    "close",
    "volume",
    "return_1d",
    "return_5d",
    "return_21d",
    "vol_21d",
    "vol_regime",
    # Technical (confirmation)
    "rsi_14",
    "macd",
    "macd_hist",
    "bb_pct",
    "bb_width",
    "volume_ratio",
    "above_sma50",
    "above_sma200",
    "golden_cross",
    # Primary signals
    "sentiment_score",
    "sentiment_velocity",
    "sentiment_accel",
    "sentiment_divergence",
    "sec_8k_count_90d",
    "sec_filing_spike",
    "iv_atm_proxy",
    "put_call_ratio",
    "iv_premium",
    "rs_vs_spy_21d",
    "beta_63d",
    "excess_return_21d",
    # Macro regime
    "vix",
    "yield_curve",
    "yield_curve_inverted",
    "vix_regime",
    "macro_regime",
    "risk_off",
]

# Per-ticker metadata — never changes
STATIC_CATEGORICALS = ["ticker", "sector"]

# What we're predicting
TARGET = "target_return_5d"

# Model hyperparameters
MAX_ENCODER_LENGTH = 63  # look back 63 trading days (~3 months)
MAX_PREDICTION_LENGTH = 5  # predict 5 days ahead


# ═══════════════════════════════════════════════════════════════════════
# SECTOR MAP — static metadata per ticker
# ═══════════════════════════════════════════════════════════════════════

TICKER_SECTOR = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Communication",
    "AMZN": "Consumer",
    "JPM": "Finance",
    "JNJ": "Healthcare",
    "XOM": "Energy",
    "BRK-B": "Diversified",
    "TSLA": "Technology",
    "SPY": "Benchmark",
}


# ═══════════════════════════════════════════════════════════════════════
# LOAD & MERGE
# ═══════════════════════════════════════════════════════════════════════


def load_features_from_s3(tickers: list) -> pd.DataFrame:
    """
    Load feature Parquet files for all tickers from S3.
    Tries sentiment-enriched path first, falls back to market features.
    """
    frames = []
    for ticker in tickers:
        df = None
        for s3_key in [
            f"features/sentiment_enriched/{ticker}/{ticker}_features_v2.parquet",
            f"features/market/{ticker}/{ticker}_features.parquet",
            f"features/multi_asset/{ticker}/{ticker}_features.parquet",
        ]:
            try:
                obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
                df = pd.read_parquet(BytesIO(obj["Body"].read()))
                df["ticker"] = ticker
                df["sector"] = TICKER_SECTOR.get(ticker, "Unknown")
                frames.append(df)
                logger.info(f"  ✓ Loaded {ticker}: {len(df)} rows")
                break
            except Exception:
                continue
        if df is None:
            logger.error(f"  ✗ Could not load {ticker}: not found in S3")

    if not frames:
        raise RuntimeError("No feature files loaded — run feature_pipeline.py first")

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    logger.info(
        f"\n  ✓ Combined dataset: {len(combined)} rows | {len(combined.columns)} cols"
    )
    return combined


# ═══════════════════════════════════════════════════════════════════════
# CALENDAR FEATURES — time-varying known
# ═══════════════════════════════════════════════════════════════════════


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar features that are known in advance for any future date.
    These give TFT's decoder information about the future time steps.
    """
    df = df.copy()
    dt = df["date"]

    df["day_of_week"] = dt.dt.dayofweek.astype(float)
    df["month"] = dt.dt.month.astype(float)
    df["quarter"] = dt.dt.quarter.astype(float)
    df["is_month_end"] = dt.dt.is_month_end.astype(float)
    df["is_quarter_end"] = dt.dt.is_quarter_end.astype(float)

    # Days until month end — counts down like an earnings anticipation signal
    import calendar

    def days_to_month_end(date):
        last_day = calendar.monthrange(date.year, date.month)[1]
        return float(last_day - date.day)

    df["days_to_month_end"] = dt.apply(days_to_month_end)

    logger.info("  ✓ Calendar features added")
    return df


# ═══════════════════════════════════════════════════════════════════════
# CLEAN & VALIDATE
# ═══════════════════════════════════════════════════════════════════════


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final cleaning before feeding into TFT.

    Key steps:
    1. Drop rows where target is NaN (last 5 rows per ticker — correct)
    2. Fill remaining NaNs with forward fill then median
    3. Add time_idx — TFT needs an integer time index per group
    4. Encode categorical columns as integers
    5. Clip extreme outliers (>5 std) to prevent gradient explosion
    """
    df = df.copy()

    # ── Drop rows with no target ──────────────────────────────────────
    # Last 5 rows of each ticker have NaN target (no future yet) — correct
    before = len(df)
    df = df.dropna(subset=[TARGET])
    logger.info(f"  ✓ Dropped {before - len(df)} rows with NaN target")

    # ── Only keep columns we actually need ────────────────────────────
    all_feature_cols = (
        TIME_VARYING_KNOWN
        + TIME_VARYING_UNKNOWN
        + STATIC_CATEGORICALS
        + [TARGET, "date"]
    )
    # Keep only columns that exist in df
    available = [c for c in all_feature_cols if c in df.columns]
    missing = [c for c in all_feature_cols if c not in df.columns]
    if missing:
        logger.warning(f"  ⚠ Missing columns (will skip): {missing}")
        # Add missing columns as zeros so model still works
        for col in missing:
            if col not in STATIC_CATEGORICALS:
                df[col] = 0.0

    df = (
        df[available + ["ticker"]].copy()
        if "ticker" not in available
        else df[available].copy()
    )

    # ── Forward fill then median fill NaNs ────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df.groupby("ticker")[numeric_cols].transform(
        lambda x: x.ffill().bfill()
    )
    # Any remaining NaNs → column median
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # ── Clip extreme outliers per numeric column ──────────────────────
    # Prevents one crazy day (e.g. COVID crash) from dominating training
    for col in numeric_cols:
        if col in [TARGET]:
            continue  # don't clip the target
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            df[col] = df[col].clip(mean - 5 * std, mean + 5 * std)

    # ── Time index — required by pytorch-forecasting ──────────────────
    # Must be a monotonically increasing integer per ticker
    df = df.sort_values(["ticker", "date"])
    df["time_idx"] = df.groupby("ticker").cumcount()

    # ── Encode categoricals as integers ──────────────────────────────
    for col in STATIC_CATEGORICALS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            df[col] = df[col].astype(str)  # TFT wants string categoricals

    logger.info(f"  ✓ Clean dataset: {len(df)} rows | {len(df.columns)} columns")
    logger.info(f"  ✓ Date range: {df['date'].min()} → {df['date'].max()}")
    logger.info(
        f"  ✓ Tickers: {df['ticker'].nunique() if 'ticker' in df.columns else 'N/A'}"
    )

    return df


# ═══════════════════════════════════════════════════════════════════════
# TRAIN / VAL SPLIT — time-based, never random
# ═══════════════════════════════════════════════════════════════════════


def time_based_split(df: pd.DataFrame, val_ratio: float = 0.15):
    """
    Split data into train and validation sets by TIME, not randomly.

    WHY THIS MATTERS:
    Random splits cause look-ahead bias — the model sees future data
    during training and appears to perform well but fails in production.
    ALWAYS split financial time series by date.

    Split:
    - Train: 2020-01-01 → 2023-08-31  (~85%)
    - Val:   2023-09-01 → 2024-12-31  (~15%)

    We hold out the most recent data for validation — this simulates
    what happens when you deploy: the model has never seen these dates.
    """
    df = df.sort_values("date")
    cutoff_idx = int(len(df) * (1 - val_ratio))
    cutoff_date = df.iloc[cutoff_idx]["date"]

    train = df[df["date"] < cutoff_date].copy()
    val = df[df["date"] >= cutoff_date].copy()

    logger.info(
        f"  ✓ Train: {len(train)} rows | {train['date'].min()} → {train['date'].max()}"
    )
    logger.info(
        f"  ✓ Val:   {len(val)} rows   | {val['date'].min()} → {val['date'].max()}"
    )

    return train, val


# ═══════════════════════════════════════════════════════════════════════
# BUILD PYTORCH-FORECASTING DATASETS
# ═══════════════════════════════════════════════════════════════════════


def build_timeseries_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """
    Convert pandas DataFrames into pytorch-forecasting TimeSeriesDataSet objects.

    TimeSeriesDataSet handles:
    - Batching sequences of length MAX_ENCODER_LENGTH + MAX_PREDICTION_LENGTH
    - Normalising each feature per group (per ticker)
    - Correctly routing features to encoder vs decoder
    """
    try:
        from pytorch_forecasting import TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer
    except ImportError:
        raise ImportError("Run: pip install pytorch-forecasting")

    # Identify which unknown columns actually exist in the data
    available_unknown = [c for c in TIME_VARYING_UNKNOWN if c in train_df.columns]
    available_known = [c for c in TIME_VARYING_KNOWN if c in train_df.columns]
    available_static = [c for c in STATIC_CATEGORICALS if c in train_df.columns]

    logger.info("\n Building TimeSeriesDataSet...")
    logger.info(f"  Known future features : {len(available_known)}")
    logger.info(f"  Unknown past features : {len(available_unknown)}")
    logger.info(f"  Static categoricals   : {len(available_static)}")
    logger.info(f"  Target                : {TARGET}")
    logger.info(f"  Encoder length        : {MAX_ENCODER_LENGTH} days")
    logger.info(f"  Prediction length     : {MAX_PREDICTION_LENGTH} days")

    training_dataset = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=TARGET,
        group_ids=["ticker"],
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=available_static,
        time_varying_known_reals=available_known,
        time_varying_unknown_reals=available_unknown,
        target_normalizer=GroupNormalizer(groups=["ticker"], transformation="softplus"),
        add_relative_time_idx=True,  # adds normalised time position
        add_target_scales=True,  # adds scale info to static features
        add_encoder_length=True,  # adds encoder length as feature
        allow_missing_timesteps=True,  # handles weekends/holidays gracefully
    )

    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        val_df,
        predict=False,
        stop_randomization=True,
    )

    logger.info(f"  ✓ Training dataset  : {len(training_dataset)} samples")
    logger.info(f"  ✓ Validation dataset: {len(validation_dataset)} samples")

    return training_dataset, validation_dataset


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════


def prepare_datasets(tickers: list = TICKERS, save_locally: bool = True):
    """
    Full dataset preparation pipeline.
    Returns training and validation TimeSeriesDataSet objects.
    """
    logger.info("=" * 60)
    logger.info("AlphaFlow — TFT Dataset Preparation")
    logger.info("=" * 60)

    Path("logs").mkdir(exist_ok=True)

    # Step 1: Load features from S3
    logger.info("\nStep 1: Loading features from S3...")
    df = load_features_from_s3(tickers)

    # Step 2: Add calendar features
    logger.info("\nStep 2: Adding calendar features...")
    df = add_calendar_features(df)

    # Step 3: Clean and validate
    logger.info("\nStep 3: Cleaning dataset...")
    df = clean_dataset(df)

    # Step 4: Time-based split
    logger.info("\nStep 4: Time-based train/val split...")
    train_df, val_df = time_based_split(df)

    # Save locally for inspection
    if save_locally:
        local_dir = Path(LOCAL_DATA_DIR) / "model_input"
        local_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(local_dir / "train.parquet", index=False)
        val_df.to_parquet(local_dir / "val.parquet", index=False)
        logger.info(f"\n  ✓ Saved train/val locally to {local_dir}")

    # Step 5: Build TimeSeriesDataSets
    logger.info("\nStep 5: Building TimeSeriesDataSets...")
    training_dataset, validation_dataset = build_timeseries_datasets(train_df, val_df)

    logger.info("\n" + "=" * 60)
    logger.info("DATASET PREPARATION COMPLETE")
    logger.info("=" * 60)

    return training_dataset, validation_dataset, train_df, val_df


if __name__ == "__main__":
    training_dataset, validation_dataset, train_df, val_df = prepare_datasets()

    print("\n── Dataset Summary ──────────────────────────────")
    print(f"  Train samples : {len(training_dataset)}")
    print(f"  Val samples   : {len(validation_dataset)}")
    print(f"  Features used : {len(training_dataset.reals)} real features")
    print(f"  Target        : {TARGET}")
    print(f"  Encoder len   : {MAX_ENCODER_LENGTH} days")
    print(f"  Predict len   : {MAX_PREDICTION_LENGTH} days")
    print("\nDataset ready. Run tft_model.py to train.")
