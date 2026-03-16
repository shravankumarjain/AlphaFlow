"""
models/regime/hmm_regime_detector.py
AlphaFlow — Hidden Markov Model Market Regime Detector
Classifies market into 4 regimes: Bull, Bear, Volatile-Sideways, Crisis
Feeds regime signal into portfolio optimizer to switch allocation strategy.
"""

import logging
import warnings

warnings.filterwarnings("ignore")

import boto3  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402
import mlflow  # noqa: E402
import json  # noqa: E402
import io  # noqa: E402, F401
import os  # noqa: E402
from pathlib import Path  # noqa: E402
from datetime import datetime  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from hmmlearn import hmm  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
import joblib  # noqa: E402

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(module)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("hmm_regime")

S3_BUCKET = os.getenv("S3_BUCKET", "alphaflow-data-lake-291572330987")
AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")
OUTPUT_DIR = Path("data/local/regime")
MODEL_DIR = Path("models/regime")

# ── Regime labels (assigned after fitting based on return/vol profile) ────────
REGIME_NAMES = {0: "bull", 1: "bear", 2: "volatile", 3: "crisis"}
REGIME_COLORS = {
    "bull": "#2ecc71",
    "bear": "#e74c3c",
    "volatile": "#f39c12",
    "crisis": "#8e44ad",
}

# ── Optimizer blend per regime (how much to trust RL vs Markowitz) ────────────
REGIME_BLEND = {
    "bull": {"rl": 0.70, "markowitz": 0.30},  # trust momentum in bull
    "bear": {"rl": 0.20, "markowitz": 0.80},  # trust variance minimization in bear
    "volatile": {"rl": 0.40, "markowitz": 0.60},  # balanced in volatile
    "crisis": {"rl": 0.10, "markowitz": 0.90},  # max defensiveness in crisis
}

# ── Max position sizes per regime ────────────────────────────────────────────
REGIME_CONSTRAINTS = {
    "bull": {"max_weight": 0.35, "min_weight": 0.02, "cash_floor": 0.00},
    "bear": {"max_weight": 0.25, "min_weight": 0.02, "cash_floor": 0.10},
    "volatile": {"max_weight": 0.30, "min_weight": 0.02, "cash_floor": 0.05},
    "crisis": {"max_weight": 0.20, "min_weight": 0.02, "cash_floor": 0.20},
}


def build_regime_features(start_date="2018-01-01", end_date=None) -> pd.DataFrame:
    """
    Build macro feature matrix for HMM.
    Uses SPY + VIX + yield curve as regime signals — same inputs Aladdin uses.
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    logger.info("  Downloading market regime features...")

    # SPY for equity returns and realized vol
    spy = yf.download(
        "SPY", start=start_date, end=end_date, auto_adjust=True, progress=False
    )
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    spy_ret = spy["Close"].pct_change()
    spy_vol = spy_ret.rolling(21).std() * np.sqrt(252)  # annualized 21d vol
    spy_ma50 = spy["Close"].rolling(50).mean()  # noqa: F841
    spy_ma200 = spy["Close"].rolling(200).mean()
    spy_trend = (spy["Close"] - spy_ma200) / spy_ma200  # % above/below 200MA

    # VIX — market's fear gauge
    vix = yf.download(
        "^VIX", start=start_date, end=end_date, auto_adjust=True, progress=False
    )
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    vix_close = vix["Close"].rename("vix")
    vix_change = vix["Close"].pct_change().rename("vix_change")

    # Yield curve — 10Y minus 2Y spread (inversion = recession signal)
    try:
        t10y = yf.download(
            "^TNX", start=start_date, end=end_date, auto_adjust=True, progress=False
        )
        t2y = yf.download(
            "^IRX", start=start_date, end=end_date, auto_adjust=True, progress=False
        )
        if isinstance(t10y.columns, pd.MultiIndex):
            t10y.columns = t10y.columns.get_level_values(0)
        if isinstance(t2y.columns, pd.MultiIndex):
            t2y.columns = t2y.columns.get_level_values(0)
        yield_spread = (t10y["Close"] - t2y["Close"] / 100).rename("yield_spread")
    except Exception:
        yield_spread = pd.Series(0, index=spy.index, name="yield_spread")

    # Combine
    df = pd.DataFrame(
        {
            "spy_return": spy_ret,
            "spy_vol_21d": spy_vol,
            "spy_trend": spy_trend,
            "vix": vix_close,
            "vix_change": vix_change,
            "yield_spread": yield_spread,
        }
    ).dropna()

    # Rolling momentum signals
    df["spy_momentum_5d"] = spy["Close"].pct_change(5).reindex(df.index)
    df["spy_momentum_21d"] = spy["Close"].pct_change(21).reindex(df.index)
    df["vol_regime"] = (
        df["spy_vol_21d"] > df["spy_vol_21d"].rolling(126).mean()
    ).astype(int)

    df = df.dropna()
    logger.info(f"  ✓ Regime features: {len(df)} days | {df.columns.tolist()}")
    return df


def fit_hmm(features_df: pd.DataFrame, n_states: int = 4) -> tuple:
    """
    Fit a Gaussian HMM with n_states hidden states.
    Returns: (model, scaler, state_sequence)
    """
    feature_cols = [
        "spy_return",
        "spy_vol_21d",
        "spy_trend",
        "vix",
        "vix_change",
        "spy_momentum_21d",
    ]
    X = features_df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info(f"  Fitting HMM with {n_states} states on {len(X)} observations...")

    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=42,
        tol=1e-4,
    )
    model.fit(X_scaled)

    states = model.predict(X_scaled)
    logger.info(f"  ✓ HMM fitted | Log-likelihood: {model.score(X_scaled):.2f}")
    return model, scaler, states


def label_regimes(features_df: pd.DataFrame, states: np.ndarray) -> dict:
    """
    Assign meaningful labels to HMM states based on their return/vol profile.
    State with highest return = bull
    State with lowest return  = bear
    State with highest vol    = crisis
    Remaining state           = volatile-sideways
    """
    features_df = features_df.copy()
    features_df["state"] = states

    stats = features_df.groupby("state").agg(
        mean_return=("spy_return", "mean"),
        mean_vol=("spy_vol_21d", "mean"),
        mean_vix=("vix", "mean"),
        count=("spy_return", "count"),
    )

    # Assign labels
    bull_state = stats["mean_return"].idxmax()
    crisis_state = stats["mean_vix"].idxmax()
    bear_state = stats[stats.index != crisis_state]["mean_return"].idxmin()
    volatile_state = [
        s for s in stats.index if s not in [bull_state, bear_state, crisis_state]
    ][0]

    state_map = {
        bull_state: "bull",
        bear_state: "bear",
        volatile_state: "volatile",
        crisis_state: "crisis",
    }

    logger.info("  Regime profiles:")
    for state, label in state_map.items():
        s = stats.loc[state]
        logger.info(
            f"    State {state} → {label:10s} | "
            f"return={s['mean_return'] * 100:+.3f}%/day | "
            f"vol={s['mean_vol']:.3f} | vix={s['mean_vix']:.1f} | "
            f"n={int(s['count'])} days"
        )

    return state_map


def get_current_regime(
    model: hmm.GaussianHMM,
    scaler: StandardScaler,
    state_map: dict,
    features_df: pd.DataFrame,  # pass the already-built features
) -> dict:
    feature_cols = [
        "spy_return",
        "spy_vol_21d",
        "spy_trend",
        "vix",
        "vix_change",
        "spy_momentum_21d",
    ]
    # Use last 63 rows of the full features we already have
    X = features_df[feature_cols].values[-63:]
    X_scaled = scaler.transform(X)
    states = model.predict(X_scaled)
    posteriors = model.predict_proba(X_scaled)

    current_state = states[-1]
    current_label = state_map.get(current_state, "unknown")
    confidence = float(posteriors[-1, current_state])

    trans_probs = {
        state_map.get(s, f"state_{s}"): float(model.transmat_[current_state, s])
        for s in range(model.n_components)
    }
    recent_states = states[-21:]
    regime_counts = {
        label: int((recent_states == s).sum()) for s, label in state_map.items()
    }

    return {
        "current_regime": current_label,
        "confidence": confidence,
        "transition_probs": trans_probs,
        "recent_21d_dist": regime_counts,
        "blend": REGIME_BLEND[current_label],
        "constraints": REGIME_CONSTRAINTS[current_label],
        "detected_at": datetime.utcnow().isoformat(),
    }


def build_historical_regime_series(
    features_df: pd.DataFrame,
    states: np.ndarray,
    state_map: dict,
) -> pd.Series:
    """Build a daily regime label series for the full history."""
    labels = pd.Series(
        [state_map.get(s, "unknown") for s in states],
        index=features_df.index,
        name="regime",
    )
    return labels


def save_regime_model(model, scaler, state_map, regime_series, s3_client):
    """Save HMM model, scaler, and regime series locally and to S3."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save model + scaler
    model_path = MODEL_DIR / "hmm_model.pkl"
    scaler_path = MODEL_DIR / "hmm_scaler.pkl"
    map_path = MODEL_DIR / "state_map.json"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    with open(map_path, "w") as f:
        json.dump({str(k): v for k, v in state_map.items()}, f)

    # Save regime series
    regime_path = OUTPUT_DIR / "historical_regimes.parquet"
    regime_series.to_frame().reset_index().to_parquet(regime_path, index=False)

    # Upload to S3
    for local_path, s3_key in [
        (model_path, "models/regime/hmm_model.pkl"),
        (scaler_path, "models/regime/hmm_scaler.pkl"),
        (map_path, "models/regime/state_map.json"),
        (regime_path, "models/regime/historical_regimes.parquet"),
    ]:
        s3_client.upload_file(str(local_path), S3_BUCKET, s3_key)

    logger.info("  ✓ Model saved locally and uploaded to S3")


def run_regime_detection() -> dict:
    """Full pipeline: build features → fit HMM → label → get current regime."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3", region_name=AWS_REGION)

    logger.info("=" * 60)
    logger.info("AlphaFlow — HMM Regime Detector")
    logger.info("=" * 60)

    # ── Step 1: Features ──────────────────────────────────────────────────────
    logger.info("\nStep 1: Building regime features (2018–present)...")
    features_df = build_regime_features(start_date="2018-01-01")

    # ── Step 2: Fit HMM ───────────────────────────────────────────────────────
    logger.info("\nStep 2: Fitting 4-state Gaussian HMM...")
    model, scaler, states = fit_hmm(features_df, n_states=4)

    # ── Step 3: Label regimes ─────────────────────────────────────────────────
    logger.info("\nStep 3: Labelling regimes...")
    state_map = label_regimes(features_df, states)

    # ── Step 4: Historical series ─────────────────────────────────────────────
    regime_series = build_historical_regime_series(features_df, states, state_map)
    regime_counts = regime_series.value_counts()
    logger.info("\n  Historical regime distribution:")
    for regime, count in regime_counts.items():
        pct = count / len(regime_series) * 100
        bar = "█" * int(pct / 2)
        logger.info(f"    {regime:12s} {bar} {pct:.1f}% ({count} days)")

    # ── Step 5: Current regime ────────────────────────────────────────────────
    logger.info("\nStep 4: Detecting current regime...")
    current = get_current_regime(model, scaler, state_map, features_df)

    logger.info("")
    logger.info("=" * 60)
    logger.info("CURRENT MARKET REGIME")
    logger.info("=" * 60)
    logger.info(f"  Regime     : {current['current_regime'].upper()}")
    logger.info(f"  Confidence : {current['confidence']:.1%}")
    logger.info(
        f"  Blend      : {current['blend']['rl']:.0%} RL + {current['blend']['markowitz']:.0%} Markowitz"
    )
    logger.info(f"  Max weight : {current['constraints']['max_weight']:.0%}")
    logger.info(f"  Cash floor : {current['constraints']['cash_floor']:.0%}")
    logger.info(f"\n  Transition probs from {current['current_regime']}:")
    for regime, prob in current["transition_probs"].items():
        logger.info(f"    → {regime:12s}: {prob:.1%}")
    logger.info("\n  Last 21 days:")
    for regime, count in current["recent_21d_dist"].items():
        logger.info(f"    {regime:12s}: {count} days")

    # ── Step 6: Save ──────────────────────────────────────────────────────────
    logger.info("\nStep 5: Saving model to S3...")
    save_regime_model(model, scaler, state_map, regime_series, s3)

    # Save current regime JSON for portfolio optimizer to read
    current_path = OUTPUT_DIR / "current_regime.json"
    with open(current_path, "w") as f:
        json.dump(current, f, indent=2)
    s3.upload_file(str(current_path), S3_BUCKET, "models/regime/current_regime.json")
    logger.info(f"  ✓ Current regime saved: {current_path}")

    # ── Step 7: MLflow ────────────────────────────────────────────────────────
    try:
        mlflow.set_tracking_uri("mlruns")
        with mlflow.start_run(run_name="hmm-regime-v1"):
            mlflow.log_param("n_states", 4)
            mlflow.log_param("current_regime", current["current_regime"])
            mlflow.log_metric(
                "hmm_log_likelihood",
                model.score(
                    scaler.transform(
                        features_df[
                            [
                                "spy_return",
                                "spy_vol_21d",
                                "spy_trend",
                                "vix",
                                "vix_change",
                                "spy_momentum_21d",
                            ]
                        ].values
                    )
                ),
            )
            mlflow.log_metric("regime_confidence", current["confidence"])
            for regime, count in regime_counts.items():
                mlflow.log_metric(f"regime_pct_{regime}", count / len(regime_series))
        logger.info("  ✓ Logged to MLflow")
    except Exception as e:
        logger.warning(f"  ⚠ MLflow skipped: {e}")

    return {
        "model": model,
        "scaler": scaler,
        "state_map": state_map,
        "regime_series": regime_series,
        "current": current,
    }


if __name__ == "__main__":
    result = run_regime_detection()
