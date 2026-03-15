# mlops/monitoring/drift_monitor.py
#
# AlphaFlow — Data & Model Drift Monitor
#
# What this does:
#   1. Loads reference data (training set) and current data (last 30 days)
#   2. Runs Evidently AI drift detection on all 65 features
#   3. Flags features that have drifted significantly
#   4. Saves drift report to reports/drift_report.json
#   5. Logs drift metrics to MLflow
#
# Why drift monitoring matters for a pitch:
#   "The model was trained on 2020-2024 data. If market regimes change
#    (e.g. interest rate environment shifts), feature distributions
#    change and the model becomes stale. We detect this automatically
#    and trigger retraining — this is production-grade MLOps."
#
# Run: python mlops/monitoring/drift_monitor.py

import warnings
warnings.filterwarnings("ignore")

import logging  # noqa: E402
import json  # noqa: E402
import mlflow  # noqa: E402
import boto3  # noqa: E402
import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402
from pathlib import Path  # noqa: E402
from io import BytesIO  # noqa: E402

import sys  # noqa: E402
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import AWS_REGION, S3_BUCKET, TICKERS, S3_FEATURES_PREFIX  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/drift_monitor.log"),
    ]
)
logger = logging.getLogger("drift_monitor")

s3 = boto3.client("s3", region_name=AWS_REGION)


# ── LOAD DATA ─────────────────────────────────────────────────────────

def load_reference_data() -> pd.DataFrame:
    """Load training set as reference distribution."""
    ref_path = Path("data/local/model_input/train.parquet")
    if not ref_path.exists():
        raise FileNotFoundError("Run dataset.py first to generate train.parquet")
    df = pd.read_parquet(ref_path)
    logger.info(f"  ✓ Reference data: {len(df)} rows | {df['date'].min()} → {df['date'].max()}")
    return df


def load_current_data(lookback_days: int = 30) -> pd.DataFrame:
    """Load most recent N days of feature data from S3."""
    frames = []
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)

    for ticker in TICKERS:
        s3_key = f"{S3_FEATURES_PREFIX}/market/{ticker}/{ticker}_features.parquet"
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
            df  = pd.read_parquet(BytesIO(obj["Body"].read()))
            df["date"] = pd.to_datetime(df["date"])
            df  = df[df["date"] >= cutoff]
            frames.append(df)
        except Exception as e:
            logger.warning(f"  ⚠ Could not load {ticker}: {e}")

    if not frames:
        raise RuntimeError("No current data loaded")

    current = pd.concat(frames, ignore_index=True)
    logger.info(f"  ✓ Current data: {len(current)} rows | last {lookback_days} days")
    return current


# ── DRIFT DETECTION ───────────────────────────────────────────────────

def compute_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> dict:
    """
    Compute drift metrics for each numeric feature.

    Method: Population Stability Index (PSI)
    PSI < 0.1  = no drift
    PSI 0.1-0.2 = moderate drift (monitor)
    PSI > 0.2  = significant drift (retrain)

    We also try Evidently if available, fall back to PSI if not.
    """
    logger.info("  Computing drift metrics...")

    # Get numeric feature columns (exclude metadata)
    exclude = ["date", "ticker", "sector", "time_idx", "target_return_5d",
               "ingested_at", "source", "dq_issues"]
    feature_cols = [c for c in reference.columns
                    if c not in exclude
                    and pd.api.types.is_numeric_dtype(reference[c])]

    # Only use columns present in both
    feature_cols = [c for c in feature_cols if c in current.columns]

    # Try Evidently first
    try:
        return _evidently_drift(reference, current, feature_cols)
    except Exception as e:
        logger.warning(f"  ⚠ Evidently not available ({e}) — using PSI fallback")
        return _psi_drift(reference, current, feature_cols)


def _evidently_drift(ref: pd.DataFrame, cur: pd.DataFrame, cols: list) -> dict:
    """Use Evidently AI for drift detection."""
    from evidently.report import Report # type: ignore
    from evidently.metric_preset import DataDriftPreset # type: ignore
    from evidently.pipeline.column_mapping import ColumnMapping # type: ignore

    # Sample to avoid memory issues
    ref_sample = ref[cols].dropna().sample(min(2000, len(ref)), random_state=42)
    cur_sample = cur[cols].dropna().sample(min(500, len(cur)), random_state=42)

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data = ref_sample,
        current_data   = cur_sample,
        column_mapping = ColumnMapping(),
    )

    result = report.as_dict()

    # Extract drift summary
    drift_metrics = result["metrics"][0]["result"]
    n_drifted     = drift_metrics.get("number_of_drifted_columns", 0)
    n_total       = drift_metrics.get("number_of_columns", len(cols))
    drift_score   = n_drifted / n_total if n_total > 0 else 0

    # Per-feature drift
    feature_drift = {}
    for col_name, col_data in drift_metrics.get("drift_by_columns", {}).items():
        feature_drift[col_name] = {
            "drift_detected": col_data.get("drift_detected", False),
            "drift_score"   : col_data.get("drift_score", 0),
            "stat_test"     : col_data.get("stattest_name", "unknown"),
        }

    # Save HTML report
    Path("reports").mkdir(exist_ok=True)
    report.save_html("reports/evidently_drift_report.html")
    logger.info("  ✓ Evidently HTML report saved: reports/evidently_drift_report.html")

    return {
        "method"         : "evidently",
        "drift_score"    : drift_score,
        "n_drifted"      : n_drifted,
        "n_total"        : n_total,
        "feature_drift"  : feature_drift,
        "last_trained"   : _get_last_trained(),
        "computed_at"    : datetime.utcnow().isoformat(),
    }


def _psi_drift(ref: pd.DataFrame, cur: pd.DataFrame, cols: list) -> dict:
    """
    PSI-based drift detection as fallback.
    PSI = sum((actual% - expected%) * ln(actual% / expected%))
    """
    feature_drift = {}
    drifted_count = 0

    for col in cols:
        try:
            ref_vals = ref[col].dropna().values
            cur_vals = cur[col].dropna().values

            if len(ref_vals) < 10 or len(cur_vals) < 10:
                continue

            # Create bins from reference distribution
            bins = np.percentile(ref_vals, np.linspace(0, 100, 11))
            bins = np.unique(bins)
            if len(bins) < 3:
                continue

            ref_counts = np.histogram(ref_vals, bins=bins)[0]
            cur_counts = np.histogram(cur_vals, bins=bins)[0]

            # Avoid division by zero
            ref_pct = ref_counts / ref_counts.sum()
            cur_pct = cur_counts / cur_counts.sum()
            ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
            cur_pct = np.where(cur_pct == 0, 0.0001, cur_pct)

            psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
            drifted = psi > 0.2

            feature_drift[col] = {
                "drift_detected": drifted,
                "drift_score"   : round(psi, 4),
                "stat_test"     : "psi",
            }
            if drifted:
                drifted_count += 1

        except Exception:
            continue

    n_total    = len(feature_drift)
    drift_score = drifted_count / n_total if n_total > 0 else 0

    return {
        "method"       : "psi",
        "drift_score"  : round(drift_score, 4),
        "n_drifted"    : drifted_count,
        "n_total"      : n_total,
        "feature_drift": feature_drift,
        "last_trained" : _get_last_trained(),
        "computed_at"  : datetime.utcnow().isoformat(),
    }


def _get_last_trained() -> str:
    """Get timestamp of last model training from checkpoint."""
    checkpoints = list(Path("models/checkpoints").glob("*.ckpt"))
    if not checkpoints:
        return "2020-01-01"
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    ts = datetime.fromtimestamp(latest.stat().st_mtime)
    return ts.isoformat()


# ── MODEL PERFORMANCE DRIFT ───────────────────────────────────────────

def check_performance_drift() -> dict:
    """
    Check if model performance has degraded since last evaluation.
    Compare recent Sharpe ratio vs historical baseline.
    """
    metrics_path = Path("reports/metrics.json")
    if not metrics_path.exists():
        return {"performance_drift": False, "reason": "no metrics found"}

    with open(metrics_path) as f:
        metrics = json.load(f)

    alphaflow = next((m for m in metrics if m["name"] == "AlphaFlow TFT"), None)
    if not alphaflow:
        return {"performance_drift": False, "reason": "no AlphaFlow metrics"}

    current_sharpe = alphaflow.get("sharpe_ratio", 0)
    SHARPE_FLOOR   = 0.0  # alert if Sharpe drops below 0

    performance_drift = current_sharpe < SHARPE_FLOOR
    return {
        "performance_drift": performance_drift,
        "current_sharpe"   : current_sharpe,
        "sharpe_floor"     : SHARPE_FLOOR,
        "action"           : "retrain" if performance_drift else "monitor",
    }


# ── MAIN ──────────────────────────────────────────────────────────────

def run_drift_monitor():
    logger.info("=" * 60)
    logger.info("AlphaFlow — Drift Monitor")
    logger.info("=" * 60)

    Path("logs").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    # Step 1: Load data
    logger.info("\nStep 1: Loading data...")
    reference = load_reference_data()
    current   = load_current_data(lookback_days=365)

    # Step 2: Data drift
    logger.info("\nStep 2: Computing data drift...")
    drift_report = compute_drift_report(reference, current)

    # Step 3: Performance drift
    logger.info("\nStep 3: Checking performance drift...")
    perf_drift = check_performance_drift()
    drift_report["performance"] = perf_drift

    # Step 4: Save report
    output_path = "reports/drift_report.json"
    with open(output_path, "w") as f:
        json.dump(drift_report, f, indent=2, default=str)

    # Step 5: Log to MLflow
    mlflow.set_tracking_uri("mlruns")
    with mlflow.start_run(run_name="drift-monitor"):
        mlflow.log_metric("drift_score",    drift_report["drift_score"])
        mlflow.log_metric("n_drifted",      drift_report["n_drifted"])
        mlflow.log_metric("n_total",        drift_report["n_total"])
        mlflow.log_metric("current_sharpe", perf_drift.get("current_sharpe", 0))
        mlflow.log_artifact(output_path)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DRIFT MONITOR SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Method       : {drift_report['method']}")
    logger.info(f"  Drift score  : {drift_report['drift_score']:.3f}")
    logger.info(f"  Drifted      : {drift_report['n_drifted']}/{drift_report['n_total']} features")
    logger.info(f"  Last trained : {drift_report['last_trained']}")
    logger.info(f"  Perf drift   : {perf_drift['performance_drift']}")

    action = "RETRAIN NEEDED" if drift_report["drift_score"] > 0.3 else "NO ACTION"
    logger.info(f"\n  → {action}")
    logger.info(f"  ✓ Report saved: {output_path}")

    return drift_report


if __name__ == "__main__":
    report = run_drift_monitor()
    print("\n── Drift Report ─────────────────────────────")
    print(f"  Score    : {report['drift_score']:.3f}")
    print(f"  Drifted  : {report['n_drifted']}/{report['n_total']} features")
    print(f"  Action   : {'RETRAIN' if report['drift_score'] > 0.3 else 'monitor'}")

    if report["drift_score"] > 0.3:
        drifted = [k for k, v in report["feature_drift"].items()
                   if v.get("drift_detected")]
        print(f"\n  Drifted features: {drifted[:10]}")