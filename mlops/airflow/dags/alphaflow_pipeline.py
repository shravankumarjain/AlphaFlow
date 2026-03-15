# mlops/airflow/dags/alphaflow_pipeline.py
#
# AlphaFlow — Master Airflow DAG
#
# This DAG orchestrates the entire pipeline on a daily schedule:
#
#   1. ingest_market_data     — pull OHLCV from yfinance → S3
#   2. ingest_news_data       — pull news + SEC filings → S3
#   3. build_features         — compute 65 features → S3
#   4. retrain_check          — check if retraining is needed (drift)
#   5. retrain_model          — retrain TFT if drift detected
#   6. run_evaluation         — backtest + metrics → MLflow
#   7. run_optimization       — Markowitz + RL allocation
#   8. health_check           — verify all outputs are fresh
#
# Schedule: runs at 6:30 AM UTC daily (after US market close + 30min)
#
# How to run locally with Docker:
#   docker-compose -f mlops/docker-compose.yml up
#   open http://localhost:8080  (Airflow UI, admin/admin)

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator # type: ignore
from airflow.operators.empty import EmptyOperator # type: ignore
from airflow.utils.dates import days_ago # type: ignore
import subprocess
import logging
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# ── DAG DEFAULT ARGS ──────────────────────────────────────────────────
default_args = {
    "owner"           : "alphaflow",
    "depends_on_past" : False,
    "start_date"      : days_ago(1),
    "email_on_failure": False,
    "email_on_retry"  : False,
    "retries"         : 2,
    "retry_delay"     : timedelta(minutes=5),
}

# ── HELPER ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(os.environ.get("ALPHAFLOW_ROOT", "/opt/airflow/alphaflow"))

def run_script(script_path: str, **kwargs) -> dict:
    """Run a Python script and return exit code."""
    full_path = PROJECT_ROOT / script_path
    logger.info(f"Running: {full_path}")
    result = subprocess.run(
        ["python", str(full_path)],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        logger.error(f"STDOUT: {result.stdout}")
        logger.error(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Script failed: {script_path}\n{result.stderr}")
    logger.info(f"✓ {script_path} completed successfully")
    return {"returncode": result.returncode, "stdout": result.stdout[-500:]}


# ── TASK FUNCTIONS ────────────────────────────────────────────────────

def task_ingest_market(**kwargs):
    return run_script("data_pipeline/ingestion/market_data.py")

def task_ingest_news(**kwargs):
    return run_script("data_pipeline/ingestion/news_data.py")

def task_build_features(**kwargs):
    return run_script("feature_engineering/feature_pipeline.py")

def task_check_drift(**kwargs) -> str:
    """
    Check if data drift has occurred since last training.
    Returns branch name: 'retrain_model' or 'skip_retrain'

    Drift is detected by Evidently in drift_monitor.py.
    If drift score > threshold → retrain.
    If last training was > 30 days ago → force retrain.
    """
    drift_report_path = PROJECT_ROOT / "reports" / "drift_report.json"

    # Force retrain if no drift report exists
    if not drift_report_path.exists():
        logger.info("No drift report found — triggering retrain")
        return "retrain_model"

    with open(drift_report_path) as f:
        report = json.load(f)

    drift_score     = report.get("drift_score", 0)
    last_trained    = report.get("last_trained", "2020-01-01")
    days_since_train = (datetime.utcnow() - datetime.fromisoformat(last_trained)).days

    DRIFT_THRESHOLD   = 0.3   # retrain if 30%+ of features drifted
    MAX_DAYS_NO_RETRAIN = 30  # force retrain every 30 days

    if drift_score > DRIFT_THRESHOLD:
        logger.info(f"Drift detected: {drift_score:.2f} > {DRIFT_THRESHOLD} — retraining")
        return "retrain_model"
    elif days_since_train > MAX_DAYS_NO_RETRAIN:
        logger.info(f"Scheduled retrain: {days_since_train} days since last training")
        return "retrain_model"
    else:
        logger.info(f"No retrain needed: drift={drift_score:.2f}, days={days_since_train}")
        return "skip_retrain"

def task_retrain_model(**kwargs):
    return run_script("models/forecasting/tft_model.py")

def task_run_evaluation(**kwargs):
    return run_script("models/forecasting/evaluate.py")

def task_run_optimization(**kwargs):
    return run_script("portfolio/optimizer/portfolio_optimizer.py")

def task_run_drift_monitor(**kwargs):
    return run_script("mlops/monitoring/drift_monitor.py")

def task_health_check(**kwargs):
    """
    Verify all outputs are fresh and pipeline completed successfully.
    Raises if any critical file is missing or stale.
    """
    checks = {
        "predictions"  : PROJECT_ROOT / "data/local/predictions.parquet",
        "allocation"   : PROJECT_ROOT / "reports/allocation.json",
        "metrics"      : PROJECT_ROOT / "reports/metrics.json",
        "drift_report" : PROJECT_ROOT / "reports/drift_report.json",
    }

    failed = []
    for name, path in checks.items():
        if not path.exists():
            failed.append(f"{name} missing: {path}")
            continue
        # Check file is less than 2 days old
        age_hours = (datetime.utcnow().timestamp() - path.stat().st_mtime) / 3600
        if age_hours > 48:
            failed.append(f"{name} stale: {age_hours:.0f}h old")

    if failed:
        raise RuntimeError("Health check failed:\n" + "\n".join(failed))

    logger.info("✓ Health check passed — all outputs fresh")
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


# ── DAG DEFINITION ────────────────────────────────────────────────────
with DAG(
    dag_id             = "alphaflow_daily_pipeline",
    default_args       = default_args,
    description        = "AlphaFlow daily data + model + portfolio pipeline",
    schedule_interval  = "30 6 * * 1-5",   # 6:30 AM UTC, Mon-Fri only
    catchup            = False,
    max_active_runs    = 1,
    tags               = ["alphaflow", "production"],
) as dag:

    # ── START ─────────────────────────────────────────────────────────
    start = EmptyOperator(task_id="start")

    # ── DATA INGESTION (parallel) ──────────────────────────────────────
    ingest_market = PythonOperator(
        task_id         = "ingest_market_data",
        python_callable = task_ingest_market,
    )

    ingest_news = PythonOperator(
        task_id         = "ingest_news_data",
        python_callable = task_ingest_news,
    )

    # ── FEATURE ENGINEERING ───────────────────────────────────────────
    build_features = PythonOperator(
        task_id         = "build_features",
        python_callable = task_build_features,
    )

    # ── DRIFT MONITORING ──────────────────────────────────────────────
    drift_monitor = PythonOperator(
        task_id         = "run_drift_monitor",
        python_callable = task_run_drift_monitor,
    )

    # ── BRANCH: retrain or skip ───────────────────────────────────────
    drift_check = BranchPythonOperator(
        task_id         = "check_drift",
        python_callable = task_check_drift,
    )

    retrain_model = PythonOperator(
        task_id         = "retrain_model",
        python_callable = task_retrain_model,
    )

    skip_retrain = EmptyOperator(task_id="skip_retrain")

    # ── EVALUATION ────────────────────────────────────────────────────
    run_evaluation = PythonOperator(
        task_id          = "run_evaluation",
        python_callable  = task_run_evaluation,
        trigger_rule     = "none_failed_min_one_success",
    )

    # ── PORTFOLIO OPTIMIZATION ────────────────────────────────────────
    run_optimization = PythonOperator(
        task_id         = "run_portfolio_optimization",
        python_callable = task_run_optimization,
    )

    # ── HEALTH CHECK ──────────────────────────────────────────────────
    health_check = PythonOperator(
        task_id         = "health_check",
        python_callable = task_health_check,
    )

    # ── END ───────────────────────────────────────────────────────────
    end = EmptyOperator(task_id="end")

    # ── DEPENDENCIES (the DAG graph) ──────────────────────────────────
    #
    #  start
    #    ├── ingest_market ──┐
    #    └── ingest_news   ──┴── build_features
    #                               │
    #                         drift_monitor
    #                               │
    #                          drift_check
    #                         /          \
    #                  retrain_model   skip_retrain
    #                         \          /
    #                        run_evaluation
    #                               │
    #                       run_optimization
    #                               │
    #                          health_check
    #                               │
    #                              end

    start >> [ingest_market, ingest_news] >> build_features
    build_features >> drift_monitor >> drift_check
    drift_check >> [retrain_model, skip_retrain]
    [retrain_model, skip_retrain] >> run_evaluation
    run_evaluation >> run_optimization >> health_check >> end