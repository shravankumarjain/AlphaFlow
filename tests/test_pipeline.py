# tests/test_pipeline.py
#
# AlphaFlow — Unit Tests
# Run with: pytest tests/ -v

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── CONFIG TESTS ──────────────────────────────────────────────────────

def test_config_tickers():
    from config import TICKERS, BENCHMARK_TICKER
    assert len(TICKERS) > 0, "TICKERS must not be empty"
    assert "SPY" not in TICKERS, "SPY should be BENCHMARK, not in TICKERS"
    assert BENCHMARK_TICKER == "SPY", "BENCHMARK_TICKER must be SPY"

def test_config_s3():
    from config import S3_BUCKET, AWS_REGION
    assert S3_BUCKET, "S3_BUCKET must be set"
    assert AWS_REGION, "AWS_REGION must be set"

def test_config_dates():
    from config import HISTORICAL_START, HISTORICAL_END
    import pandas as pd
    start = pd.Timestamp(HISTORICAL_START)
    end   = pd.Timestamp(HISTORICAL_END)
    assert start < end, "HISTORICAL_START must be before HISTORICAL_END"
    assert start.year >= 2018, "Start date should be >= 2018"


# ── DATA VALIDATION TESTS ─────────────────────────────────────────────

def test_ohlcv_validation():
    """Test the OHLCV validation logic."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data_pipeline.ingestion.market_data import validate_ohlcv

    # Valid data
    df_valid = pd.DataFrame({
        "date"  : pd.date_range("2024-01-01", periods=10),
        "open"  : np.random.uniform(100, 200, 10),
        "high"  : np.random.uniform(200, 250, 10),
        "low"   : np.random.uniform(80, 100, 10),
        "close" : np.random.uniform(100, 200, 10),
        "volume": np.random.randint(1000000, 5000000, 10).astype(float),
    })
    result = validate_ohlcv(df_valid, "TEST")
    assert len(result) > 0, "Valid data should pass validation"

    # Bad data: high < low
    df_bad = df_valid.copy()
    df_bad.loc[0, "high"] = 50   # high < low
    df_bad.loc[0, "low"]  = 100
    result_bad = validate_ohlcv(df_bad, "TEST_BAD")
    assert len(result_bad) < len(df_bad), "Row with high < low should be dropped"

    # Bad data: negative volume
    df_neg_vol = df_valid.copy()
    df_neg_vol.loc[0, "volume"] = -1000
    result_neg = validate_ohlcv(df_neg_vol, "TEST_NEG")
    assert len(result_neg) < len(df_neg_vol), "Row with negative volume should be dropped"


# ── FEATURE ENGINEERING TESTS ─────────────────────────────────────────

def test_technical_indicators():
    """Test that technical indicators are computed correctly."""
    from feature_engineering.feature_pipeline import add_technical_indicators

    n = 250
    df = pd.DataFrame({
        "date"  : pd.date_range("2023-01-01", periods=n),
        "open"  : np.random.uniform(100, 200, n),
        "high"  : np.random.uniform(200, 250, n),
        "low"   : np.random.uniform(80, 100, n),
        "close" : np.random.uniform(100, 200, n),
        "volume": np.random.randint(1000000, 5000000, n).astype(float),
    })

    result = add_technical_indicators(df)

    # Check RSI bounds
    rsi = result["rsi_14"].dropna()
    assert rsi.min() >= 0, "RSI must be >= 0"
    assert rsi.max() <= 100, "RSI must be <= 100"

    # Check MACD exists
    assert "macd" in result.columns
    assert "macd_signal" in result.columns
    assert "macd_hist" in result.columns

    # Check Bollinger Bands ordering
    valid = result.dropna(subset=["bb_upper", "bb_lower"])
    assert (valid["bb_upper"] >= valid["bb_lower"]).all(), "BB upper must be >= BB lower"

    # Check volatility is positive
    vol = result["vol_21d"].dropna()
    assert (vol >= 0).all(), "Volatility must be non-negative"


def test_calendar_features():
    """Test calendar feature generation."""
    from models.forecasting.dataset import add_calendar_features

    df = pd.DataFrame({
        "date"  : pd.date_range("2024-01-01", periods=100),
        "close" : np.random.uniform(100, 200, 100),
        "ticker": "TEST",
    })

    result = add_calendar_features(df)

    assert "day_of_week" in result.columns
    assert "month" in result.columns
    assert "quarter" in result.columns
    assert "is_month_end" in result.columns

    # Day of week should be 0-4 for weekdays
    dow = result["day_of_week"]
    assert dow.min() >= 0
    assert dow.max() <= 6


# ── PORTFOLIO OPTIMIZATION TESTS ──────────────────────────────────────

def test_markowitz_weights_sum_to_one():
    """Test that Markowitz weights sum to 1."""
    from portfolio.optimizer.portfolio_optimizer import MarkowitzOptimizer

    # Create simple test data
    n_days    = 252
    tickers   = ["AAPL", "MSFT", "GOOGL"]
    returns   = pd.DataFrame(
        np.random.normal(0.001, 0.02, (n_days, len(tickers))),
        columns=tickers,
        index=pd.date_range("2023-01-01", periods=n_days)
    )
    feature_data = {
        t: pd.DataFrame({
            "date"     : pd.date_range("2023-01-01", periods=n_days),
            "return_1d": returns[t].values,
            "close"    : (100 * (1 + returns[t]).cumprod()).values,
        })
        for t in tickers
    }

    optimizer = MarkowitzOptimizer(max_weight=0.5, min_weight=0.1)
    result    = optimizer.optimize(feature_data)
    weights   = result["weights"]

    # Weights must sum to ~1
    total = sum(weights.values())
    assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected 1.0"

    # All weights within bounds
    for t, w in weights.items():
        assert w >= 0.09, f"{t} weight {w} below min"
        assert w <= 0.51, f"{t} weight {w} above max"


def test_sharpe_ratio_calculation():
    """Test financial metrics computation."""
    from models.forecasting.evaluate import compute_metrics

    # Known returns
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))

    metrics = compute_metrics(returns, "test")

    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics
    assert "hit_rate" in metrics
    assert metrics["max_drawdown"] <= 0, "Max drawdown must be <= 0"
    assert 0 <= metrics["hit_rate"] <= 100, "Hit rate must be 0-100"


# ── DRIFT MONITOR TESTS ───────────────────────────────────────────────

def test_psi_drift_no_drift():
    """Test PSI drift detection with identical distributions."""
    from mlops.monitoring.drift_monitor import _psi_drift

    n = 500
    cols = ["feature_a", "feature_b"]
    ref = pd.DataFrame(np.random.normal(0, 1, (n, 2)), columns=cols)
    cur = pd.DataFrame(np.random.normal(0, 1, (n, 2)), columns=cols)

    result = _psi_drift(ref, cur, cols)
    assert result["drift_score"] < 0.3, "Identical distributions should not show high drift"


def test_psi_drift_with_drift():
    """Test PSI drift detection with shifted distributions."""
    from mlops.monitoring.drift_monitor import _psi_drift

    n = 500
    cols = ["feature_a"]
    ref = pd.DataFrame(np.random.normal(0, 1, (n, 1)), columns=cols)
    cur = pd.DataFrame(np.random.normal(5, 1, (n, 1)), columns=cols)  # shifted by 5 std

    result = _psi_drift(ref, cur, cols)
    assert result["drift_score"] > 0.3, "Heavily shifted distribution should show drift"


# ── INTEGRATION TEST ──────────────────────────────────────────────────

def test_s3_connectivity():
    """Test that S3 is accessible."""
    import boto3
    from config import S3_BUCKET, AWS_REGION
    from botocore.exceptions import NoCredentialsError, ClientError

    try:
        s3 = boto3.client("s3", region_name=AWS_REGION)
        s3.head_bucket(Bucket=S3_BUCKET)
        assert True, "S3 bucket accessible"
    except NoCredentialsError:
        pytest.skip("AWS credentials not available in CI")
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            pytest.fail(f"S3 bucket {S3_BUCKET} does not exist")
        elif e.response["Error"]["Code"] in ["403", "301"]:
            pass  # bucket exists, access limited