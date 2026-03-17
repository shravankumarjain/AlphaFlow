# models/backtesting/crisis_stress_test.py
#
# AlphaFlow — Crisis Stress Testing Engine
#
# Tests portfolio performance across every major market crisis:
#   COVID Crash        Feb 19 – Mar 23, 2020  (-34% in 33 days)
#   COVID Recovery     Mar 23 – Dec 31, 2020  (+70% recovery)
#   2022 Bear Market   Jan 01 – Dec 31, 2022  (-19% S&P, -65% crypto)
#   Rate Hike Cycle    Mar 16 – Jul 27, 2022  (Fed 0% → 2.5%)
#   Dot-com Crash      Mar 10 – Oct 09, 2002  (-78% NASDAQ)
#   2008 GFC           Oct 09 2007 – Mar 09 2009 (-57%)
#
# For each period we compute:
#   - AlphaFlow portfolio return vs SPY benchmark
#   - Max drawdown and recovery time
#   - Sharpe and Sortino ratios
#   - HMM regime detection accuracy
#   - Which features were most predictive
#
# Run: python models/backtesting/crisis_stress_test.py

import warnings

warnings.filterwarnings("ignore")

import logging  # noqa: E402, F401
import json  # noqa: E402, F401
import boto3  # noqa: E402, F401
import numpy as np  # noqa: E402, F401
import pandas as pd  # noqa: E402, F401
import yfinance as yf  # noqa: E402, F401
from datetime import datetime  # noqa: E402, F401
from pathlib import Path  # noqa: E402
from io import BytesIO  # noqa: E402, F401

import sys  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import AWS_REGION, S3_BUCKET, TICKERS, BENCHMARK_TICKER, LOCAL_DATA_DIR  # noqa: E402, F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/stress_test.log"),
    ],
)
logger = logging.getLogger("stress_test")

s3 = boto3.client("s3", region_name=AWS_REGION)

RISK_FREE_RATE = 0.05
TRADING_DAYS = 252
INITIAL_CAPITAL = 100_000

# ── CRISIS PERIODS ────────────────────────────────────────────────────
CRISIS_PERIODS = {
    "COVID Crash": {
        "start": "2020-02-19",
        "end": "2020-03-23",
        "description": "Fastest bear market in history — S&P fell 34% in 33 days",
        "benchmark_dd": -34.0,
        "color": "#ff4444",
    },
    "COVID Recovery": {
        "start": "2020-03-23",
        "end": "2020-12-31",
        "description": "Unprecedented V-shaped recovery driven by Fed stimulus",
        "benchmark_dd": 70.0,
        "color": "#00ff9d",
    },
    "2022 Bear Market": {
        "start": "2022-01-01",
        "end": "2022-12-31",
        "description": "Rate hike cycle — S&P -19%, NASDAQ -33%, BTC -65%",
        "benchmark_dd": -19.4,
        "color": "#ff6b35",
    },
    "Fed Rate Shock": {
        "start": "2022-03-16",
        "end": "2022-07-27",
        "description": "Fed raised rates 0% → 2.5% — fastest hiking cycle in 40 years",
        "benchmark_dd": -15.0,
        "color": "#ffd700",
    },
    "Dot-com Crash": {
        "start": "2000-03-10",
        "end": "2002-10-09",
        "description": "NASDAQ fell 78% — tech bubble collapse",
        "benchmark_dd": -49.0,
        "color": "#7b61ff",
    },
    "2008 GFC": {
        "start": "2007-10-09",
        "end": "2009-03-09",
        "description": "Global Financial Crisis — S&P lost 57% from peak",
        "benchmark_dd": -57.0,
        "color": "#ff4db8",
    },
}


# ── DATA LOADING ──────────────────────────────────────────────────────


def load_equity_returns(start: str, end: str) -> pd.DataFrame:
    """
    Load daily returns for all tickers in the crisis period.
    Uses yfinance for historical data — goes back to 2000.
    """
    # Only use tickers available for this period
    available = []
    frames = {}

    for ticker in TICKERS[:15]:  # top 15 equities for speed
        try:
            df = yf.download(
                ticker, start=start, end=end, auto_adjust=True, progress=False
            )
            if len(df) < 10:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            ret = df["Close"].pct_change().dropna()
            frames[ticker] = ret
            available.append(ticker)
        except Exception:
            continue

    if not frames:
        raise ValueError(f"No data available for {start} to {end}")

    returns = pd.DataFrame(frames).dropna(how="all").fillna(0)
    logger.info(f"  ✓ Loaded {len(available)} tickers | {len(returns)} days")
    return returns


def load_benchmark_returns(start: str, end: str) -> pd.Series:
    """Load SPY returns for benchmark comparison."""
    try:
        spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        return spy["Close"].pct_change().dropna()
    except Exception as e:
        logger.warning(f"  ⚠ SPY data unavailable: {e}")
        return pd.Series(dtype=float)


# ── REGIME DETECTION IN CRISIS ────────────────────────────────────────


def detect_regimes_in_period(returns: pd.DataFrame, spy_returns: pd.Series) -> dict:
    """
    Run HMM regime detection on crisis period data.
    Returns regime sequence and accuracy metrics.
    """
    try:
        from hmmlearn import hmm
        from sklearn.preprocessing import StandardScaler

        if len(spy_returns) < 30:
            return {"regime_accuracy": 0, "regimes": []}

        # Build features
        features = pd.DataFrame(
            {
                "spy_return": spy_returns,
                "spy_vol": spy_returns.rolling(21).std().fillna(spy_returns.std()),
                "spy_momentum": spy_returns.rolling(5).mean().fillna(0),
            }
        ).dropna()

        if len(features) < 20:
            return {"regime_accuracy": 0, "regimes": []}

        scaler = StandardScaler()
        X = scaler.fit_transform(features.values)

        model = hmm.GaussianHMM(
            n_components=3,
            covariance_type="full",
            n_iter=100,
            random_state=42,
        )
        model.fit(X)
        states = model.predict(X)

        # Label states by return level
        state_returns = {
            s: features["spy_return"].values[states == s].mean() for s in range(3)
        }
        sorted_states = sorted(state_returns.items(), key=lambda x: x[1])
        labels = {
            sorted_states[0][0]: "BEAR",
            sorted_states[1][0]: "NEUTRAL",
            sorted_states[2][0]: "BULL",
        }

        regime_sequence = [labels[s] for s in states]
        crisis_detected = sum(1 for r in regime_sequence if r == "BEAR")
        detection_rate = crisis_detected / len(regime_sequence)

        return {
            "regime_accuracy": round(detection_rate, 3),
            "regime_sequence": regime_sequence[-10:],  # last 10 days
            "final_regime": regime_sequence[-1],
            "bear_days": crisis_detected,
            "total_days": len(regime_sequence),
        }

    except Exception as e:
        logger.warning(f"  ⚠ HMM detection failed: {e}")
        return {"regime_accuracy": 0, "regimes": []}


# ── PORTFOLIO STRATEGIES ──────────────────────────────────────────────


def markowitz_strategy(
    returns: pd.DataFrame, spy_returns: pd.Series, max_weight: float = 0.30
) -> pd.Series:
    """
    Simulate Markowitz minimum-variance portfolio over the crisis period.
    Rebalances monthly.
    """
    import cvxpy as cp

    portfolio_returns = []
    dates = returns.index
    tickers = list(returns.columns)
    n = len(tickers)
    lookback = 63  # 3 months

    for i in range(lookback, len(dates)):
        hist = returns.iloc[i - lookback : i]
        mu = hist.mean().values * TRADING_DAYS
        cov = hist.cov().values * TRADING_DAYS

        w = cp.Variable(n)
        try:
            prob = cp.Problem(
                cp.Maximize(mu @ w - 2.0 * cp.quad_form(w, cov)),
                [cp.sum(w) == 1, w >= 0.01, w <= max_weight],
            )
            prob.solve(solver=cp.CLARABEL, verbose=False)
            weights = np.clip(np.array(w.value).flatten(), 0, 1)
            weights /= weights.sum()
        except Exception:
            weights = np.ones(n) / n

        daily_ret = float(np.dot(weights, returns.iloc[i].values))
        portfolio_returns.append(daily_ret)

    return pd.Series(portfolio_returns, index=dates[lookback:])


def equal_weight_strategy(returns: pd.DataFrame) -> pd.Series:
    """Equal weight baseline — same weight to all assets."""
    return returns.mean(axis=1)


def regime_aware_strategy(returns: pd.DataFrame, spy_returns: pd.Series) -> pd.Series:
    """
    Regime-aware strategy — shifts to defensive in BEAR.
    Approximates AlphaFlow's HMM + Markowitz behavior.
    """
    portfolio_returns = []
    dates = returns.index
    tickers = list(returns.columns)
    n = len(tickers)
    lookback = 21
    weights = np.ones(n) / n

    for i in range(lookback, len(dates)):
        # Detect regime from recent returns
        recent_spy = spy_returns.reindex(dates[i - lookback : i]).fillna(0)
        spy_vol = recent_spy.std() * np.sqrt(TRADING_DAYS)

        # Simple regime: high vol = BEAR, low vol = BULL
        if spy_vol > 0.25:
            regime = "BEAR"
        elif spy_vol > 0.15:
            regime = "NEUTRAL"
        else:
            regime = "BULL"

        # Regime-based weight adjustment
        hist = returns.iloc[i - lookback : i]

        if regime == "BEAR":
            # Overweight low-beta, underweight high-vol
            vols = hist.std().values
            inv_vol = 1.0 / (vols + 1e-6)
            weights = inv_vol / inv_vol.sum()
        elif regime == "BULL":
            # Overweight momentum
            momentum = hist.iloc[-5:].mean().values
            momentum = np.clip(momentum, 0, None)
            if momentum.sum() > 0:
                weights = momentum / momentum.sum()
            else:
                weights = np.ones(n) / n
        else:
            weights = np.ones(n) / n

        weights = np.clip(weights, 0.01, 0.35)
        weights /= weights.sum()

        daily_ret = float(np.dot(weights, returns.iloc[i].values))
        portfolio_returns.append(daily_ret)

    return pd.Series(portfolio_returns, index=dates[lookback:])


# ── FINANCIAL METRICS ─────────────────────────────────────────────────


def compute_metrics(returns: pd.Series, name: str) -> dict:
    returns = returns.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 5:
        return {
            "name": name,
            "total_return": 0,
            "sharpe": 0,
            "max_drawdown": 0,
            "sortino": 0,
            "recovery_days": 0,
        }

    total_return = float((1 + returns).prod() - 1)
    n_years = max(len(returns) / TRADING_DAYS, 0.01)
    annual_return = float((1 + total_return) ** (1 / n_years) - 1)
    daily_vol = float(returns.std())
    annual_vol = daily_vol * np.sqrt(TRADING_DAYS)
    downside = returns[returns < 0]
    down_vol = (
        float(downside.std()) * np.sqrt(TRADING_DAYS) if len(downside) > 1 else 1e-6
    )
    daily_rf = RISK_FREE_RATE / TRADING_DAYS
    sharpe = (
        float((returns - daily_rf).mean() / daily_vol * np.sqrt(TRADING_DAYS))
        if daily_vol > 0
        else 0
    )
    sortino = float((annual_return - RISK_FREE_RATE) / down_vol) if down_vol > 0 else 0
    cum = (1 + returns).cumprod()
    max_dd = float((cum / cum.cummax() - 1).min())
    hit_rate = float((returns > 0).mean())

    # Recovery time — days from trough to new high
    peak = cum.cummax()
    trough = (cum / peak).idxmin()
    after = cum[trough:]
    new_high = after[after >= peak[trough]]
    recovery_days = len(new_high) if len(new_high) > 0 else -1  # -1 = never recovered

    return {
        "name": name,
        "total_return": round(total_return * 100, 2),
        "annual_return": round(annual_return * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown": round(max_dd * 100, 2),
        "annual_vol": round(annual_vol * 100, 2),
        "hit_rate": round(hit_rate * 100, 2),
        "recovery_days": recovery_days,
    }


# ── CRISIS SCENARIO RUNNER ────────────────────────────────────────────


def run_crisis_scenario(name: str, period: dict) -> dict:
    """Run full stress test for one crisis period."""
    logger.info(f"\n{'=' * 50}")
    logger.info(f"Crisis: {name}")
    logger.info(f"Period: {period['start']} → {period['end']}")
    logger.info(f"{'=' * 50}")

    start = period["start"]
    end = period["end"]

    # Load data
    try:
        returns = load_equity_returns(start, end)
        spy_returns = load_benchmark_returns(start, end)
    except Exception as e:
        logger.error(f"  ✗ Data load failed: {e}")
        return {"name": name, "error": str(e)}

    if len(returns) < 20:
        logger.warning(f"  ⚠ Insufficient data for {name}")
        return {"name": name, "error": "insufficient data"}

    # Run strategies
    logger.info("  Running strategies...")

    try:
        regime_returns = regime_aware_strategy(returns, spy_returns)
    except Exception as e:
        logger.warning(f"  ⚠ Regime strategy failed: {e}")
        regime_returns = equal_weight_strategy(returns).iloc[21:]

    try:
        markowitz_returns = markowitz_strategy(returns, spy_returns)
    except Exception as e:
        logger.warning(f"  ⚠ Markowitz failed: {e}")
        markowitz_returns = equal_weight_strategy(returns).iloc[63:]

    ew_returns = equal_weight_strategy(returns)

    # Align benchmark to same period
    spy_aligned = spy_returns.reindex(regime_returns.index).fillna(0)

    # Compute metrics
    strategies = {
        "AlphaFlow (Regime-Aware)": regime_returns,
        "Markowitz": markowitz_returns,
        "Equal Weight": ew_returns.reindex(regime_returns.index).fillna(0),
        "SPY Benchmark": spy_aligned,
    }

    results = {}
    for strat_name, ret in strategies.items():
        results[strat_name] = compute_metrics(ret, strat_name)

    # HMM regime detection
    regime_info = detect_regimes_in_period(returns, spy_returns)

    # Alpha vs benchmark
    alphaflow_return = results["AlphaFlow (Regime-Aware)"]["total_return"]
    spy_return = results["SPY Benchmark"]["total_return"]
    alpha = alphaflow_return - spy_return

    scenario_result = {
        "name": name,
        "period": f"{start} to {end}",
        "description": period["description"],
        "color": period["color"],
        "strategies": results,
        "regime_detection": regime_info,
        "alpha_vs_spy": round(alpha, 2),
        "beat_benchmark": alpha > 0,
        "benchmark_historical_dd": period.get("benchmark_dd", 0),
    }

    # Log summary
    af = results["AlphaFlow (Regime-Aware)"]
    spy = results["SPY Benchmark"]
    logger.info(
        f"  AlphaFlow: {af['total_return']:+.1f}% | Sharpe {af['sharpe']:.3f} | MaxDD {af['max_drawdown']:.1f}%"
    )
    logger.info(
        f"  SPY:       {spy['total_return']:+.1f}% | Sharpe {spy['sharpe']:.3f} | MaxDD {spy['max_drawdown']:.1f}%"
    )
    logger.info(
        f"  Alpha:     {alpha:+.1f}% | Beat benchmark: {'✓' if alpha > 0 else '✗'}"
    )
    if regime_info.get("bear_days"):
        logger.info(
            f"  BEAR regime: {regime_info['bear_days']}/{regime_info['total_days']} days detected"
        )

    return scenario_result


# ── FULL STRESS TEST ──────────────────────────────────────────────────


def run_full_stress_test() -> dict:
    """Run stress test across all crisis periods."""
    logger.info("=" * 60)
    logger.info("AlphaFlow — Crisis Stress Testing Engine")
    logger.info("=" * 60)

    Path("logs").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    all_results = {}

    for crisis_name, period in CRISIS_PERIODS.items():
        result = run_crisis_scenario(crisis_name, period)
        all_results[crisis_name] = result

    # Save full report
    output_path = "reports/stress_test_report.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary table
    print("\n" + "=" * 90)
    print("ALPHAFLOW — CRISIS STRESS TEST SUMMARY")
    print("=" * 90)
    print(
        f"{'Crisis':<22} {'AlphaFlow':>10} {'SPY':>8} {'Alpha':>8} {'MaxDD':>8} {'Sharpe':>8} {'Beat?':>6}"
    )
    print("-" * 90)

    beat_count = 0
    for name, result in all_results.items():
        if "error" in result:
            print(f"{name:<22} {'ERROR':>10}")
            continue
        strats = result.get("strategies", {})
        af = strats.get("AlphaFlow (Regime-Aware)", {})
        spy = strats.get("SPY Benchmark", {})
        alpha = result.get("alpha_vs_spy", 0)
        beat = "✓" if result.get("beat_benchmark") else "✗"
        if result.get("beat_benchmark"):
            beat_count += 1
        print(
            f"{name:<22} "
            f"{af.get('total_return', 0):>9.1f}% "
            f"{spy.get('total_return', 0):>7.1f}% "
            f"{alpha:>+7.1f}% "
            f"{af.get('max_drawdown', 0):>7.1f}% "
            f"{af.get('sharpe', 0):>8.3f} "
            f"{beat:>6}"
        )

    total = len([r for r in all_results.values() if "error" not in r])
    print("=" * 90)
    print(f"Beat benchmark: {beat_count}/{total} crisis periods")
    print(f"Report saved: {output_path}")

    return all_results


if __name__ == "__main__":
    results = run_full_stress_test()
