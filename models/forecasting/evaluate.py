# models/forecasting/evaluate.py
# AlphaFlow — Backtesting & Evaluation Engine

import warnings

warnings.filterwarnings("ignore")

import logging  # noqa: E402
import json  # noqa: E402
import mlflow  # noqa: E402
import torch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402
from pathlib import Path  # noqa: E402
from pytorch_forecasting import TemporalFusionTransformer  # noqa: E402

import sys  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import LOCAL_DATA_DIR  # noqa: E402
from models.forecasting.dataset import prepare_datasets, TARGET  # noqa: E402
from models.forecasting.tft_model import build_dataloaders  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/evaluate.log"),
    ],
)
logger = logging.getLogger("evaluate")

RISK_FREE_RATE = 0.05
TRADING_DAYS = 252
INITIAL_CAPITAL = 100_000


# ── Load Model ────────────────────────────────────────────────────────
def load_best_model(
    checkpoint_dir: str = "models/checkpoints",
) -> TemporalFusionTransformer:
    checkpoints = list(Path(checkpoint_dir).glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints in {checkpoint_dir}")
    best = min(checkpoints, key=lambda p: float(p.stem.split("val_loss=")[1]))
    logger.info(f"  ✓ Loading: {best.name}")
    model = TemporalFusionTransformer.load_from_checkpoint(str(best))
    model.eval()
    return model


# ── Generate Predictions ──────────────────────────────────────────────
def generate_predictions(model, val_loader, val_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("  Generating predictions...")
    with torch.no_grad():
        predictions = model.predict(val_loader, mode="quantiles", return_x=True)

    preds = predictions.output
    if hasattr(preds, "numpy"):
        preds = preds.numpy()
    else:
        preds = np.array(preds)

    if preds.ndim == 3:
        p10 = preds[:, 0, 0]
        p50 = preds[:, 0, 1]
        p90 = preds[:, 0, 2]
    elif preds.ndim == 2:
        p50 = preds[:, 0]
        p10 = p50 * 0.9
        p90 = p50 * 1.1
    else:
        p50 = preds.flatten()
        p10 = p50 * 0.9
        p90 = p50 * 1.1

    n = len(p50)
    val_aligned = val_df.tail(n).copy().reset_index(drop=True)

    pred_df = pd.DataFrame(
        {
            "date": pd.to_datetime(val_aligned["date"].values),
            "ticker": val_aligned["ticker"].values.astype(str),
            "actual": val_aligned[TARGET].values,
            "pred_p10": p10,
            "pred_p50": p50,
            "pred_p90": p90,
            "uncertainty": p90 - p10,
        }
    )

    logger.info(
        f"  ✓ {len(pred_df)} predictions | pred_p50 range: {p50.min():.5f} to {p50.max():.5f}"
    )
    logger.info(
        f"  ✓ actual range: {pred_df['actual'].min():.5f} to {pred_df['actual'].max():.5f}"
    )
    return pred_df


# ── Generate Signals ──────────────────────────────────────────────────
def generate_signals(pred_df: pd.DataFrame) -> pd.DataFrame:
    threshold = 0.001
    max_uncert = pred_df["uncertainty"].quantile(0.95)

    pred_df = pred_df.copy()
    pred_df["signal"] = 0.0

    long_mask = (pred_df["pred_p50"] > threshold) & (
        pred_df["uncertainty"] < max_uncert
    )
    short_mask = (pred_df["pred_p50"] < -threshold) & (
        pred_df["uncertainty"] < max_uncert
    )

    pred_df.loc[long_mask, "signal"] = 1.0
    pred_df.loc[short_mask, "signal"] = -1.0

    pred_df["position_size"] = pred_df["signal"]

    logger.info(
        f"  ✓ Signals: {long_mask.sum()} long | {short_mask.sum()} short | {(pred_df['signal'] == 0).sum()} flat"
    )
    return pred_df


# ── Backtest ──────────────────────────────────────────────────────────
def run_backtest(pred_df: pd.DataFrame) -> pd.DataFrame:
    TC = 0.001
    daily = (
        pred_df.groupby("date")
        .apply(
            lambda x: pd.Series(
                {
                    "avg_signal": x["position_size"].mean(),
                    "avg_actual": x["actual"].mean(),
                }
            )
        )
        .reset_index()
        .sort_values("date")
    )

    daily["signal_change"] = daily["avg_signal"].diff().abs().fillna(0)
    daily["gross_return"] = daily["avg_signal"] * daily["avg_actual"] / 5
    daily["transaction_cost"] = daily["signal_change"] * TC
    daily["net_return"] = daily["gross_return"] - daily["transaction_cost"]
    daily["portfolio_value"] = INITIAL_CAPITAL * (1 + daily["net_return"]).cumprod()
    daily["cumulative_return"] = daily["portfolio_value"] / INITIAL_CAPITAL - 1

    logger.info(f"  ✓ Backtest: {len(daily)} days")
    return daily


# ── Benchmarks ────────────────────────────────────────────────────────
def build_benchmarks(pred_df: pd.DataFrame) -> dict:
    # Equal weight — average actual return across all tickers per day
    ew = pred_df.groupby("date")["actual"].mean().sort_index() / 5
    ew_value = INITIAL_CAPITAL * (1 + ew).cumprod()

    # Buy & Hold SPY — ticker "9" is SPY encoding
    spy = (
        pred_df[pred_df["ticker"] == "6"].groupby("date")["actual"].mean().sort_index()
        / 5
    )
    if spy.empty:
        spy = ew
    bh_value = INITIAL_CAPITAL * (1 + spy).cumprod()

    # Random signals
    np.random.seed(42)
    rand_signals = np.random.choice([-1, 0, 1], size=len(ew))
    rand_ret = rand_signals * ew.values
    rand_value = INITIAL_CAPITAL * np.cumprod(1 + rand_ret)

    return {
        "Buy Hold": pd.Series(bh_value.values, index=bh_value.index),
        "Equal Weight": pd.Series(ew_value.values, index=ew_value.index),
        "Random": pd.Series(rand_value, index=ew.index),
    }


# ── Financial Metrics ─────────────────────────────────────────────────
def compute_metrics(returns: pd.Series, name: str) -> dict:
    returns = pd.Series(returns).dropna().replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 5:
        return {
            "name": name,
            "total_return": 0,
            "annual_return": 0,
            "annual_vol": 0,
            "sharpe_ratio": 0,
            "sortino_ratio": 0,
            "max_drawdown": 0,
            "calmar_ratio": 0,
            "hit_rate": 0,
        }

    total_return = float((1 + returns).prod() - 1)
    n_years = len(returns) / TRADING_DAYS
    annual_return = float((1 + total_return) ** (1 / max(n_years, 0.01)) - 1)
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
    calmar = float(annual_return / abs(max_dd)) if max_dd != 0 else 0
    hit_rate = float((returns > 0).mean())

    return {
        "name": name,
        "total_return": round(total_return * 100, 2),
        "annual_return": round(annual_return * 100, 2),
        "annual_vol": round(annual_vol * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "max_drawdown": round(max_dd * 100, 2),
        "calmar_ratio": round(calmar, 3),
        "hit_rate": round(hit_rate * 100, 2),
    }


# ── Charts ────────────────────────────────────────────────────────────
def create_backtest_charts(strategy_df, benchmarks, all_metrics):
    Path("reports").mkdir(exist_ok=True)
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Portfolio Value",
            "Drawdown",
            "Daily Returns Distribution",
            "Sharpe Comparison",
        ],
        vertical_spacing=0.15,
    )

    dates = pd.to_datetime(strategy_df["date"])

    # Portfolio value
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=strategy_df["portfolio_value"],
            name="AlphaFlow TFT",
            line=dict(color="#00e5ff", width=2),
        ),
        row=1,
        col=1,
    )
    colors = ["#ffd700", "#7b61ff", "#ff4444"]
    for (bname, bseries), color in zip(benchmarks.items(), colors):
        fig.add_trace(
            go.Scatter(
                x=bseries.index,
                y=bseries.values,
                name=bname,
                line=dict(color=color, dash="dash", width=1.5),
            ),
            row=1,
            col=1,
        )

    # Drawdown
    cum = (1 + strategy_df["net_return"]).cumprod()
    dd = (cum / cum.cummax() - 1) * 100
    fig.add_trace(
        go.Scatter(
            x=dates, y=dd, fill="tozeroy", name="Drawdown", line=dict(color="#ff4444")
        ),
        row=1,
        col=2,
    )

    # Return distribution
    fig.add_trace(
        go.Histogram(
            x=strategy_df["net_return"] * 100,
            nbinsx=40,
            name="Returns",
            marker_color="#00ff9d",
        ),
        row=2,
        col=1,
    )

    # Sharpe comparison
    names = [m["name"] for m in all_metrics]
    sharpes = [m["sharpe_ratio"] for m in all_metrics]
    bar_col = ["#00e5ff" if n == "AlphaFlow TFT" else "#555" for n in names]
    fig.add_trace(
        go.Bar(x=names, y=sharpes, marker_color=bar_col, name="Sharpe"), row=2, col=2
    )

    fig.update_layout(
        title="AlphaFlow — Backtest Report",
        paper_bgcolor="#070a0f",
        plot_bgcolor="#0d1117",
        font=dict(color="#c9d1d9"),
        height=800,
    )

    path = "reports/backtest_report.html"
    fig.write_html(path)
    logger.info(f"  ✓ Chart saved: {path}")
    return path


# ── Print Table ───────────────────────────────────────────────────────
def print_metrics_table(all_metrics):
    print("\n" + "=" * 78)
    print("ALPHAFLOW — BACKTEST RESULTS")
    print("=" * 78)
    print(
        f"{'Strategy':<20} {'Return%':>8} {'Annual%':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD%':>8} {'Hit%':>7}"
    )
    print("-" * 78)
    for m in all_metrics:
        marker = " ◀" if m["name"] == "AlphaFlow TFT" else ""
        print(
            f"{m['name']:<20} {m['total_return']:>7.1f}% {m['annual_return']:>7.1f}% "
            f"{m['sharpe_ratio']:>8.3f} {m['sortino_ratio']:>8.3f} "
            f"{m['max_drawdown']:>7.1f}% {m['hit_rate']:>6.1f}%{marker}"
        )
    print("=" * 78)


# ── Main ──────────────────────────────────────────────────────────────
def run_evaluation():
    logger.info("=" * 60)
    logger.info("AlphaFlow — Backtesting & Evaluation")
    logger.info("=" * 60)

    Path("logs").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    # Step 1: Datasets
    logger.info("\nStep 1: Loading datasets...")
    training_dataset, validation_dataset, train_df, val_df = prepare_datasets()
    _, val_loader = build_dataloaders(training_dataset, validation_dataset)

    # Step 2: Model
    logger.info("\nStep 2: Loading model...")
    model = load_best_model()

    # Step 3: Predictions
    logger.info("\nStep 3: Predictions...")
    pred_df = generate_predictions(model, val_loader, val_df)
    pred_df.to_parquet(Path(LOCAL_DATA_DIR) / "predictions.parquet", index=False)

    # Step 4: Signals
    logger.info("\nStep 4: Signals...")
    pred_df = generate_signals(pred_df)

    # Step 5: Backtest
    logger.info("\nStep 5: Backtest...")
    strategy_df = run_backtest(pred_df)

    # Step 6: Benchmarks
    logger.info("\nStep 6: Benchmarks...")
    benchmarks = build_benchmarks(pred_df)

    # Step 7: Metrics
    logger.info("\nStep 7: Metrics...")
    all_metrics = [compute_metrics(strategy_df["net_return"], "AlphaFlow TFT")]
    for bname, bseries in benchmarks.items():
        bret = bseries.pct_change().dropna()
        all_metrics.append(compute_metrics(bret, bname))

    # Step 8: Charts
    logger.info("\nStep 8: Charts...")
    chart_path = create_backtest_charts(strategy_df, benchmarks, all_metrics)  # noqa: F841

    # Step 9: MLflow
    logger.info("\nStep 9: MLflow...")
    mlflow.set_tracking_uri("mlruns")
    with mlflow.start_run(run_name="backtest-v1"):
        for m in all_metrics:
            prefix = m["name"].lower().replace(" ", "_")
            for k, v in m.items():
                if k != "name" and isinstance(v, (int, float)):
                    mlflow.log_metric(f"{prefix}_{k}", v)

    # Step 10: Report
    print_metrics_table(all_metrics)
    with open("reports/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info("  ✓ Done. Open: reports/backtest_report.html")
    return all_metrics, strategy_df, pred_df


if __name__ == "__main__":
    all_metrics, strategy_df, pred_df = run_evaluation()
    print("\nopen reports/backtest_report.html")
