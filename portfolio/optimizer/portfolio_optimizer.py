"""
portfolio/optimizer/portfolio_optimizer.py
AlphaFlow — Portfolio Optimization Engine
Markowitz + RL blended allocation, regime-aware via HMM detector.
"""

import warnings

warnings.filterwarnings("ignore")

import logging, json, sys  # noqa: E401, E402
import numpy as np  # noqa: E401, E402
import pandas as pd  # noqa: E402
import cvxpy as cp  # noqa: E402
from pathlib import Path  # noqa: E402
from datetime import datetime  # noqa: E402
from io import BytesIO  # noqa: E402
import boto3  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import AWS_REGION, S3_BUCKET, TICKERS, S3_FEATURES_PREFIX, LOCAL_DATA_DIR  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/portfolio.log")],
)
logger = logging.getLogger("portfolio")

s3 = boto3.client("s3", region_name=AWS_REGION)

TICKER_NAMES = {
    "0": "AAPL",
    "1": "MSFT",
    "2": "GOOGL",
    "3": "AMZN",
    "4": "JPM",
    "5": "JNJ",
    "6": "SPY",
    "7": "BRK-B",
    "8": "TSLA",
    "9": "XOM",
}


# ── Regime loading ────────────────────────────────────────────────────────────


def load_regime() -> dict:
    """Load current regime from HMM detector output."""
    regime_path = Path("data/local/regime/current_regime.json")
    if regime_path.exists():
        with open(regime_path) as f:
            data = json.load(f)
        logger.info(
            f"  ✓ Regime loaded: {data['current_regime'].upper()} "
            f"(confidence {data['confidence']:.0%})"
        )
        return data
    # Fallback defaults
    logger.warning("  ⚠ No regime file found — using neutral defaults")
    return {
        "current_regime": "neutral",
        "confidence": 0.5,
        "blend": {"rl": 0.5, "markowitz": 0.5},
        "constraints": {"max_weight": 0.30, "min_weight": 0.02, "cash_floor": 0.0},
    }


# ── Data loading ──────────────────────────────────────────────────────────────


def load_features(tickers: list = None) -> dict:
    if tickers is None:
        tickers = TICKERS
    data = {}
    for ticker in tickers:
        # Try sentiment-enriched first, fall back to original
        for key in [
            f"features/sentiment_enriched/{ticker}/{ticker}_features_v2.parquet",
            f"{S3_FEATURES_PREFIX}/market/{ticker}/{ticker}_features.parquet",
        ]:
            try:
                obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
                df = pd.read_parquet(BytesIO(obj["Body"].read()))
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
                data[ticker] = df
                break
            except Exception:
                continue
        if ticker not in data:
            logger.warning(f"  ⚠ Could not load {ticker}")
    logger.info(f"  ✓ Loaded features for {len(data)} tickers")
    return data


def load_predictions() -> pd.DataFrame:
    pred_path = Path(LOCAL_DATA_DIR) / "predictions.parquet"
    if pred_path.exists():
        df = pd.read_parquet(pred_path)
        df["date"] = pd.to_datetime(df["date"])
        logger.info(f"  ✓ Loaded {len(df)} TFT predictions")
        return df
    logger.warning("  ⚠ No predictions found — run evaluate.py first")
    return pd.DataFrame()


# ── Markowitz optimizer ───────────────────────────────────────────────────────


class MarkowitzOptimizer:
    def __init__(
        self, risk_free_rate=0.05, max_weight=0.30, min_weight=0.02, lookback_days=252
    ):
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.lookback_days = lookback_days

    def build_return_matrix(self, feature_data: dict) -> pd.DataFrame:
        returns = {}
        for ticker, df in feature_data.items():
            col = "return_1d" if "return_1d" in df.columns else None
            if col:
                returns[ticker] = df.set_index("date")[col]
        if not returns:
            raise ValueError("No return data available")
        return pd.DataFrame(returns).dropna().tail(self.lookback_days)

    def optimize(self, feature_data: dict, expected_returns: dict = None) -> dict:
        logger.info("  Running Markowitz optimization...")
        ret_matrix = self.build_return_matrix(feature_data)
        tickers = list(ret_matrix.columns)
        n = len(tickers)

        hist_mu = ret_matrix.mean().values * 252
        if expected_returns:
            tft_signal = np.array([expected_returns.get(t, 0.0) for t in tickers]) * 52
            mu = 0.6 * tft_signal + 0.4 * hist_mu
        else:
            mu = hist_mu

        cov = ret_matrix.cov().values * 252
        w = cp.Variable(n)

        problem = cp.Problem(
            cp.Maximize(mu @ w - 2.0 * cp.quad_form(w, cov)),
            [cp.sum(w) == 1, w >= self.min_weight, w <= self.max_weight],
        )
        try:
            problem.solve(solver=cp.CLARABEL, verbose=False)
            weights = (
                np.array(w.value).flatten()
                if problem.status in ["optimal", "optimal_inaccurate"]
                else np.ones(n) / n
            )
            weights = np.clip(weights, 0, 1)
            weights /= weights.sum()
        except Exception as e:
            logger.warning(f"  ⚠ Solver failed: {e} — equal weight fallback")
            weights = np.ones(n) / n

        result_w = {t: round(float(w), 4) for t, w in zip(tickers, weights)}
        port_ret = float(mu @ weights)
        port_std = float(np.sqrt(weights @ cov @ weights))
        sharpe = (port_ret - self.risk_free_rate) / port_std if port_std > 0 else 0

        logger.info(
            f"  ✓ Markowitz: return={port_ret:.2%} vol={port_std:.2%} sharpe={sharpe:.3f}"
        )
        return {
            "weights": result_w,
            "expected_return": port_ret,
            "expected_vol": port_std,
            "expected_sharpe": sharpe,
        }


# ── RL Environment ────────────────────────────────────────────────────────────


class PortfolioEnv:
    def __init__(
        self,
        feature_data,
        pred_df,
        initial_capital=100_000,
        transaction_cost=0.001,
        max_drawdown_limit=0.15,
    ):
        self.feature_data = feature_data
        self.pred_df = pred_df
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_drawdown_limit = max_drawdown_limit
        self.tickers = list(feature_data.keys())
        self.n_tickers = len(self.tickers)
        self._build_price_matrix()
        self.state_dim = self.n_tickers * 3 + 3
        self.action_dim = self.n_tickers
        self.reset()

    def _build_price_matrix(self):
        returns = {
            t: df.set_index("date")["return_1d"]
            for t, df in self.feature_data.items()
            if "return_1d" in df.columns
        }
        vols = {
            t: df.set_index("date")["vol_21d"]
            for t, df in self.feature_data.items()
            if "vol_21d" in df.columns
        }
        self.return_matrix = pd.DataFrame(returns).dropna()
        self.vol_matrix = (
            pd.DataFrame(vols).reindex(self.return_matrix.index).ffill().fillna(0.2)
        )
        self.dates = self.return_matrix.index.tolist()
        self.pred_lookup = {}
        if not self.pred_df.empty:
            for _, row in self.pred_df.iterrows():
                d = row["date"]
                t = TICKER_NAMES.get(str(row["ticker"]), str(row["ticker"]))
                self.pred_lookup.setdefault(d, {})[t] = row["pred_p50"]
        logger.info(f"  ✓ Env: {len(self.dates)} days | {self.n_tickers} tickers")

    def reset(self):
        self.current_step = min(200, len(self.dates) - 100)
        self.portfolio_value = self.initial_capital
        self.peak_value = self.initial_capital
        self.weights = np.ones(self.n_tickers) / self.n_tickers
        self.returns_history = []
        self.days_since_rebal = 0
        return self._get_state()

    def _get_state(self):
        date = self.dates[self.current_step]
        preds = np.array(
            [self.pred_lookup.get(date, {}).get(t, 0.0) for t in self.tickers]
        )
        vols = np.clip(self.vol_matrix.iloc[self.current_step].values, 0, 1)
        dd = (self.portfolio_value - self.peak_value) / self.peak_value
        return np.concatenate(
            [
                self.weights,
                preds,
                vols,
                [0.0, dd, min(self.days_since_rebal / 20.0, 1.0)],
            ]
        ).astype(np.float32)

    def step(self, action):
        action = np.array(action, dtype=np.float32)
        exp_a = np.exp(action - action.max())
        new_weights = exp_a / exp_a.sum()
        tc_cost = np.abs(new_weights - self.weights).sum() * self.transaction_cost
        self.weights = new_weights
        returns = self.return_matrix.iloc[self.current_step].values
        port_ret = float(np.dot(self.weights, returns)) - tc_cost
        self.portfolio_value *= 1 + port_ret
        self.peak_value = max(self.peak_value, self.portfolio_value)
        dd = (self.portfolio_value - self.peak_value) / self.peak_value
        reward = port_ret - (0.1 * abs(dd) if dd < -self.max_drawdown_limit else 0)
        reward -= 0.01 * float(np.sum(self.weights**2))
        self.returns_history.append(port_ret)
        self.days_since_rebal = (
            0
            if np.abs(new_weights - self.weights).sum() > 0.05
            else self.days_since_rebal + 1
        )
        self.current_step += 1
        done = self.current_step >= len(self.dates) - 1
        return (
            self._get_state() if not done else np.zeros(self.state_dim, np.float32),
            reward,
            done,
            {
                "portfolio_value": self.portfolio_value,
                "weights": self.weights.tolist(),
                "return": port_ret,
            },
        )


class GymWrapper:
    metadata = {"render_modes": []}
    spec = None

    def __init__(self, env):
        import gymnasium as gym

        self.env = env
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(env.state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-3.0, high=3.0, shape=(env.action_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        return self.env.reset(), {}

    def step(self, action):
        s, r, done, info = self.env.step(action)
        return s, r, done, False, info

    def render(self):
        pass


class PPOPortfolioAgent:
    def __init__(self, env):
        self.env = env
        self.model = None

    def train(self, total_timesteps=50_000):
        from stable_baselines3 import PPO

        gym_env = GymWrapper(self.env)
        logger.info(f"  Training PPO for {total_timesteps} timesteps...")
        self.model = PPO(
            "MlpPolicy",
            gym_env,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=0,
            policy_kwargs=dict(net_arch=[256, 256]),
        )
        self.model.learn(total_timesteps=total_timesteps)
        logger.info("  ✓ PPO training complete")

    def predict_weights(self, state):
        action, _ = self.model.predict(state, deterministic=True)
        exp_a = np.exp(action - action.max())
        return exp_a / exp_a.sum()

    def save(self, path="models/rl_agent"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if self.model:
            self.model.save(path)

    def load(self, path="models/rl_agent"):
        from stable_baselines3 import PPO

        self.model = PPO.load(path)


# ── Master allocator ──────────────────────────────────────────────────────────


class AlphaFlowAllocator:
    def __init__(self, markowitz: MarkowitzOptimizer, rl_agent=None):
        self.markowitz = markowitz
        self.rl_agent = rl_agent

    def allocate(
        self, feature_data: dict, pred_df: pd.DataFrame, regime_data: dict
    ) -> dict:
        regime = regime_data["current_regime"]
        rl_blend = regime_data["blend"]["rl"]
        mkw_blend = regime_data["blend"]["markowitz"]
        logger.info(
            f"  Regime: {regime.upper()} | "
            f"{mkw_blend:.0%} Markowitz + {rl_blend:.0%} RL"
        )

        # TFT expected returns
        expected_returns = {}
        if not pred_df.empty:
            latest = pred_df.sort_values("date").groupby("ticker").last()
            for enc, row in latest.iterrows():
                t = TICKER_NAMES.get(str(enc), str(enc))
                expected_returns[t] = float(row["pred_p50"]) / 5

        mkt_result = self.markowitz.optimize(feature_data, expected_returns)
        mkt_weights = mkt_result["weights"]

        if self.rl_agent and self.rl_agent.model:
            env = PortfolioEnv(feature_data, pred_df)
            state = env.reset()
            rl_raw = self.rl_agent.predict_weights(state)
            cols = self.markowitz.build_return_matrix(feature_data).columns
            rl_weights = {t: float(w) for t, w in zip(cols, rl_raw)}
        else:
            rl_weights = mkt_weights

        # Blend
        all_tickers = set(mkt_weights) | set(rl_weights)
        blended = {
            t: round(
                mkw_blend * mkt_weights.get(t, 0.0) + rl_blend * rl_weights.get(t, 0.0),
                4,
            )
            for t in all_tickers
        }
        total = sum(blended.values())
        if total > 0:
            blended = {t: round(w / total, 4) for t, w in blended.items()}
        blended = dict(sorted(blended.items(), key=lambda x: x[1], reverse=True))

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "regime": regime,
            "regime_confidence": regime_data.get("confidence", 0),
            "rl_blend": rl_blend,
            "markowitz_blend": mkw_blend,
            "weights": blended,
            "markowitz_weights": mkt_weights,
            "expected_returns": expected_returns,
            "markowitz_sharpe": mkt_result["expected_sharpe"],
        }


# ── Main ──────────────────────────────────────────────────────────────────────


def run_portfolio_optimization(train_rl: bool = True) -> dict:
    logger.info("=" * 60)
    logger.info("AlphaFlow — Portfolio Optimization Engine")
    logger.info("=" * 60)
    Path("logs").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    logger.info("\nStep 1: Loading regime from HMM detector...")
    regime_data = load_regime()

    logger.info("\nStep 2: Loading feature data...")
    feature_data = load_features()

    logger.info("\nStep 3: Loading TFT predictions...")
    pred_df = load_predictions()

    logger.info("\nStep 4: Markowitz optimization...")
    markowitz = MarkowitzOptimizer(
        risk_free_rate=0.05,
        max_weight=regime_data["constraints"]["max_weight"],
        min_weight=regime_data["constraints"].get("min_weight", 0.02),
        lookback_days=252,
    )

    rl_agent = None
    if train_rl:
        logger.info("\nStep 5: Training RL agent...")
        env = PortfolioEnv(feature_data, pred_df)
        rl_agent = PPOPortfolioAgent(env)
        try:
            rl_agent.train(total_timesteps=50_000)
            rl_agent.save("models/rl_agent")
        except Exception as e:
            logger.warning(f"  ⚠ RL failed: {e} — Markowitz only")
            rl_agent = None

    logger.info("\nStep 6: Computing final allocation...")
    allocator = AlphaFlowAllocator(markowitz, rl_agent)
    allocation = allocator.allocate(feature_data, pred_df, regime_data)

    with open("reports/allocation.json", "w") as f:
        json.dump(allocation, f, indent=2, default=str)

    logger.info("\n" + "=" * 60)
    logger.info("PORTFOLIO ALLOCATION")
    logger.info("=" * 60)
    logger.info(
        f"  Regime  : {allocation['regime'].upper()} "
        f"(confidence {allocation['regime_confidence']:.0%})"
    )
    logger.info(
        f"  Blend   : {allocation['markowitz_blend']:.0%} Markowitz "
        f"+ {allocation['rl_blend']:.0%} RL"
    )
    logger.info(f"  Sharpe  : {allocation['markowitz_sharpe']:.3f}")
    logger.info("")
    logger.info("  FINAL WEIGHTS:")
    for ticker, weight in allocation["weights"].items():
        bar = "█" * int(weight * 40)
        logger.info(f"    {ticker:6s} {bar} {weight:.2%}")
    logger.info("\n  ✓ Saved to reports/allocation.json")
    return allocation


if __name__ == "__main__":
    allocation = run_portfolio_optimization(train_rl=False)
    print("\n── Final Portfolio Allocation ───────────────────")
    print(
        f"  Regime : {allocation['regime'].upper()} "
        f"(confidence {allocation['regime_confidence']:.0%})"
    )
    print(
        f"  Blend  : {allocation['markowitz_blend']:.0%} Markowitz "
        f"+ {allocation['rl_blend']:.0%} RL"
    )
    print()
    for ticker, weight in allocation["weights"].items():
        bar = "█" * int(weight * 40)
        print(f"  {ticker:6s} {bar} {weight:.2%}")
