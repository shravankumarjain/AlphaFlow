# portfolio/optimizer/portfolio_optimizer.py
#
# AlphaFlow — Portfolio Optimization Engine
#
# Two optimizers working together:
#
# 1. MARKOWITZ (Classical Baseline)
#    Mean-variance optimization — Nobel Prize winning framework (1952)
#    Finds the portfolio weights that maximize Sharpe ratio
#    given expected returns and covariance matrix
#    Constraint: weights sum to 1, no short selling
#
# 2. RL AGENT (PPO — Proximal Policy Optimization)
#    Learns to allocate capital dynamically based on:
#    - TFT price forecasts (expected returns per ticker)
#    - Macro regime signal (risk-on vs risk-off)
#    - Current portfolio state (existing weights, drawdown)
#    - Volatility regime (expand/contract positions)
#
#    Why RL over Markowitz?
#    Markowitz assumes returns are stationary — they're not.
#    RL adapts to changing market regimes in real time.
#    This is closer to how Aladdin's allocation engine works.
#
# Output: optimal_weights dict — {ticker: weight} summing to 1.0
#
# Run: python portfolio/optimizer/portfolio_optimizer.py

import warnings

warnings.filterwarnings("ignore")

import logging  # noqa: E402
import json  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import cvxpy as cp  # noqa: E402
from pathlib import Path  # noqa: E402
from datetime import datetime  # noqa: E402
import boto3  # noqa: E402
from io import BytesIO  # noqa: E402

import sys  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import (  # noqa: E402
    AWS_REGION,
    S3_BUCKET,
    TICKERS,
    S3_FEATURES_PREFIX,
    LOCAL_DATA_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/portfolio.log"),
    ],
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


# ═══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════


def load_features(tickers: list = None) -> dict:
    """
    Load feature Parquet files from S3 for all tickers.
    Returns dict of {ticker: DataFrame}.
    """
    if tickers is None:
        tickers = TICKERS

    data = {}
    for ticker in tickers:
        s3_key = f"{S3_FEATURES_PREFIX}/market/{ticker}/{ticker}_features.parquet"
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
            df = pd.read_parquet(BytesIO(obj["Body"].read()))
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            data[ticker] = df
        except Exception as e:
            logger.warning(f"  ⚠ Could not load {ticker}: {e}")

    logger.info(f"  ✓ Loaded features for {len(data)} tickers")
    return data


def load_predictions() -> pd.DataFrame:
    """Load TFT predictions from evaluate.py output."""
    pred_path = Path(LOCAL_DATA_DIR) / "predictions.parquet"
    if pred_path.exists():
        pred_df = pd.read_parquet(pred_path)
        pred_df["date"] = pd.to_datetime(pred_df["date"])
        logger.info(f"  ✓ Loaded {len(pred_df)} TFT predictions")
        return pred_df
    else:
        logger.warning("  ⚠ No predictions found — run evaluate.py first")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════
# MARKOWITZ OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════


class MarkowitzOptimizer:
    """
    Mean-Variance Portfolio Optimizer using cvxpy.

    Solves: maximize Sharpe ratio
    Subject to:
        - weights sum to 1
        - all weights >= 0 (long only)
        - max single position = max_weight (concentration limit)
        - minimum position = min_weight (diversification floor)

    This is the industry standard baseline.
    Any quant fund will benchmark against this.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        max_weight: float = 0.40,
        min_weight: float = 0.02,
        lookback_days: int = 252,
    ):
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.lookback_days = lookback_days

    def build_return_matrix(self, feature_data: dict) -> pd.DataFrame:
        """
        Build a returns matrix: rows=dates, cols=tickers.
        Use adjusted close returns from feature data.
        """
        returns = {}
        for ticker, df in feature_data.items():
            if "return_1d" in df.columns:
                ret = df.set_index("date")["return_1d"]
                returns[ticker] = ret
            elif "close" in df.columns:
                ret = df.set_index("date")["close"].pct_change()
                returns[ticker] = ret

        if not returns:
            raise ValueError("No return data available")

        ret_matrix = pd.DataFrame(returns).dropna()
        return ret_matrix.tail(self.lookback_days)

    def optimize(
        self,
        feature_data: dict,
        expected_returns: dict = None,
    ) -> dict:
        """
        Run Markowitz optimization.

        If expected_returns provided (from TFT), use them as mu.
        Otherwise use historical mean returns.

        Returns: {ticker: weight}
        """
        logger.info("  Running Markowitz optimization...")

        # Build returns matrix
        ret_matrix = self.build_return_matrix(feature_data)
        tickers = list(ret_matrix.columns)
        n = len(tickers)

        # Expected returns (mu)
        if expected_returns:
            mu = (
                np.array(
                    [expected_returns.get(t, ret_matrix[t].mean()) for t in tickers]
                )
                * 252
            )  # annualise
        else:
            mu = ret_matrix.mean().values * 252

        # Covariance matrix (annualised)
        cov_matrix = ret_matrix.cov().values * 252

        # ── cvxpy optimization ────────────────────────────────────────
        w = cp.Variable(n)

        # Portfolio return and variance
        port_return = mu @ w
        port_var = cp.quad_form(w, cov_matrix)

        # Sharpe ratio maximisation via quadratic programming
        # We maximise return - lambda * variance (risk-adjusted)
        risk_aversion = 2.0
        objective = cp.Maximize(port_return - risk_aversion * port_var)

        constraints = [
            cp.sum(w) == 1,  # fully invested
            w >= self.min_weight,  # minimum diversification
            w <= self.max_weight,  # concentration limit
        ]

        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.CLARABEL, verbose=False)

            if problem.status not in ["optimal", "optimal_inaccurate"]:
                logger.warning(
                    f"  ⚠ Solver status: {problem.status} — using equal weight"
                )
                weights = np.ones(n) / n
            else:
                weights = np.array(w.value).flatten()
                weights = np.clip(weights, 0, 1)
                weights = weights / weights.sum()  # normalise

        except Exception as e:
            logger.warning(f"  ⚠ Optimization failed: {e} — using equal weight")
            weights = np.ones(n) / n

        result = {ticker: round(float(w), 4) for ticker, w in zip(tickers, weights)}

        # Portfolio metrics
        port_ret = float(mu @ weights)
        port_std = float(np.sqrt(weights @ cov_matrix @ weights))
        sharpe = (port_ret - self.risk_free_rate) / port_std if port_std > 0 else 0

        logger.info(
            f"  ✓ Markowitz: expected_return={port_ret:.2%} vol={port_std:.2%} sharpe={sharpe:.3f}"
        )
        logger.info(f"  ✓ Weights: {result}")

        return {
            "weights": result,
            "expected_return": port_ret,
            "expected_vol": port_std,
            "expected_sharpe": sharpe,
            "method": "markowitz",
        }


# ═══════════════════════════════════════════════════════════════════════
# RL TRADING ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════


class PortfolioEnv:
    """
    Custom Gymnasium-compatible trading environment.

    State space (what the agent observes):
        - Current portfolio weights (n_tickers)
        - TFT predicted returns per ticker (n_tickers)
        - Current volatility regime per ticker (n_tickers)
        - Macro regime signal (1 value: 0=risk-on, 1=neutral, 2=risk-off)
        - Current drawdown (1 value)
        - Days since last rebalance (1 value)

    Action space:
        - Target portfolio weights (n_tickers), continuous [0, 1]
        - Agent outputs raw logits, softmax gives valid weights

    Reward:
        - Daily portfolio return
        - Minus transaction costs for rebalancing
        - Minus drawdown penalty (encourages risk management)
        - Sharpe-adjusted: reward / volatility (encourages consistency)
    """

    def __init__(
        self,
        feature_data: dict,
        pred_df: pd.DataFrame,
        initial_capital: float = 100_000,
        transaction_cost: float = 0.001,
        max_drawdown_limit: float = 0.15,
    ):
        self.feature_data = feature_data
        self.pred_df = pred_df
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_drawdown_limit = max_drawdown_limit

        self.tickers = list(feature_data.keys())
        self.n_tickers = len(self.tickers)

        # Build aligned price/return matrix
        self._build_price_matrix()

        # State and action dimensions
        self.state_dim = (
            self.n_tickers * 3 + 3
        )  # weights + preds + vols + macro + dd + days
        self.action_dim = self.n_tickers

        self.reset()

    def _build_price_matrix(self):
        """Build aligned return matrix across all tickers and dates."""
        returns = {}
        vols = {}
        for ticker, df in self.feature_data.items():
            if "return_1d" in df.columns:
                returns[ticker] = df.set_index("date")["return_1d"]
            if "vol_21d" in df.columns:
                vols[ticker] = df.set_index("date")["vol_21d"]

        self.return_matrix = pd.DataFrame(returns).dropna()
        self.vol_matrix = (
            pd.DataFrame(vols).reindex(self.return_matrix.index).ffill().fillna(0.2)
        )
        self.dates = self.return_matrix.index.tolist()

        # Build prediction lookup: date → {ticker: pred_p50}
        self.pred_lookup = {}
        if not self.pred_df.empty:
            for _, row in self.pred_df.iterrows():
                date = row["date"]
                ticker = TICKER_NAMES.get(str(row["ticker"]), str(row["ticker"]))
                if date not in self.pred_lookup:
                    self.pred_lookup[date] = {}
                self.pred_lookup[date][ticker] = row["pred_p50"]

        logger.info(
            f"  ✓ Env: {len(self.dates)} trading days | {self.n_tickers} tickers"
        )

    def reset(self):
        """Reset environment to start."""
        # Start 200 days in to have history for features
        self.current_step = min(200, len(self.dates) - 100)
        self.portfolio_value = self.initial_capital
        self.peak_value = self.initial_capital
        self.weights = np.ones(self.n_tickers) / self.n_tickers
        self.returns_history = []
        self.days_since_rebal = 0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Construct state vector for current timestep."""
        date = self.dates[self.current_step]

        # Current weights
        w = self.weights

        # TFT predictions for this date
        preds = np.array(
            [self.pred_lookup.get(date, {}).get(t, 0.0) for t in self.tickers]
        )

        # Current volatility
        vols = self.vol_matrix.iloc[self.current_step].values
        vols = np.clip(vols, 0, 1)

        # Macro signal (use VIX regime from first ticker's features)
        macro = 0.0
        first_df = list(self.feature_data.values())[0]
        first_df_date = first_df[first_df["date"] == date]
        if not first_df_date.empty and "macro_regime" in first_df_date.columns:
            macro = float(first_df_date["macro_regime"].values[0])

        # Drawdown
        drawdown = (self.portfolio_value - self.peak_value) / self.peak_value

        # Days since rebalance (normalised)
        days_norm = min(self.days_since_rebal / 20.0, 1.0)

        state = np.concatenate([w, preds, vols, [macro / 2.0, drawdown, days_norm]])
        return state.astype(np.float32)

    def step(self, action: np.ndarray):
        """
        Execute one trading step.

        action: raw weights from RL agent (will be softmaxed)
        Returns: (next_state, reward, done, info)
        """
        # Convert action to valid weights via softmax
        action = np.array(action, dtype=np.float32)
        exp_action = np.exp(action - action.max())
        new_weights = exp_action / exp_action.sum()

        # Transaction costs
        weight_change = np.abs(new_weights - self.weights).sum()
        tc_cost = weight_change * self.transaction_cost

        # Apply new weights
        old_weights = self.weights.copy()  # noqa: F841
        self.weights = new_weights

        # Get actual returns for this step
        date = self.dates[self.current_step]
        returns = self.return_matrix.iloc[self.current_step].values

        # Portfolio return
        port_return = float(np.dot(self.weights, returns)) - tc_cost

        # Update portfolio value
        self.portfolio_value *= 1 + port_return
        self.peak_value = max(self.peak_value, self.portfolio_value)

        # Drawdown
        drawdown = (self.portfolio_value - self.peak_value) / self.peak_value

        # ── Reward function ───────────────────────────────────────────
        # Base reward: portfolio return
        reward = port_return

        # Drawdown penalty: penalise if drawdown exceeds limit
        if drawdown < -self.max_drawdown_limit:
            reward -= 0.1 * abs(drawdown)

        # Diversification bonus: penalise concentration
        herfindahl = float(np.sum(self.weights**2))
        reward -= 0.01 * herfindahl

        # Track history
        self.returns_history.append(port_return)
        self.days_since_rebal = 0 if weight_change > 0.05 else self.days_since_rebal + 1

        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.dates) - 1

        next_state = (
            self._get_state()
            if not done
            else np.zeros(self.state_dim, dtype=np.float32)
        )

        info = {
            "portfolio_value": self.portfolio_value,
            "weights": self.weights.tolist(),
            "return": port_return,
            "drawdown": drawdown,
            "date": str(date),
        }

        return next_state, reward, done, info


# ═══════════════════════════════════════════════════════════════════════
# RL AGENT — PPO with simple MLP policy
# ═══════════════════════════════════════════════════════════════════════


class PPOPortfolioAgent:
    """
    Proximal Policy Optimization agent for portfolio allocation.

    We use Stable-Baselines3's PPO with a custom environment wrapper
    that makes our PortfolioEnv gymnasium-compatible.

    Policy network: MLP with [256, 256] hidden layers
    Value network:  MLP with [256, 256] hidden layers
    """

    def __init__(self, env: PortfolioEnv):
        self.env = env
        self.model = None

    def train(self, total_timesteps: int = 50_000) -> None:
        """Train the PPO agent."""
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.env_util import make_vec_env  # noqa: F401
            import gymnasium as gym  # noqa: F401

            # Wrap our env to be gymnasium compatible
            gym_env = GymWrapper(self.env)

            logger.info(f"  Training PPO agent for {total_timesteps} timesteps...")
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
                ent_coef=0.01,  # exploration bonus
                verbose=1,
                policy_kwargs=dict(net_arch=[256, 256]),
            )

            self.model.learn(total_timesteps=total_timesteps)
            logger.info("  ✓ PPO training complete")

        except Exception as e:
            logger.error(f"  ✗ PPO training failed: {e}")
            raise

    def predict_weights(self, state: np.ndarray) -> np.ndarray:
        """Get portfolio weights from trained agent."""
        if self.model is None:
            raise RuntimeError("Agent not trained — call train() first")
        action, _ = self.model.predict(state, deterministic=True)
        exp_action = np.exp(action - action.max())
        return exp_action / exp_action.sum()

    def save(self, path: str = "models/rl_agent"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if self.model:
            self.model.save(path)
            logger.info(f"  ✓ RL agent saved: {path}")

    def load(self, path: str = "models/rl_agent"):
        from stable_baselines3 import PPO

        self.model = PPO.load(path)
        logger.info(f"  ✓ RL agent loaded: {path}")


# ═══════════════════════════════════════════════════════════════════════
# GYMNASIUM WRAPPER — makes PortfolioEnv compatible with SB3
# ═══════════════════════════════════════════════════════════════════════


class GymWrapper:
    """Wraps PortfolioEnv to be compatible with Stable-Baselines3."""

    metadata = {"render_modes": []}
    spec = None

    def __init__(self, env: PortfolioEnv):
        import gymnasium as gym

        self.env = env
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(env.state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-3.0, high=3.0, shape=(env.action_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        state = self.env.reset()
        return state, {}

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        truncated = False
        return state, reward, done, truncated, info

    def render(self):
        pass


# ═══════════════════════════════════════════════════════════════════════
# ALLOCATION ENGINE — combines Markowitz + RL
# ═══════════════════════════════════════════════════════════════════════


class AlphaFlowAllocator:
    """
    Master allocation engine combining both optimizers.

    Regime-based blending:
    - Risk-ON  (macro_regime=0): 30% Markowitz + 70% RL
    - Neutral  (macro_regime=1): 50% Markowitz + 50% RL
    - Risk-OFF (macro_regime=2): 70% Markowitz + 30% RL

    Rationale: RL is more aggressive and adaptive.
    In risk-off environments we lean on the more conservative
    Markowitz weights to protect capital.
    This is how multi-strategy funds blend signals.
    """

    def __init__(
        self, markowitz: MarkowitzOptimizer, rl_agent: PPOPortfolioAgent = None
    ):
        self.markowitz = markowitz
        self.rl_agent = rl_agent

    def allocate(
        self,
        feature_data: dict,
        pred_df: pd.DataFrame,
        macro_regime: int = 1,
    ) -> dict:
        """
        Compute final portfolio allocation.

        Returns allocation dict with weights, metadata, and reasoning.
        """
        logger.info(
            f"\n  Macro regime: {macro_regime} ({'risk-on' if macro_regime == 0 else 'neutral' if macro_regime == 1 else 'risk-off'})"
        )

        # Extract TFT expected returns per ticker
        expected_returns = {}
        if not pred_df.empty:
            latest = pred_df.sort_values("date").groupby("ticker").last()
            for enc_ticker, row in latest.iterrows():
                real_ticker = TICKER_NAMES.get(str(enc_ticker), str(enc_ticker))
                expected_returns[real_ticker] = float(row["pred_p50"]) / 5

        # ── Markowitz weights ─────────────────────────────────────────
        mkt_result = self.markowitz.optimize(feature_data, expected_returns)
        mkt_weights = mkt_result["weights"]

        # ── RL weights ────────────────────────────────────────────────
        if self.rl_agent and self.rl_agent.model:
            env = PortfolioEnv(feature_data, pred_df)
            state = env.reset()
            rl_raw = self.rl_agent.predict_weights(state)
            rl_weights = {
                t: float(w)
                for t, w in zip(
                    self.markowitz.build_return_matrix(feature_data).columns, rl_raw
                )
            }
        else:
            rl_weights = mkt_weights  # fallback to Markowitz if RL not trained

        # ── Blend weights by macro regime ─────────────────────────────
        rl_blend = [0.7, 0.5, 0.3][macro_regime]
        mkt_blend = 1 - rl_blend

        all_tickers = set(mkt_weights.keys()) | set(rl_weights.keys())
        blended = {}
        for t in all_tickers:
            mw = mkt_weights.get(t, 0.0)
            rw = rl_weights.get(t, 0.0)
            blended[t] = round(mkt_blend * mw + rl_blend * rw, 4)

        # Normalise
        total = sum(blended.values())
        if total > 0:
            blended = {t: round(w / total, 4) for t, w in blended.items()}

        # Sort by weight descending
        blended = dict(sorted(blended.items(), key=lambda x: x[1], reverse=True))

        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "macro_regime": macro_regime,
            "regime_label": ["risk-on", "neutral", "risk-off"][macro_regime],
            "rl_blend": rl_blend,
            "markowitz_blend": mkt_blend,
            "weights": blended,
            "markowitz_weights": mkt_weights,
            "rl_weights": rl_weights,
            "expected_returns": expected_returns,
            "markowitz_sharpe": mkt_result["expected_sharpe"],
        }

        return result


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════


def run_portfolio_optimization(train_rl: bool = True) -> dict:
    """Full portfolio optimization pipeline."""
    logger.info("=" * 60)
    logger.info("AlphaFlow — Portfolio Optimization Engine")
    logger.info("=" * 60)

    Path("logs").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    # ── Step 1: Load data ─────────────────────────────────────────────
    logger.info("\nStep 1: Loading feature data...")
    feature_data = load_features()

    logger.info("\nStep 2: Loading TFT predictions...")
    pred_df = load_predictions()

    # ── Step 3: Markowitz ─────────────────────────────────────────────
    logger.info("\nStep 3: Markowitz optimization...")
    markowitz = MarkowitzOptimizer(
        risk_free_rate=0.05,
        max_weight=0.40,
        min_weight=0.02,
        lookback_days=252,
    )

    # ── Step 4: RL Agent ──────────────────────────────────────────────
    rl_agent = None
    if train_rl:
        logger.info("\nStep 4: Training RL agent...")
        env = PortfolioEnv(feature_data, pred_df)
        rl_agent = PPOPortfolioAgent(env)
        try:
            rl_agent.train(total_timesteps=50_000)
            rl_agent.save("models/rl_agent")
        except Exception as e:
            logger.warning(f"  ⚠ RL training failed: {e} — using Markowitz only")
            rl_agent = None

    # ── Step 5: Allocate ──────────────────────────────────────────────
    logger.info("\nStep 5: Computing final allocation...")
    allocator = AlphaFlowAllocator(markowitz, rl_agent)

    # Get current macro regime from features
    macro_regime = 1  # default neutral
    first_ticker = list(feature_data.keys())[0]
    if "macro_regime" in feature_data[first_ticker].columns:
        macro_regime = int(feature_data[first_ticker]["macro_regime"].dropna().iloc[-1])

    allocation = allocator.allocate(feature_data, pred_df, macro_regime)

    # ── Step 6: Save & report ─────────────────────────────────────────
    output_path = "reports/allocation.json"
    with open(output_path, "w") as f:
        json.dump(allocation, f, indent=2, default=str)

    logger.info("\n" + "=" * 60)
    logger.info("PORTFOLIO ALLOCATION")
    logger.info("=" * 60)
    logger.info(f"  Regime  : {allocation['regime_label']}")
    logger.info(
        f"  Blend   : {allocation['markowitz_blend']:.0%} Markowitz + {allocation['rl_blend']:.0%} RL"
    )
    logger.info(
        f"  Sharpe  : {allocation['markowitz_sharpe']:.3f} (Markowitz baseline)"
    )
    logger.info("")
    logger.info("  FINAL WEIGHTS:")
    for ticker, weight in allocation["weights"].items():
        bar = "█" * int(weight * 40)
        logger.info(f"    {ticker:6s} {bar} {weight:.2%}")

    logger.info(f"\n  ✓ Saved to {output_path}")
    return allocation


if __name__ == "__main__":
    allocation = run_portfolio_optimization(train_rl=True)

    print("\n── Final Portfolio Allocation ───────────────────")
    print(f"  Regime : {allocation['regime_label']}")
    print(
        f"  Blend  : {allocation['markowitz_blend']:.0%} Markowitz + {allocation['rl_blend']:.0%} RL"
    )
    print()
    for ticker, weight in allocation["weights"].items():
        bar = "█" * int(weight * 40)
        print(f"  {ticker:6s} {bar} {weight:.2%}")
    print("\n  Saved: reports/allocation.json")
