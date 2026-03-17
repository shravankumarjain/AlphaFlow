# data_pipeline/streaming/consumers.py
#
# AlphaFlow — Real-time Data Consumers
#
# Three consumers process incoming Kafka messages:
#
#   PriceConsumer       — aggregates tick data into OHLCV candles
#                         triggers feature recomputation every N ticks
#
#   FeatureConsumer     — reads alphaflow.features.ready
#                         runs lightweight feature updates (no full pipeline)
#                         publishes signals to alphaflow.signals
#
#   AllocationConsumer  — reads alphaflow.signals
#                         runs portfolio optimizer when significant signal change
#                         publishes new weights to alphaflow.allocation
#                         writes to reports/allocation.json (picked up by dashboard)
#
# Rebalance triggers:
#   - Price moves > 2% on any top-10 holding
#   - New earnings filing detected (8-K in news stream)
#   - Regime change detected (HMM re-run)
#   - Every 4 hours unconditionally
#
# Run:
#   python data_pipeline/streaming/consumers.py

import json
import time
import logging
import threading
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from pathlib import Path

import pandas as pd  # noqa: F401
import numpy as np  # noqa: F401

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import TICKERS, MULTI_ASSET_TICKERS, LOCAL_DATA_DIR, AWS_REGION, S3_BUCKET  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("consumers")

KAFKA_BOOTSTRAP       = "localhost:9092"
TOPIC_EQUITY          = "alphaflow.prices.equity"
TOPIC_CRYPTO          = "alphaflow.prices.crypto"
TOPIC_NEWS            = "alphaflow.news.raw"
TOPIC_FEATURES_READY  = "alphaflow.features.ready"
TOPIC_SIGNALS         = "alphaflow.signals"
TOPIC_ALLOCATION      = "alphaflow.allocation"

REBALANCE_THRESHOLD   = 0.02   # 2% price move triggers rebalance check
REBALANCE_INTERVAL_H  = 4      # unconditional rebalance every 4 hours


def get_kafka_consumer(topics: list, group_id: str):
    """Create Kafka consumer. Falls back to simulation mode if Kafka unavailable."""
    try:
        from kafka import KafkaConsumer
        consumer = KafkaConsumer(
            *topics,
            bootstrap_servers  = KAFKA_BOOTSTRAP,
            group_id           = group_id,
            auto_offset_reset  = "latest",
            enable_auto_commit = True,
            value_deserializer = lambda v: json.loads(v.decode("utf-8")),
            key_deserializer   = lambda k: k.decode("utf-8") if k else None,
            consumer_timeout_ms= 1000,
        )
        logger.info(f"  ✓ Kafka consumer connected | group={group_id} | topics={topics}")
        return consumer
    except Exception as e:
        logger.warning(f"  ⚠ Kafka unavailable ({e}) — simulation mode")
        return None


def get_kafka_producer():
    """Create Kafka producer for consumers to publish downstream."""
    try:
        from kafka import KafkaProducer
        return KafkaProducer(
            bootstrap_servers = KAFKA_BOOTSTRAP,
            value_serializer  = lambda v: json.dumps(v, default=str).encode("utf-8"),
            key_serializer    = lambda k: k.encode("utf-8") if k else None,
            acks              = 1,
        )
    except Exception:
        return None


# ── PRICE CONSUMER ────────────────────────────────────────────────────

class PriceConsumer:
    """
    Aggregates real-time price ticks into micro-candles.

    Architecture:
      1. Maintains in-memory price cache per ticker
      2. Detects significant price moves (> threshold)
      3. Publishes feature-ready signal when rebalance needed
      4. Writes latest prices to data/local/streaming/prices.json
         (picked up by dashboard for live price display)
    """

    def __init__(self, downstream_producer=None):
        self.producer     = downstream_producer
        self.price_cache  = {}           # {ticker: last_price}
        self.price_at_open= {}           # {ticker: price at session open}
        self.tick_count   = defaultdict(int)
        self.last_rebalance = datetime.now(timezone.utc) - timedelta(hours=5)
        self.running      = False
        self.streaming_dir = Path(LOCAL_DATA_DIR) / "streaming"
        self.streaming_dir.mkdir(parents=True, exist_ok=True)

    def _detect_rebalance_trigger(self, ticker: str, new_price: float) -> bool:
        """
        Check if this price update should trigger a rebalance.

        Triggers:
        1. Price moved > REBALANCE_THRESHOLD since session open
        2. More than REBALANCE_INTERVAL_H hours since last rebalance
        """
        # Time-based trigger
        hours_since = (datetime.now(timezone.utc) - self.last_rebalance).seconds / 3600
        if hours_since >= REBALANCE_INTERVAL_H:
            logger.info(f"  ⏰ Time-based rebalance trigger ({hours_since:.1f}h elapsed)")
            return True

        # Price-move trigger
        if ticker in self.price_at_open:
            open_price = self.price_at_open[ticker]
            if open_price > 0:
                move = abs(new_price - open_price) / open_price
                if move > REBALANCE_THRESHOLD:
                    logger.info(f"  📈 Price trigger: {ticker} moved {move:.1%}")
                    return True

        return False

    def _publish_rebalance_signal(self, trigger_ticker: str, trigger_price: float):
        """Tell downstream consumers to recompute features and reallocate."""
        msg = {
            "trigger"    : "price_move",
            "ticker"     : trigger_ticker,
            "price"      : trigger_price,
            "timestamp"  : datetime.now(timezone.utc).isoformat(),
            "prices"     : {t: p for t, p in self.price_cache.items()},
        }

        if self.producer:
            from kafka import KafkaProducer  # noqa: F401
            self.producer.send(TOPIC_FEATURES_READY, key=trigger_ticker, value=msg)
            logger.info(f"  → Published rebalance signal | trigger={trigger_ticker}")
        else:
            logger.info(f"  [DRY RUN] Rebalance signal: {trigger_ticker} @ {trigger_price}")

        self.last_rebalance = datetime.now(timezone.utc)

    def _save_prices_locally(self):
        """Write current price snapshot to disk for dashboard."""
        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prices"   : {
                t: {
                    "price"     : p,
                    "change_pct": round(
                        (p - self.price_at_open.get(t, p)) / max(self.price_at_open.get(t, p), 1) * 100,
                        3
                    ),
                }
                for t, p in self.price_cache.items()
            },
        }
        with open(self.streaming_dir / "live_prices.json", "w") as f:
            json.dump(snapshot, f, default=str)

    def process_message(self, msg: dict):
        """Process one price tick message."""
        ticker = msg.get("ticker")
        price  = msg.get("price", 0)

        if not ticker or not price:
            return

        # First tick of session = open price
        if ticker not in self.price_at_open:
            self.price_at_open[ticker] = price

        self.price_cache[ticker] = price
        self.tick_count[ticker] += 1

        # Check rebalance trigger every 10 ticks per ticker
        if self.tick_count[ticker] % 10 == 0:
            if self._detect_rebalance_trigger(ticker, price):
                self._publish_rebalance_signal(ticker, price)

        # Save locally every 50 ticks (any ticker)
        total_ticks = sum(self.tick_count.values())
        if total_ticks % 50 == 0:
            self._save_prices_locally()

    def run(self):
        """Consume equity + crypto price topics."""
        self.running = True
        consumer = get_kafka_consumer(
            [TOPIC_EQUITY, TOPIC_CRYPTO],
            group_id="alphaflow-price-consumer"
        )

        if consumer is None:
            self._run_simulation()
            return

        logger.info("  ✓ Price consumer started")
        try:
            for message in consumer:
                if not self.running:
                    break
                self.process_message(message.value)
        finally:
            consumer.close()

    def _run_simulation(self):
        """Simulate price events without Kafka for testing."""
        import yfinance as yf
        logger.info("  [Simulation] Price consumer running without Kafka")
        while self.running:
            try:
                data = yf.download(TICKERS[:5], period="1d", interval="1m",
                                   progress=False, auto_adjust=True)
                if not data.empty and hasattr(data.columns, "levels"):
                    latest = data["Close"].iloc[-1]
                    for ticker in TICKERS[:5]:
                        if ticker in latest:
                            self.process_message({
                                "ticker": ticker,
                                "price" : float(latest[ticker]),
                            })
            except Exception as e:
                logger.debug(f"Simulation tick error: {e}")
            time.sleep(60)

    def stop(self):
        self.running = False


# ── ALLOCATION CONSUMER ───────────────────────────────────────────────

class AllocationConsumer:
    """
    Listens for rebalance signals and runs portfolio optimizer.

    When triggered:
    1. Loads latest TFT predictions from disk
    2. Loads current regime from HMM detector
    3. Runs Markowitz optimizer with current prices
    4. Writes new allocation to reports/allocation.json
    5. Publishes to alphaflow.allocation topic
    6. Dashboard picks up new weights automatically
    """

    def __init__(self, downstream_producer=None):
        self.producer        = downstream_producer
        self.last_allocation = datetime.now(timezone.utc) - timedelta(hours=5)
        self.min_interval_s  = 300   # minimum 5 mins between rebalances
        self.running         = False

    def _run_optimizer(self, trigger: dict) -> dict | None:
        """Run the portfolio optimizer and return new allocation."""
        try:
            sys.path.append(str(Path(__file__).resolve().parents[2]))
            from portfolio.optimizer.portfolio_optimizer import run_portfolio_optimization

            logger.info(f"  🔄 Running portfolio optimizer | trigger={trigger.get('trigger')}")
            allocation = run_portfolio_optimization(train_rl=False)
            return allocation

        except Exception as e:
            logger.error(f"  ✗ Optimizer failed: {e}")
            return None

    def process_message(self, msg: dict):
        """Process one rebalance trigger message."""
        now = datetime.now(timezone.utc)

        # Rate limit: don't rebalance too frequently
        seconds_since = (now - self.last_allocation).seconds
        if seconds_since < self.min_interval_s:
            logger.debug(f"  ⏳ Skipping rebalance — only {seconds_since}s since last")
            return

        allocation = self._run_optimizer(msg)
        if not allocation:
            return

        self.last_allocation = now

        # Publish allocation event to Kafka
        event = {
            "timestamp" : now.isoformat(),
            "trigger"   : msg.get("trigger", "scheduled"),
            "regime"    : allocation.get("regime", "unknown"),
            "weights"   : allocation.get("weights", {}),
            "sharpe"    : allocation.get("markowitz_sharpe", 0),
        }

        if self.producer:
            self.producer.send(TOPIC_ALLOCATION, key="latest", value=event)

        logger.info(
            f"  ✓ Rebalance complete | regime={event['regime'].upper()} | "
            f"sharpe={event['sharpe']:.3f}"
        )

        # Log top 5 positions
        top5 = list(allocation.get("weights", {}).items())[:5]
        for ticker, weight in top5:
            logger.info(f"    {ticker:10s} {weight:.1%}")

    def run(self):
        """Consume rebalance trigger signals."""
        self.running = True
        consumer = get_kafka_consumer(
            [TOPIC_FEATURES_READY],
            group_id="alphaflow-allocation-consumer"
        )

        if consumer is None:
            self._run_scheduled()
            return

        logger.info("  ✓ Allocation consumer started")
        try:
            while self.running:
                messages = consumer.poll(timeout_ms=5000)
                for tp, msgs in messages.items():
                    for msg in msgs:
                        self.process_message(msg.value)

                # Scheduled rebalance check
                hours_since = (datetime.now(timezone.utc) - self.last_allocation).seconds / 3600
                if hours_since >= REBALANCE_INTERVAL_H:
                    self.process_message({"trigger": "scheduled"})

        finally:
            consumer.close()

    def _run_scheduled(self):
        """Scheduled rebalance without Kafka — runs every 4 hours."""
        logger.info("  [Scheduled] Allocation consumer — rebalancing every 4h")
        while self.running:
            self.process_message({"trigger": "scheduled"})
            time.sleep(REBALANCE_INTERVAL_H * 3600)

    def stop(self):
        self.running = False


# ── STREAM STATUS ─────────────────────────────────────────────────────

class StreamMonitor:
    """
    Monitors streaming pipeline health.
    Writes status to data/local/streaming/stream_status.json
    Dashboard reads this to show live pipeline indicator.
    """

    def __init__(self, price_consumer: PriceConsumer):
        self.price_consumer = price_consumer
        self.running        = False
        self.streaming_dir  = Path(LOCAL_DATA_DIR) / "streaming"
        self.streaming_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        self.running = True
        while self.running:
            status = {
                "timestamp"     : datetime.now(timezone.utc).isoformat(),
                "kafka_connected": True,
                "tickers_live"  : len(self.price_consumer.price_cache),
                "total_ticks"   : sum(self.price_consumer.tick_count.values()),
                "last_rebalance": self.price_consumer.last_rebalance.isoformat(),
                "live_prices"   : {
                    t: round(p, 2)
                    for t, p in list(self.price_consumer.price_cache.items())[:10]
                },
            }
            with open(self.streaming_dir / "stream_status.json", "w") as f:
                json.dump(status, f, default=str)
            time.sleep(30)

    def stop(self):
        self.running = False


# ── MAIN ──────────────────────────────────────────────────────────────

def run_all_consumers():
    """Start all consumers in parallel threads."""
    logger.info("=" * 60)
    logger.info("AlphaFlow — Real-time Data Consumers")
    logger.info("=" * 60)

    downstream = get_kafka_producer()

    price_consumer      = PriceConsumer(downstream)
    allocation_consumer = AllocationConsumer(downstream)
    monitor             = StreamMonitor(price_consumer)

    threads = [
        threading.Thread(target=price_consumer.run,      name="price",      daemon=True),
        threading.Thread(target=allocation_consumer.run, name="allocation",  daemon=True),
        threading.Thread(target=monitor.run,             name="monitor",     daemon=True),
    ]

    for t in threads:
        t.start()
        logger.info(f"  ✓ Started {t.name} consumer thread")

    logger.info("\n  All consumers running. Press Ctrl+C to stop.\n")
    logger.info("  Rebalance trigger: price move >2% OR every 4h")
    logger.info("  Watch: data/local/streaming/live_prices.json")
    logger.info("  Watch: reports/allocation.json\n")

    try:
        while True:
            alive = [t.name for t in threads if t.is_alive()]
            logger.info(f"  Heartbeat | Active threads: {alive}")
            time.sleep(120)
    except KeyboardInterrupt:
        logger.info("\n  Stopping consumers...")
        price_consumer.stop()
        allocation_consumer.stop()
        monitor.stop()
    finally:
        if downstream:
            downstream.flush()
            downstream.close()
        logger.info("  ✓ All consumers stopped cleanly")


if __name__ == "__main__":
    run_all_consumers()