# data_pipeline/streaming/producers.py
#
# AlphaFlow — Real-time Data Producers
#
# Two producers run in parallel threads:
#
#   EquityProducer   — polls yfinance every 60s during market hours
#                      publishes to alphaflow.prices.equity
#
#   CryptoProducer   — Binance WebSocket, real-time tick data 24/7
#                      publishes to alphaflow.prices.crypto
#
#   NewsProducer     — polls yfinance news every 5 mins
#                      publishes to alphaflow.news.raw
#
# Message format (all topics):
#   {
#     "ticker"    : "AAPL",
#     "price"     : 182.34,
#     "volume"    : 1234567,
#     "timestamp" : "2026-03-17T14:30:00Z",
#     "source"    : "yfinance|binance",
#     "asset_class": "equity|crypto"
#   }
#
# Run:
#   python data_pipeline/streaming/producers.py
#   (runs until Ctrl+C)

import json
import time
import logging
import threading
import websocket
import requests
from datetime import datetime, timezone
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import TICKERS, MULTI_ASSET_TICKERS  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("producers")

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC_EQUITY = "alphaflow.prices.equity"
TOPIC_CRYPTO = "alphaflow.prices.crypto"
TOPIC_NEWS = "alphaflow.news.raw"

CRYPTO_MAP = {
    "BTC": "btcusdt",
    "ETH": "ethusdt",
    "SOL": "solusdt",
    "BNB": "bnbusdt",
}


def get_kafka_producer():
    """Create Kafka producer with retry logic."""
    try:
        from kafka import KafkaProducer

        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks=1,
            retries=3,
            request_timeout_ms=5000,
            linger_ms=10,  # batch small messages
        )
        logger.info("  ✓ Kafka producer connected")
        return producer
    except Exception as e:
        logger.error(f"  ✗ Kafka connection failed: {e}")
        logger.info("  → Running in DRY RUN mode (no Kafka needed)")
        return None


def publish(producer, topic: str, key: str, message: dict):
    """Publish message to Kafka topic. Falls back to log if no producer."""
    if producer:
        producer.send(topic, key=key, value=message)
    else:
        logger.info(
            f"  [DRY RUN] {topic} | {key}: {message.get('price', message.get('close', '?'))}"
        )


# ── EQUITY PRODUCER ───────────────────────────────────────────────────


class EquityProducer:
    """
    Polls yfinance every 60 seconds for latest prices.
    During market hours: 9:30 AM - 4:00 PM ET Mon-Fri.
    Outside hours: polls every 5 mins for pre/post-market data.
    """

    def __init__(self, producer, tickers: list = None):
        self.producer = producer
        self.tickers = tickers or TICKERS
        self.running = False

    def _is_market_hours(self) -> bool:
        """Check if US equity market is open."""
        now = datetime.now(timezone.utc)
        # Mon=0, Fri=4
        if now.weekday() > 4:
            return False
        # 13:30 - 20:00 UTC = 9:30 AM - 4:00 PM ET
        market_open = now.replace(hour=13, minute=30, second=0)
        market_close = now.replace(hour=20, minute=0, second=0)
        return market_open <= now <= market_close

    def _fetch_and_publish(self):
        """Fetch latest prices for all equity tickers and publish."""
        import yfinance as yf

        try:
            # Batch download for efficiency
            data = yf.download(
                self.tickers,
                period="1d",
                interval="1m",
                progress=False,
                auto_adjust=True,
            )

            if data.empty:
                return

            # Get latest row
            if hasattr(data.columns, "levels"):
                close_data = data["Close"].iloc[-1]
                vol_data = data["Volume"].iloc[-1]
            else:
                close_data = data["Close"].iloc[-1:].squeeze()
                vol_data = data["Volume"].iloc[-1:].squeeze()

            published = 0
            for ticker in self.tickers:
                try:
                    price = float(close_data[ticker]) if ticker in close_data else None
                    volume = int(vol_data[ticker]) if ticker in vol_data else 0

                    if price and price > 0:
                        msg = {
                            "ticker": ticker,
                            "price": round(price, 4),
                            "volume": volume,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "source": "yfinance",
                            "asset_class": "equity",
                        }
                        publish(self.producer, TOPIC_EQUITY, ticker, msg)
                        published += 1
                except Exception:
                    continue

            logger.info(f"  [Equity] Published {published}/{len(self.tickers)} prices")

        except Exception as e:
            logger.warning(f"  [Equity] Fetch error: {e}")

    def run(self):
        """Run the equity producer loop."""
        self.running = True
        logger.info(f"  ✓ Equity producer started | {len(self.tickers)} tickers")

        while self.running:
            self._fetch_and_publish()
            # Poll every 60s during market hours, 300s otherwise
            interval = 60 if self._is_market_hours() else 300
            time.sleep(interval)

    def stop(self):
        self.running = False


# ── CRYPTO PRODUCER ───────────────────────────────────────────────────


class CryptoProducer:
    """
    Connects to Binance WebSocket for real-time tick data.
    No API key needed — public endpoint.
    Updates every ~1 second per ticker.
    """

    def __init__(self, producer, symbols: list = None):
        self.producer = producer
        self.symbols = symbols or list(CRYPTO_MAP.values())
        self.running = False
        self.ws = None

    def _on_message(self, ws, message):
        """Handle incoming WebSocket tick message."""
        try:
            data = json.loads(message)

            # Binance combined stream format
            if "data" in data:
                data = data["data"]

            symbol = data.get("s", "").upper()  # e.g. BTCUSDT
            price = float(data.get("c", 0))  # current price
            volume = float(data.get("v", 0))  # 24h volume

            # Map back to our ticker name
            ticker = next(
                (k for k, v in CRYPTO_MAP.items() if v.upper() == symbol),
                symbol.replace("USDT", ""),
            )

            if price > 0:
                msg = {
                    "ticker": ticker,
                    "price": round(price, 4),
                    "volume": round(volume, 2),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": "binance_ws",
                    "asset_class": "crypto",
                    "symbol": symbol,
                }
                publish(self.producer, TOPIC_CRYPTO, ticker, msg)

        except Exception as e:
            logger.debug(f"  [Crypto] Message parse error: {e}")

    def _on_error(self, ws, error):
        logger.warning(f"  [Crypto] WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        logger.info(f"  [Crypto] WebSocket closed: {close_status_code}")
        # Auto-reconnect after 5 seconds
        if self.running:
            time.sleep(5)
            self.run()

    def _on_open(self, ws):
        logger.info(f"  ✓ Crypto WebSocket connected | {len(self.symbols)} pairs")

    def run(self):
        self.running = True
        import ssl

        streams = "/".join([f"{s}@ticker" for s in self.symbols])
        url = f"wss://stream.binance.com:9443/stream?streams={streams}"
        self.ws = websocket.WebSocketApp(
            url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        try:
            self.ws.run_forever(
                ping_interval=30,
                ping_timeout=10,
                sslopt={"cert_reqs": ssl.CERT_NONE},  # ← fixes SSL
            )
        except Exception as e:
            logger.warning(f"WebSocket failed: {e}")
            self._run_rest_fallback()

    def _run_rest_fallback(self):
        """REST API fallback when WebSocket unavailable."""
        while self.running:
            try:
                for ticker, symbol in CRYPTO_MAP.items():
                    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol.upper()}USDT"
                    resp = requests.get(url, timeout=5)
                    if resp.status_code == 200:
                        price = float(resp.json()["price"])
                        msg = {
                            "ticker": ticker,
                            "price": round(price, 4),
                            "volume": 0,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "source": "binance_rest",
                            "asset_class": "crypto",
                        }
                        publish(self.producer, TOPIC_CRYPTO, ticker, msg)
            except Exception as e:
                logger.warning(f"  [Crypto] REST error: {e}")
            time.sleep(10)

    def stop(self):
        self.running = False
        if self.ws:
            self.ws.close()


# ── NEWS PRODUCER ─────────────────────────────────────────────────────


class NewsProducer:
    """
    Polls yfinance news every 5 minutes.
    Publishes new articles to alphaflow.news.raw.
    Deduplicates using URL hash.
    """

    def __init__(self, producer, tickers: list = None):
        self.producer = producer
        self.tickers = tickers or TICKERS[:10]  # top 10 for rate limits
        self.seen_urls = set()
        self.running = False

    def _fetch_and_publish(self):
        import yfinance as yf

        new_articles = 0

        for ticker in self.tickers:
            try:
                t = yf.Ticker(ticker)
                for item in t.news or []:
                    url = item.get("content", {}).get("canonicalUrl", {}).get("url", "")
                    if url and url not in self.seen_urls:
                        self.seen_urls.add(url)
                        msg = {
                            "ticker": ticker,
                            "title": item.get("content", {}).get("title", ""),
                            "url": url,
                            "published": item.get("content", {}).get("pubDate", ""),
                            "source": item.get("content", {})
                            .get("provider", {})
                            .get("displayName", ""),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "asset_class": "equity",
                        }
                        publish(self.producer, TOPIC_NEWS, ticker, msg)
                        new_articles += 1
            except Exception:
                continue

        if new_articles > 0:
            logger.info(f"  [News] Published {new_articles} new articles")

    def run(self):
        self.running = True
        logger.info(f"  ✓ News producer started | {len(self.tickers)} tickers")
        while self.running:
            self._fetch_and_publish()
            time.sleep(300)  # every 5 mins

    def stop(self):
        self.running = False


# ── ORCHESTRATOR ──────────────────────────────────────────────────────


def run_all_producers():
    """
    Start all producers in parallel threads.
    Runs until Ctrl+C.
    """
    logger.info("=" * 60)
    logger.info("AlphaFlow — Real-time Data Producers")
    logger.info("=" * 60)
    logger.info(f"  Equity tickers : {len(TICKERS)}")
    logger.info(f"  Crypto pairs   : {len(CRYPTO_MAP)}")
    logger.info(f"  Kafka broker   : {KAFKA_BOOTSTRAP}")
    logger.info("=" * 60)

    producer = get_kafka_producer()

    # Create producers
    equity_prod = EquityProducer(producer, TICKERS)
    crypto_prod = CryptoProducer(producer, list(CRYPTO_MAP.values()))
    news_prod = NewsProducer(producer, TICKERS[:10])

    # Start in threads
    threads = [
        threading.Thread(target=equity_prod.run, name="equity", daemon=True),
        threading.Thread(target=crypto_prod.run, name="crypto", daemon=True),
        threading.Thread(target=news_prod.run, name="news", daemon=True),
    ]

    for t in threads:
        t.start()
        logger.info(f"  ✓ Started {t.name} producer thread")

    logger.info("\n  All producers running. Press Ctrl+C to stop.\n")

    try:
        while True:
            # Health check every 60s
            alive = [t.name for t in threads if t.is_alive()]
            dead = [t.name for t in threads if not t.is_alive()]
            if dead:
                logger.warning(f"  ⚠ Dead threads: {dead}")
            logger.info(f"  Heartbeat | Active: {alive}")
            time.sleep(60)

    except KeyboardInterrupt:
        logger.info("\n  Stopping producers...")
        equity_prod.stop()
        crypto_prod.stop()
        news_prod.stop()

    finally:
        if producer:
            producer.flush()
            producer.close()
        logger.info("  ✓ All producers stopped cleanly")


if __name__ == "__main__":
    run_all_producers()
