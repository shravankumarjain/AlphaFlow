"""
models/sentiment/finbert_scorer.py
AlphaFlow — FinBERT Sentiment Scorer
Runs ProsusAI/finbert on EDGAR text sections.
Computes sentiment DELTA (quarter-over-quarter change) — the leading signal.
"""

import logging
import warnings
warnings.filterwarnings("ignore")

import torch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pathlib import Path  # noqa: E402
from datetime import datetime  # noqa: E402, F401
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # noqa: E402
from torch.nn.functional import softmax  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(module)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("finbert_scorer")

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_NAME  = "ProsusAI/finbert"        # best financial sentiment model
MAX_TOKENS  = 512                        # FinBERT's max sequence length
BATCH_SIZE  = 8                          # adjust down if OOM on M2
OUTPUT_DIR  = Path("data/local/sentiment")
INPUT_DIR   = Path("data/local/edgar_text")

# Section weights — MDA and risk factors carry more signal than full text
SECTION_WEIGHTS = {
    "mda"         : 0.35,
    "risk_factors": 0.25,
    "results_ops" : 0.25,
    "outlook"     : 0.15,
    "full_text"   : 0.10,
}


class FinBERTScorer:
    """Runs ProsusAI/finbert on financial text, returns pos/neg/neutral scores."""

    def __init__(self):
        self.device = self._get_device()
        logger.info(f"Loading FinBERT on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
        logger.info("✓ FinBERT loaded")

    @staticmethod
    def _get_device() -> str:
        if torch.backends.mps.is_available():
            return "mps"      # Apple Silicon GPU
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _chunk_text(self, text: str, overlap: int = 50) -> list[str]:
        """
        Split long text into overlapping chunks of MAX_TOKENS.
        Overlap prevents losing context at chunk boundaries.
        """
        tokens = self.tokenizer(
            text,
            return_tensors=None,
            truncation=False,
            add_special_tokens=False,
        )["input_ids"]

        chunks = []
        step   = MAX_TOKENS - overlap - 2  # -2 for [CLS] [SEP]
        for i in range(0, len(tokens), step):
            chunk_ids = tokens[i : i + MAX_TOKENS - 2]
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunks.append(chunk_text)
            if i + MAX_TOKENS >= len(tokens):
                break
        return chunks if chunks else [text[:1000]]

    def score_text(self, text: str) -> dict[str, float]:
        """
        Score a single text string.
        Returns: {"positive": float, "negative": float, "neutral": float, "composite": float}
        composite = positive - negative (range: -1 to +1)
        """
        if not text or len(text.strip()) < 20:
            return {"positive": 0.333, "negative": 0.333, "neutral": 0.333, "composite": 0.0}

        chunks      = self._chunk_text(text)
        all_scores  = []

        for chunk in chunks:
            try:
                inputs = self.tokenizer(
                    chunk,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_TOKENS,
                    padding=True,
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs   = softmax(outputs.logits, dim=-1).cpu().numpy()[0]

                # FinBERT label order: positive=0, negative=1, neutral=2
                all_scores.append({
                    "positive": float(probs[0]),
                    "negative": float(probs[1]),
                    "neutral" : float(probs[2]),
                })
            except Exception as e:
                logger.warning(f"  ⚠ Chunk scoring failed: {e}")
                continue

        if not all_scores:
            return {"positive": 0.333, "negative": 0.333, "neutral": 0.333, "composite": 0.0}

        # Average across chunks (weighted by chunk position — earlier = more important)
        n      = len(all_scores)
        weights = np.array([1.0 / (i + 1) for i in range(n)])
        weights /= weights.sum()

        avg = {
            "positive": float(np.average([s["positive"] for s in all_scores], weights=weights)),
            "negative": float(np.average([s["negative"] for s in all_scores], weights=weights)),
            "neutral" : float(np.average([s["neutral"]  for s in all_scores], weights=weights)),
        }
        avg["composite"] = avg["positive"] - avg["negative"]
        return avg

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score all rows in an EDGAR text DataFrame."""
        results = []
        total   = len(df)

        for idx, row in df.iterrows():
            if idx % 20 == 0:
                logger.info(f"  Scoring row {idx}/{total} | {row['ticker']} {row['form_type']}")

            scores = self.score_text(row["text"])
            results.append({
                "ticker"    : row["ticker"],
                "form_type" : row["form_type"],
                "filed"     : row["filed"],
                "section"   : row["section"],
                **scores,
                "text_length": row.get("text_length", 0),
            })

        return pd.DataFrame(results)


def compute_weighted_filing_score(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate section-level scores into a single filing-level score.
    Uses SECTION_WEIGHTS to emphasise MDA over boilerplate.
    """
    records = []
    for (ticker, form_type, filed), group in scored_df.groupby(["ticker", "form_type", "filed"]):
        weighted_composite = 0.0
        total_weight       = 0.0
        for _, row in group.iterrows():
            w = SECTION_WEIGHTS.get(row["section"], 0.10)
            weighted_composite += w * row["composite"]
            total_weight       += w
        if total_weight > 0:
            weighted_composite /= total_weight
        records.append({
            "ticker"           : ticker,
            "form_type"        : form_type,
            "filed"            : pd.to_datetime(filed),
            "sentiment_score"  : weighted_composite,
            "avg_positive"     : group["positive"].mean(),
            "avg_negative"     : group["negative"].mean(),
            "avg_neutral"      : group["neutral"].mean(),
            "n_sections"       : len(group),
        })
    return pd.DataFrame(records).sort_values(["ticker", "form_type", "filed"])


def compute_sentiment_delta(filing_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Compute quarter-over-quarter sentiment DELTA.
    This is the leading signal — the CHANGE in tone matters more than the absolute level.
    A company moving from neutral to negative is a sell signal BEFORE the price moves.
    """
    deltas = []
    for ticker, group in filing_scores.groupby("ticker"):
        for form_type, fg in group.groupby("form_type"):
            fg = fg.sort_values("filed").reset_index(drop=True)
            fg["sentiment_delta"]         = fg["sentiment_score"].diff()
            fg["sentiment_delta_2q"]      = fg["sentiment_score"].diff(2)  # 2-quarter delta
            fg["sentiment_acceleration"]  = fg["sentiment_delta"].diff()   # 2nd derivative
            fg["sentiment_ma3"]           = fg["sentiment_score"].rolling(3, min_periods=1).mean()
            fg["sentiment_vs_ma3"]        = fg["sentiment_score"] - fg["sentiment_ma3"]  # deviation from trend
            deltas.append(fg)
    return pd.concat(deltas, ignore_index=True)


def build_daily_sentiment_features(
    delta_df   : pd.DataFrame,
    price_dates: pd.DatetimeIndex,
    tickers    : list[str],
) -> pd.DataFrame:
    """
    Forward-fill sentiment scores to daily frequency.
    A 10-Q filed on March 15 affects all trading days until the next filing.
    This is how you turn quarterly filings into a daily ML feature.
    """
    daily_records = []
    for ticker in tickers:
        t_df = delta_df[delta_df["ticker"] == ticker].copy()
        if t_df.empty:
            # No filings — fill with neutral zeros
            for d in price_dates:
                daily_records.append({
                    "date"                    : d,
                    "ticker"                  : ticker,
                    "sentiment_score"         : 0.0,
                    "sentiment_delta"         : 0.0,
                    "sentiment_delta_2q"      : 0.0,
                    "sentiment_acceleration"  : 0.0,
                    "sentiment_vs_ma3"        : 0.0,
                    "filing_age_days"         : 999,
                    "has_recent_filing"       : 0,
                })
            continue

        t_df = t_df.sort_values("filed").drop_duplicates("filed")
        t_df = t_df.set_index("filed")

        # Reindex to daily — forward fill (use the most recent filing score)
        daily_idx = pd.DatetimeIndex(price_dates)
        t_reindexed = t_df[[
            "sentiment_score", "sentiment_delta", "sentiment_delta_2q",
            "sentiment_acceleration", "sentiment_vs_ma3"
        ]].reindex(daily_idx, method="ffill").fillna(0)

        # Filing age — how stale is the most recent filing?
        filing_dates = pd.DatetimeIndex(t_df.index)
        for d in daily_idx:
            past_filings  = filing_dates[filing_dates <= d]
            filing_age    = (d - past_filings[-1]).days if len(past_filings) > 0 else 999
            has_recent    = int(filing_age <= 90)  # filed within last quarter

            row = t_reindexed.loc[d]
            daily_records.append({
                "date"                    : d,
                "ticker"                  : ticker,
                "sentiment_score"         : float(row["sentiment_score"]),
                "sentiment_delta"         : float(row["sentiment_delta"]),
                "sentiment_delta_2q"      : float(row["sentiment_delta_2q"]),
                "sentiment_acceleration"  : float(row["sentiment_acceleration"]),
                "sentiment_vs_ma3"        : float(row["sentiment_vs_ma3"]),
                "filing_age_days"         : filing_age,
                "has_recent_filing"       : has_recent,
            })

    return pd.DataFrame(daily_records)


def run_finbert_scoring(tickers: list = None) -> pd.DataFrame:
    """Full pipeline: load EDGAR text → score → delta → daily features."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load EDGAR text ───────────────────────────────────────────────
    logger.info("Step 1: Loading EDGAR text...")
    combined_path = INPUT_DIR / "all_tickers_edgar_text.parquet"
    if not combined_path.exists():
        logger.error(f"  ✗ EDGAR text not found at {combined_path}")
        logger.error("  Run edgar_downloader.py first.")
        return pd.DataFrame()

    edgar_df = pd.read_parquet(combined_path)

    if tickers is not None:
        edgar_df = edgar_df[edgar_df["ticker"].isin(tickers)]

    logger.info(f"  ✓ Loaded {len(edgar_df)} text sections for {edgar_df['ticker'].nunique()} tickers")

    # ── Step 2: FinBERT scoring ───────────────────────────────────────────────
    logger.info("Step 2: Running FinBERT scoring...")
    scorer       = FinBERTScorer()
    scored_df    = scorer.score_dataframe(edgar_df)
    scored_path  = OUTPUT_DIR / "finbert_section_scores.parquet"
    scored_df.to_parquet(scored_path, index=False)
    logger.info(f"  ✓ Scored {len(scored_df)} sections → {scored_path}")

    # ── Step 3: Aggregate to filing level ─────────────────────────────────────
    logger.info("Step 3: Aggregating to filing level...")
    filing_scores = compute_weighted_filing_score(scored_df)
    filing_path   = OUTPUT_DIR / "filing_level_scores.parquet"
    filing_scores.to_parquet(filing_path, index=False)
    logger.info(f"  ✓ {len(filing_scores)} filing scores → {filing_path}")

    # ── Step 4: Compute sentiment delta ───────────────────────────────────────
    logger.info("Step 4: Computing sentiment deltas...")
    delta_df   = compute_sentiment_delta(filing_scores)
    delta_path = OUTPUT_DIR / "sentiment_deltas.parquet"
    delta_df.to_parquet(delta_path, index=False)
    logger.info(f"  ✓ Delta features computed → {delta_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("FinBERT Scoring Complete")
    logger.info("=" * 60)
    
    all_tickers = delta_df["ticker"].unique().tolist()
    for ticker in all_tickers:
        t_df = delta_df[delta_df["ticker"] == ticker]
        if t_df.empty:
            continue
        latest     = t_df.sort_values("filed").iloc[-1]
        score      = latest["sentiment_score"]
        delta      = latest["sentiment_delta"] if pd.notna(latest["sentiment_delta"]) else 0
        direction  = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        sentiment  = "POSITIVE" if score > 0.05 else ("NEGATIVE" if score < -0.05 else "NEUTRAL")
        logger.info(f"  {ticker:8s} | {sentiment:8s} | score={score:+.3f} | delta={delta:+.3f} {direction}")

    return delta_df


if __name__ == "__main__":
    delta_df = run_finbert_scoring(tickers=None) 
    if not delta_df.empty:
        print("\nTop signals by sentiment delta:")
        top = (
            delta_df.dropna(subset=["sentiment_delta"])
            .sort_values("sentiment_delta", key=abs, ascending=False)
            .head(10)[["ticker","form_type","filed","sentiment_score","sentiment_delta"]]
        )
        print(top.to_string(index=False))