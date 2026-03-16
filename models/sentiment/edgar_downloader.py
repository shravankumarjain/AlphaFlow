"""
models/sentiment/edgar_downloader.py
AlphaFlow — EDGAR Full Text Downloader
Correctly navigates the filing index to find the actual narrative document.
"""

import re, time, logging, requests  # noqa: E401
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(module)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("edgar_downloader")

EDGAR_BASE = "https://data.sec.gov"
HEADERS = {
    "User-Agent": "AlphaFlow research@alphaflow.ai",
    "Accept-Encoding": "gzip, deflate",
}

TICKER_CIK = {
    "AAPL" : "0000320193",
    "MSFT" : "0000789019",
    "GOOGL": "0001652044",
    "AMZN" : "0001018724",
    "JPM"  : "0000019617",
    "JNJ"  : "0000200406",
    "XOM"  : "0000034088",
    "BRK-B": "0001067983",
    "TSLA" : "0001318605",
}

FORM_TYPES = ["10-K", "10-Q", "8-K"]
OUTPUT_DIR = Path("data/local/edgar_text")


def get_filings_index(cik: str, form_type: str, start_date: str, end_date: str) -> list:
    """Fetch filing list from SEC submissions API."""
    cik_padded = cik.lstrip("0").zfill(10)
    url = f"{EDGAR_BASE}/submissions/CIK{cik_padded}.json"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        time.sleep(0.12)
    except Exception as e:
        logger.warning(f"  ⚠ Submissions API failed for CIK {cik}: {e}")
        return []

    filings = []
    recent = data.get("filings", {}).get("recent", {})
    forms       = recent.get("form", [])
    filed_dates = recent.get("filingDate", [])
    accessions  = recent.get("accessionNumber", [])

    for i, form in enumerate(forms):
        if form != form_type:
            continue
        filed = filed_dates[i] if i < len(filed_dates) else ""
        if not (start_date <= filed <= end_date):
            continue
        acc = accessions[i] if i < len(accessions) else ""
        filings.append({"cik": cik, "form_type": form_type,
                         "filed": filed, "accession": acc})
    return filings


def get_narrative_doc_url(cik: str, accession: str) -> str | None:
    """
    Fetch the filing index JSON and find the actual narrative HTML document.
    Strategy: skip XBRL/XML files, find the largest .htm file — that's the 10-K/10-Q text.
    """
    cik_num  = cik.lstrip("0")
    acc_path = accession.replace("-", "")
    index_url = f"https://data.sec.gov/Archives/edgar/data/{cik_num}/{acc_path}/{accession}-index.json"

    try:
        resp = requests.get(index_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        index_data = resp.json()
        time.sleep(0.12)
    except Exception as e:  # noqa: F841
        # fallback: try the htm index page
        return get_narrative_doc_url_htm(cik, accession)

    # Find the best document — largest .htm that isn't XBRL/exhibit
    docs = index_data.get("documents", [])
    candidates = []
    for doc in docs:
        name = doc.get("name", "").lower()
        dtype = doc.get("type", "").upper()
        size  = doc.get("size", 0) or 0

        # Skip exhibits, XBRL inline, and XML
        if any(x in name for x in ["ex-", "ex_", "xbrl", "r1.", "r2.", "r3."]):
            continue
        if dtype.startswith("EX-") or dtype in ("XML", "GRAPHIC"):
            continue
        if name.endswith(".htm") or name.endswith(".html"):
            candidates.append((size, name))

    if not candidates:
        return get_narrative_doc_url_htm(cik, accession)

    # Pick the largest htm file — it's the main narrative document
    candidates.sort(reverse=True)
    best_name = candidates[0][1]
    return f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{acc_path}/{best_name}"


def get_narrative_doc_url_htm(cik: str, accession: str) -> str | None:
    """Fallback: parse the HTML index page to find document links."""
    cik_num  = cik.lstrip("0")
    acc_path = accession.replace("-", "")
    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{acc_path}/{accession}-index.htm"

    try:
        resp = requests.get(index_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        time.sleep(0.12)

        # Find all document links — pick largest non-exhibit htm
        links = re.findall(
            r'href="(/Archives/edgar/data/[^"]+\.htm)"[^>]*>.*?</a>.*?(\d+)',
            resp.text, re.IGNORECASE | re.DOTALL
        )
        candidates = [(int(size), url) for url, size in links
                      if not any(x in url.lower() for x in ["ex-", "ex_", "xbrl"])]
        if candidates:
            candidates.sort(reverse=True)
            return "https://www.sec.gov" + candidates[0][1]
    except Exception:
        pass
    return None


def extract_text_from_doc(doc_url: str) -> str:
    """Download narrative document and extract clean plain text."""
    try:
        resp = requests.get(doc_url, headers=HEADERS, timeout=40)
        resp.raise_for_status()
        time.sleep(0.15)

        raw = resp.text

        # Remove scripts, styles, XBRL tags
        raw = re.sub(r"<(script|style|ix:[^>]+)[^>]*>.*?</\1>", " ", raw,
                     flags=re.DOTALL | re.IGNORECASE)
        # Strip remaining HTML tags
        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"&[a-z#0-9]+;", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Drop TOC region — it's always near the start and contains "Item X."
        # Find where the real narrative begins (after page 10 worth of TOC)
        # Heuristic: skip first 3000 chars if they look like a TOC
        toc_check = text[:3000]
        toc_hits  = len(re.findall(r"Item\s+\d+[A-Z]?\.", toc_check))
        if toc_hits > 6:
            # It's a TOC — find where content restarts after it
            narrative_start = re.search(
                r"(?i)(PART\s+I\b|ITEM\s+1\b.*?BUSINESS|MANAGEMENT.{0,20}DISCUSSION)",
                text[3000:])
            if narrative_start:
                text = text[3000 + narrative_start.start():]

        return text[:80_000]
    except Exception as e:
        logger.warning(f"  ⚠ Text extraction failed ({doc_url[-60:]}): {e}")
        return ""


def extract_key_sections(raw_text: str) -> dict:
    """Extract MDA, Risk Factors, Results of Operations for FinBERT."""
    sections = {}
    patterns = {
        "mda": (
            r"(?i)(?:item\s+7\.?\s*)?management.{0,15}discussion"
            r".{0,60}analysis\s*(.{1000,20000}?)(?=item\s+7a|item\s+8|$)"
        ),
        "risk_factors": (
            r"(?i)(?:item\s+1a\.?\s*)?risk\s+factors\s*(.{1000,15000}?)"
            r"(?=item\s+1b|item\s+2|unresolved|$)"
        ),
        "results_ops": (
            r"(?i)results\s+of\s+operations\s*(.{1000,12000}?)"
            r"(?=liquidity|capital\s+resources|item\s+\d|$)"
        ),
        "outlook": (
            r"(?i)(?:outlook|forward.looking\s+statements?|guidance)"
            r"\s*(.{300,6000}?)(?=\n\n\n|\Z)"
        ),
    }
    for name, pattern in patterns.items():
        match = re.search(pattern, raw_text, re.DOTALL)
        if match:
            text = match.group(1).strip()
            # Must be real text, not another TOC
            if len(text) >= 500 and len(re.findall(r"Item\s+\d", text)) < 5:
                sections[name] = text[:4000]

    if not sections:
        # Last resort: take a meaty middle chunk of the document
        mid = len(raw_text) // 4
        sections["full_text"] = raw_text[mid : mid + 4000]

    return sections


def download_edgar_for_ticker(ticker: str, start_date="2021-01-01", end_date=None) -> pd.DataFrame:
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    cik = TICKER_CIK.get(ticker)
    if not cik:
        return pd.DataFrame()

    all_records = []
    for form_type in FORM_TYPES:
        filings = get_filings_index(cik, form_type, start_date, end_date)
        logger.info(f"  {ticker} | {form_type}: {len(filings)} filings")

        for filing in filings[:12]:
            doc_url = get_narrative_doc_url(cik, filing["accession"])
            if not doc_url:
                logger.warning(f"    ⚠ No narrative doc found for {filing['accession']}")
                continue

            raw_text = extract_text_from_doc(doc_url)
            if not raw_text:
                continue

            sections = extract_key_sections(raw_text)
            for section_name, text in sections.items():
                all_records.append({
                    "ticker"     : ticker,
                    "form_type"  : form_type,
                    "filed"      : filing["filed"],
                    "section"    : section_name,
                    "text"       : text,
                    "text_length": len(text),
                    "fetched_at" : datetime.utcnow().isoformat(),
                })
            logger.info(f"    ✓ {filing['filed']} | {len(sections)} sections "
                        f"| lengths: {[len(v) for v in sections.values()]}")

    df = pd.DataFrame(all_records)
    logger.info(f"  ✓ {ticker}: {len(df)} sections total")
    return df


def run_edgar_download(tickers=None, start_date="2021-01-01", end_date=None) -> pd.DataFrame:
    if tickers is None:
        tickers = list(TICKER_CIK.keys())
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_dfs = []

    logger.info("=" * 60)
    logger.info("AlphaFlow — EDGAR Text Downloader")
    logger.info("=" * 60)
    logger.info(f"Tickers: {len(tickers)} | {start_date} → {end_date}")
    logger.info("")

    for ticker in tickers:
        logger.info(f"Processing {ticker}...")
        df = download_edgar_for_ticker(ticker, start_date, end_date)
        if not df.empty:
            df.to_parquet(OUTPUT_DIR / f"{ticker}_edgar_text.parquet", index=False)
            all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_parquet(OUTPUT_DIR / "all_tickers_edgar_text.parquet", index=False)
        logger.info("")
        logger.info("=" * 60)
        logger.info("EDGAR Download Complete")
        logger.info(f"  Total sections : {len(combined)}")
        logger.info(f"  Tickers        : {combined['ticker'].nunique()}")
        logger.info(f"  Avg text length: {int(combined['text_length'].mean())} chars")
        logger.info(f"  Form types     : {combined['form_type'].value_counts().to_dict()}")
        return combined

    logger.warning("No EDGAR data downloaded.")
    return pd.DataFrame()


if __name__ == "__main__":
    df = run_edgar_download(
        tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "JNJ", "XOM", "BRK-B", "TSLA"],
        start_date="2021-01-01",
    )
    if not df.empty:
        print(df[["ticker","form_type","filed","section","text_length"]].head(15).to_string())