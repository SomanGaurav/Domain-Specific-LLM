"""Download 10-K filings from SEC EDGAR to build the domain corpus.

SEC requires a descriptive User-Agent with contact info on all requests, and
asks clients to stay under 10 requests/second.

Usage:
    python -m finllm.data.download_corpus --tickers AAPL MSFT JPM GS --num-filings 3
"""

import argparse
import json
import re
import time
from pathlib import Path

import requests

USER_AGENT = "finllm research project somangaurav03@gmail.com"
HEADERS = {"User-Agent": USER_AGENT}
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}"
REQUEST_DELAY_S = 0.15


def get_cik_map() -> dict[str, int]:
    """Map ticker symbol -> CIK number for all EDGAR-registered companies."""
    resp = requests.get(TICKER_MAP_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return {entry["ticker"]: entry["cik_str"] for entry in resp.json().values()}


def list_10k_filings(cik: int, limit: int) -> list[dict]:
    """Return metadata for the most recent `limit` 10-K filings of a company."""
    resp = requests.get(SUBMISSIONS_URL.format(cik=cik), headers=HEADERS, timeout=30)
    resp.raise_for_status()
    recent = resp.json()["filings"]["recent"]

    filings = []
    for form, accession, doc, date in zip(
        recent["form"], recent["accessionNumber"], recent["primaryDocument"], recent["filingDate"]
    ):
        if form == "10-K":
            filings.append({"accession": accession.replace("-", ""), "doc": doc, "date": date})
        if len(filings) >= limit:
            break
    return filings


def strip_html(html: str) -> str:
    """Crude HTML -> text conversion adequate for corpus building."""
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;|&#160;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def download_filings(tickers: list[str], num_filings: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cik_map = get_cik_map()
    manifest = []

    for ticker in tickers:
        cik = cik_map.get(ticker.upper())
        if cik is None:
            print(f"[skip] unknown ticker: {ticker}")
            continue
        for filing in list_10k_filings(cik, num_filings):
            url = ARCHIVES_URL.format(cik=cik, accession=filing["accession"], doc=filing["doc"])
            resp = requests.get(url, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            text = strip_html(resp.text)

            out_path = out_dir / f"{ticker.upper()}_10K_{filing['date']}.txt"
            out_path.write_text(text)
            manifest.append({"ticker": ticker.upper(), "url": url, "date": filing["date"],
                             "file": out_path.name, "chars": len(text)})
            print(f"[ok] {out_path.name} ({len(text):,} chars)")
            time.sleep(REQUEST_DELAY_S)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Downloaded {len(manifest)} filings to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--num-filings", type=int, default=3, help="10-Ks per company")
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw/sec_filings"))
    args = parser.parse_args()
    download_filings(args.tickers, args.num_filings, args.out_dir)


if __name__ == "__main__":
    main()
