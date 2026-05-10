#!/usr/bin/env python3
"""
Scrapes the full NIST/SEMATECH e-Handbook of Statistical Methods (all 8 parts).

Parts covered:
  1. EDA  — Exploratory Data Analysis
  2. MPC  — Measurement Process Characterization
  3. PPC  — Production Process Characterization
  4. PMD  — Process Modeling
  5. PRI  — Process Improvement (DoE)
  6. PMC  — Process or Product Monitoring and Control
  7. PRC  — Product and Process Comparisons
  8. APR  — Assessing Product Reliability

Produces data/nist_chunks_raw.json with one chunk per handbook page
(pages below 100 tokens are filtered as navigation-only).

Design: Single-pass BFS crawl — each page is fetched exactly once.
HTML is parsed immediately during the crawl and cached in memory.
This avoids the double-fetch pattern of separate discovery + scraping phases.

Usage:
    python ingest/scrape_nist.py [--output PATH]
"""

import argparse
import json
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

import requests
import tiktoken
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENTRY_POINTS = [
    "https://www.itl.nist.gov/div898/handbook/eda/eda.htm",  # Part 1: EDA
    "https://www.itl.nist.gov/div898/handbook/mpc/mpc.htm",  # Part 2: Measurement Process Characterization
    "https://www.itl.nist.gov/div898/handbook/ppc/ppc.htm",  # Part 3: Production Process Characterization
    "https://www.itl.nist.gov/div898/handbook/pmd/pmd.htm",  # Part 4: Process Modeling
    "https://www.itl.nist.gov/div898/handbook/pri/pri.htm",  # Part 5: Process Improvement (DoE)
    "https://www.itl.nist.gov/div898/handbook/pmc/pmc.htm",  # Part 6: Process/Product Monitoring and Control
    "https://www.itl.nist.gov/div898/handbook/prc/prc.htm",  # Part 7: Product and Process Comparisons
    "https://www.itl.nist.gov/div898/handbook/apr/apr.htm",  # Part 8: Assessing Product Reliability
]

SECTION_MAP = {
    "eda": "Exploratory Data Analysis",
    "mpc": "Measurement Process Characterization",
    "ppc": "Production Process Characterization",
    "pmd": "Process Modeling",
    "pri": "Process Improvement",
    "pmc": "Process or Product Monitoring and Control",
    "prc": "Product and Process Comparisons",
    "apr": "Assessing Product Reliability",
}

MIN_TOKENS = 100   # Pages below this are navigation-only — filtered out
MAX_TOKENS = 1000  # Pages above this are flagged for splitting in chunk_and_enrich.py

REQUEST_TIMEOUT = 15  # seconds per request
REQUEST_DELAY = 0.5   # seconds between requests (polite crawling)
MAX_RETRIES = 3

HEADERS = {"User-Agent": "NIST-DoE-RAG-research-bot/1.0 (educational; building open RAG system)"}

_tokenizer = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base (required for text-embedding-3-small)."""
    return len(_tokenizer.encode(text))


# ---------------------------------------------------------------------------
# URL utilities
# ---------------------------------------------------------------------------

def parse_chunk_id(url: str) -> str | None:
    """
    Parse a NIST handbook URL into a dot-separated section ID.

    PRI (Part 5, the DoE section) chunk IDs use bare section numbers so that
    existing eval-set golden IDs (e.g. "3.4.1.2") remain valid:
      .../pri/section3/pri3412.htm  →  "3.4.1.2"

    All other parts are prefixed to prevent collisions (every part numbers its
    chapters internally starting from 1):
      .../eda/section1/eda11.htm    →  "eda.1.1"
      .../mpc/section2/mpc21.htm    →  "mpc.2.1"
      .../ppc/section1/ppc11.htm    →  "ppc.1.1"
      .../pmd/section4/pmd41.htm    →  "pmd.4.1"
      .../pmc/section1/pmc11.htm    →  "pmc.1.1"
      .../prc/section1/prc11.htm    →  "prc.1.1"
      .../apr/section1/apr11.htm    →  "apr.1.1"

    Returns None for index/root pages (e.g. pri.htm) that have no numeric suffix.
    """
    filename = Path(urlparse(url).path).stem
    match = re.match(r"^(eda|mpc|ppc|pri|pmd|pmc|prc|apr)(\d+)$", filename)
    if not match:
        return None
    prefix = match.group(1)
    digits = ".".join(match.group(2))
    # PRI keeps bare IDs; all other parts are prefixed to avoid collisions
    if prefix == "pri":
        return digits
    return f"{prefix}.{digits}"


def get_section_name(url: str) -> str:
    """Return the Part name for this URL based on its path prefix."""
    for prefix, name in SECTION_MAP.items():
        if f"/{prefix}/" in url:
            return name
    return "Unknown"


def normalize_url(url: str) -> str:
    """Strip URL fragment (#anchor) to prevent the same page being fetched multiple times."""
    parsed = urlparse(url)
    return urlunparse(parsed._replace(fragment=""))


def is_enqueueable_url(url: str) -> bool:
    """
    Return True if this URL should be enqueued for crawling.

    We restrict to URLs whose filename is either:
    - A part-level index page (e.g. eda.htm, pri.htm, pmd.htm) — needed for link discovery
    - A standard content page matching {prefix}{digits}.htm (e.g. pri3412.htm, eda11.htm)

    This prevents queue explosion from EDA section 3's hundreds of individual technique
    pages (runseqpl.htm, histogra.htm, etc.) which follow non-standard naming conventions
    and cannot be parsed into valid chunk_ids anyway.
    """
    parsed = urlparse(url)
    if (
        "itl.nist.gov" not in parsed.netloc
        or "/div898/handbook/" not in parsed.path
        or not parsed.path.endswith(".htm")
        or not any(f"/{p}/" in parsed.path for p in SECTION_MAP)
    ):
        return False
    filename = Path(parsed.path).stem
    # Part-level index pages (entry points for link discovery)
    if re.match(r"^(eda|mpc|ppc|pri|pmd|pmc|prc|apr)$", filename):
        return True
    # Standard content pages with parseable chunk_ids
    if re.match(r"^(eda|mpc|ppc|pri|pmd|pmc|prc|apr)\d+$", filename):
        return True
    return False


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def fetch_page(url: str) -> requests.Response | None:
    """
    Fetch a URL with exponential-backoff retries.
    Returns the Response on success, None after MAX_RETRIES consecutive failures.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"\n  [WARN] Failed after {MAX_RETRIES} attempts: {url} — {exc}", flush=True)
    return None


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------

def extract_main_content(soup: BeautifulSoup) -> BeautifulSoup:
    """
    Return the main content element from a NIST handbook page.

    NIST pages use a table-based layout: narrow left sidebar + wide content column.
    We identify the content column as the <td> with the most text, which reliably
    selects the content area over navigation sidebars and header/footer rows.
    """
    for tag in soup.find_all(["script", "style"]):
        tag.decompose()

    tds = soup.find_all("td")
    if not tds:
        return soup.find("body") or soup

    return max(tds, key=lambda td: len(td.get_text(strip=True)))


def extract_title(soup: BeautifulSoup, content: BeautifulSoup) -> str:
    """
    Extract the page title.

    Priority: <h2> in content area (NIST's primary content heading) → <h1> →
    <title> tag (with NIST boilerplate stripped).
    """
    for tag_name in ("h2", "h1"):
        tag = content.find(tag_name)
        if tag:
            text = tag.get_text(strip=True)
            if len(text) > 3:
                return text

    title_tag = soup.find("title")
    if title_tag:
        text = title_tag.get_text(strip=True)
        text = re.sub(r"\s*-?\s*NIST.*$", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"^\d[\d.]*\.\s*", "", text).strip()
        if text:
            return text

    return "Untitled"


def extract_chapter(content: BeautifulSoup) -> str:
    """
    Extract the chapter name from the first <h3> in the content area,
    which NIST uses for chapter-level headings within the content column.
    Falls back to the first substantial anchor text (breadcrumb link).
    """
    h3 = content.find("h3")
    if h3:
        return h3.get_text(strip=True)

    for anchor in content.find_all("a", limit=10):
        text = anchor.get_text(strip=True)
        if 5 < len(text) < 80:
            return text

    return ""


def extract_linked_chunk_ids(content: BeautifulSoup, page_url: str) -> list[str]:
    """
    Find all internal NIST handbook links within the content area and return
    their chunk IDs. These populate linked_chunks for neighbor expansion at
    retrieval time (Sprint 2).
    """
    seen: set[str] = set()
    chunk_ids: list[str] = []

    for anchor in content.find_all("a", href=True):
        absolute = normalize_url(urljoin(page_url, anchor["href"]))
        if is_enqueueable_url(absolute) and absolute != normalize_url(page_url):
            cid = parse_chunk_id(absolute)
            if cid and cid not in seen:
                chunk_ids.append(cid)
                seen.add(cid)

    return chunk_ids


def extract_text(content: BeautifulSoup) -> str:
    """
    Extract clean plain text from the content element.
    Preserves paragraph breaks via newlines, collapses whitespace, removes blank lines.
    """
    raw = content.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in raw.split("\n") if line.strip()]
    text = "\n".join(lines)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Single-pass BFS crawl (discovery + scraping combined)
# ---------------------------------------------------------------------------

def crawl_and_scrape(
    entry_points: list[str],
    output_path: str,
) -> list[dict]:
    """
    Single-pass BFS crawl: fetch each page exactly once, parse it immediately,
    follow its links to enqueue new pages, and build the chunk if it passes filters.

    This avoids the double-fetch pattern of separate discovery + scraping phases.
    Saves chunks to output_path as they accumulate, writing the full JSON at the end.

    Returns the list of saved chunks.
    """
    visited: set[str] = set(normalize_url(u) for u in entry_points)
    queue: deque[str] = deque(normalize_url(u) for u in entry_points)
    chunks: list[dict] = []
    filtered_count = 0
    crawled_count = 0

    print("Starting single-pass BFS crawl (discovery + scraping combined)...", flush=True)
    print(f"Entry points: {len(entry_points)} parts (eda, mpc, ppc, pmd, pri, pmc, prc, apr)", flush=True)
    print("Note: restricting to standard-format pages ({prefix}{digits}.htm) to avoid technique-page explosion.", flush=True)

    while queue:
        url = queue.popleft()
        crawled_count += 1

        response = fetch_page(url)
        time.sleep(REQUEST_DELAY)

        if response is None:
            filtered_count += 1
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        # Enqueue new URLs found on this page — normalized to strip fragments, filtered to
        # standard-format pages only (prevents queue explosion from EDA technique pages)
        for anchor in soup.find_all("a", href=True):
            absolute = normalize_url(urljoin(url, anchor["href"]))
            if is_enqueueable_url(absolute) and absolute not in visited:
                visited.add(absolute)
                queue.append(absolute)

        # Attempt to build a chunk from this page
        chunk_id = parse_chunk_id(url)
        if chunk_id is None:
            # Index/root page (e.g. pri.htm) — used for discovery only, not a content chunk
            print(f"  [disc] {url} (index, not a content chunk, queue={len(queue)})", flush=True)
            continue

        content = extract_main_content(soup)
        text = extract_text(content)
        token_count = count_tokens(text)

        if token_count < MIN_TOKENS:
            filtered_count += 1
            print(f"  [skip] {chunk_id} — {token_count} tokens (below floor)", flush=True)
            continue

        chunk = {
            "chunk_id": chunk_id,
            "url": url,
            "title": extract_title(soup, content),
            "section": get_section_name(url),
            "chapter": extract_chapter(content),
            "text": text,
            "token_count": token_count,
            "linked_chunks": extract_linked_chunk_ids(content, url),
            "raw_html": str(content),
            "enriched": False,
        }
        chunks.append(chunk)

        oversized_marker = " ⚠ >1000 tokens" if token_count > MAX_TOKENS else ""
        print(
            f"  [save] {chunk_id:12s} {token_count:5d} tok  {chunk['title'][:50]}{oversized_marker}",
            flush=True,
        )

    # Write output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    # Summary
    tokens = [c["token_count"] for c in chunks]
    oversized = sum(1 for t in tokens if t > MAX_TOKENS)

    print(f"\n{'='*55}", flush=True)
    print(f"Pages crawled:                         {crawled_count}", flush=True)
    print(f"Pages filtered (<{MIN_TOKENS} tok or no chunk_id): {filtered_count}", flush=True)
    print(f"Chunks saved:                          {len(chunks)}", flush=True)
    if tokens:
        mean = int(sum(tokens) / len(tokens))
        print(f"Token distribution:                    min={min(tokens)}, mean={mean}, max={max(tokens)}", flush=True)
    print(f"Chunks exceeding {MAX_TOKENS} tokens (need split): {oversized}", flush=True)
    print(f"Output: {output_path}", flush=True)
    print(f"{'='*55}", flush=True)

    return chunks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape full NIST handbook (all 8 parts) and produce raw chunks JSON."
    )
    parser.add_argument(
        "--output",
        default="data/nist_chunks_raw.json",
        help="Output path for raw chunks JSON (default: data/nist_chunks_raw.json)",
    )
    args = parser.parse_args()
    crawl_and_scrape(entry_points=ENTRY_POINTS, output_path=args.output)
