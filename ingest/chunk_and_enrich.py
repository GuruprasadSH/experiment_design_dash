#!/usr/bin/env python3
"""
Processes raw NIST chunks: splits oversized pages at heading boundaries,
assigns metadata (doe_phase, topic_tags, complexity), and optionally enriches
each chunk with a Claude-generated context sentence prefix.

Usage:
    python ingest/chunk_and_enrich.py [--input PATH] [--output PATH] [--enrich]
"""

import argparse
import json
import re
import time
from pathlib import Path

import tiktoken
from bs4 import BeautifulSoup, NavigableString
from dotenv import load_dotenv

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_TOKENS = 1000  # Hard ceiling — chunks above this are split
MIN_TOKENS = 100   # Floor — sub-chunks below this are dropped after splitting

ENRICHMENT_SLEEP = 0.5  # Seconds between Claude API calls to respect rate limits

# Keywords used to classify each chunk into a DoE workflow phase
_PHASE_KEYWORDS: dict[str, list[str]] = {
    "planning": [
        "design", "factor", "level", "randomiz", "replicate", "center point",
        "screening", "fractional", "full factorial", "blocking", "sample size",
        "run order", "confound", "alias", "resolution", "power", "treatment",
    ],
    "analysis": [
        "anova", "residual", "interaction", "p-value", "f-test", "regression",
        "coefficient", "normal probability", "r-squared", "significant",
        "main effect", "effect estimate", "fitted", "model", "lack of fit",
        "variance", "sum of squares",
    ],
    "execution": [
        "run order", "randomization order", "conducting", "execution",
        "measurement", "collecting data", "data collection", "observations",
    ],
}

# Predefined topic tag vocabulary matched against chunk title + text excerpt
_TOPIC_VOCAB: list[str] = [
    "factorial", "fractional-factorial", "screening", "ANOVA", "regression",
    "blocking", "randomization", "center-points", "response-surface",
    "confounding", "interaction", "residuals", "Yates", "Taguchi",
    "mixture-design", "latin-square", "Box-Behnken", "CCD",
]

_ENRICHMENT_PROMPT = (
    'Given this NIST handbook page titled "{title}" from section "{section}":\n\n'
    "PAGE CONTENT (excerpt):\n{text}\n\n"
    "Write 2-3 sentences specifically for search retrieval purposes that do ALL of the following:\n"
    "1. Name the core statistical concept(s) covered using exact terminology\n"
    "2. Describe in plain language what practical problem this solves — use words a "
    "non-statistician engineer would use, not textbook language\n"
    "3. List alternative phrasings or plain-English synonyms someone might search for "
    "when they need this concept but don't know its name\n\n"
    "Be specific. Do not write a generic summary. "
    "Answer with only the 2-3 sentences, nothing else."
)

_tokenizer = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base (required for text-embedding-3-small)."""
    return len(_tokenizer.encode(text))


# ---------------------------------------------------------------------------
# Metadata classifiers
# ---------------------------------------------------------------------------

def assign_doe_phase(title: str, text: str) -> str:
    """
    Classify a chunk into a DoE workflow phase via keyword scoring.

    Scores each phase by counting keyword matches in title + first 500 chars
    of text. The phase with the highest score wins. Defaults to 'planning'
    when no keywords match, since most NIST DoE pages cover design planning.
    """
    combined = (title + " " + text[:500]).lower()
    scores = {
        phase: sum(1 for kw in kws if kw.lower() in combined)
        for phase, kws in _PHASE_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "planning"


def assign_topic_tags(title: str, text: str) -> list[str]:
    """
    Match predefined DoE topic tags against the chunk title and text excerpt.
    Returns the list of matched tags (may be empty).
    """
    combined = (title + " " + text[:200]).lower()
    return [tag for tag in _TOPIC_VOCAB if tag.lower() in combined]


def assign_complexity(chunk_id: str) -> str:
    """
    Assign complexity level based on the section depth in the chunk ID.

    introductory: ≤ 2 dot-separated levels  (e.g. "3.1", "eda.1.1")
    intermediate:   3 levels                 (e.g. "3.4.1", "eda.1.2.3")
    advanced:     ≥ 4 levels                 (e.g. "3.4.1.2", "eda.1.2.3.4")

    Part prefixes (e.g. "eda.", "pmd.") and alpha split suffixes (e.g. "3.4.1.2a")
    are stripped before counting depth levels.
    """
    # Strip part prefix (e.g. "eda.1.1" → "1.1", "pmd.4.1" → "4.1")
    clean_id = re.sub(r"^(eda|pmd)\.", "", chunk_id)
    # Strip alpha suffix from heading-boundary splits (e.g. "3.4.1.2a" → "3.4.1.2")
    clean_id = re.sub(r"[a-z]+$", "", clean_id)
    depth = len(clean_id.split("."))
    if depth <= 2:
        return "introductory"
    elif depth == 3:
        return "intermediate"
    else:
        return "advanced"


# ---------------------------------------------------------------------------
# Splitting: heading-boundary and paragraph-boundary
# ---------------------------------------------------------------------------

def _walk_and_split(
    element,
    sections: list[tuple[str, list[str]]],
    current: list,
) -> None:
    """
    Recursive depth-first walk over a BeautifulSoup element tree.

    Accumulates text into `current` (a [title, lines] pair) and flushes
    it into `sections` whenever an h2 or h3 heading is encountered.
    This handles headings nested arbitrarily deep inside tables.

    Args:
        element:  Current node (tag or NavigableString)
        sections: Completed sections accumulated so far — mutated in place
        current:  [title_str, lines_list] — the in-progress section — mutated in place
    """
    if isinstance(element, NavigableString):
        # Bare text node — NavigableString has name=None, not name absent
        text = str(element).strip()
        if text:
            current[1].append(text)
        return

    if element.name in ("script", "style"):
        return

    if element.name in ("h2", "h3"):
        # Flush the current section, start a new one
        if current[1]:
            sections.append((current[0], list(current[1])))
        current[0] = element.get_text(strip=True)
        current[1].clear()
        return

    for child in element.children:
        _walk_and_split(child, sections, current)


def split_at_headings(raw_html: str, parent: dict) -> list[dict]:
    """
    Split an oversized chunk at h2/h3 heading boundaries.

    Walks the full HTML tree to find headings even when nested inside tables.
    Each heading starts a new sub-chunk; content before the first heading
    becomes a "preamble" sub-chunk using the parent title.

    Falls back to paragraph-based splitting when no headings are found.
    Returns the original chunk (unchanged) if splitting produces only one result.
    """
    soup = BeautifulSoup(raw_html, "html.parser")

    if not soup.find(["h2", "h3"]):
        return split_at_paragraphs(raw_html, parent)

    sections: list[tuple[str, list[str]]] = []
    current: list = [parent["title"], []]

    for child in soup.children:
        _walk_and_split(child, sections, current)

    # Flush the final in-progress section
    if current[1]:
        sections.append((current[0], list(current[1])))

    if len(sections) <= 1:
        return [parent]  # No meaningful split — return original

    # Convert (title, lines) pairs to text
    text_sections = [(title, "\n".join(lines)) for title, lines in sections]
    sub_chunks = _build_sub_chunks(text_sections, parent)

    return sub_chunks if len(sub_chunks) > 1 else [parent]


def split_at_paragraphs(raw_html: str, parent: dict) -> list[dict]:
    """
    Fallback splitter: groups paragraphs until approaching MAX_TOKENS,
    then starts a new sub-chunk. Used when no h2/h3 headings are found.

    Individual paragraphs that exceed MAX_TOKENS are themselves split
    at sentence boundaries via _split_long_text.
    """
    soup = BeautifulSoup(raw_html, "html.parser")
    paragraphs = [
        p.get_text(separator=" ", strip=True)
        for p in soup.find_all("p")
        if p.get_text(strip=True)
    ]

    if not paragraphs:
        return [parent]

    # Expand any single paragraph that exceeds MAX_TOKENS into sentences
    expanded: list[str] = []
    for para in paragraphs:
        if count_tokens(para) > MAX_TOKENS:
            expanded.extend(_split_long_text(para))
        else:
            expanded.append(para)

    sections: list[tuple[str, str]] = []
    buffer: list[str] = []
    buffer_tokens = 0

    for para in expanded:
        para_tokens = count_tokens(para)
        if buffer and buffer_tokens + para_tokens > MAX_TOKENS:
            sections.append((parent["title"], "\n".join(buffer)))
            buffer = [para]
            buffer_tokens = para_tokens
        else:
            buffer.append(para)
            buffer_tokens += para_tokens

    if buffer:
        sections.append((parent["title"], "\n".join(buffer)))

    return _build_sub_chunks(sections, parent) if sections else [parent]


def _split_long_text(text: str) -> list[str]:
    """
    Last-resort splitter: splits text at sentence boundaries (`. `) into
    pieces no larger than MAX_TOKENS. Used when a single paragraph exceeds
    MAX_TOKENS (e.g. text extracted from a large HTML table).

    Falls back to hard truncation at MAX_TOKENS if no sentence boundaries exist.
    """
    # Try splitting on '. ' (sentence boundary)
    sentences = re.split(r"(?<=\. )", text)
    if len(sentences) <= 1:
        # No sentence boundaries — hard-truncate by token using the encoder
        tokens = _tokenizer.encode(text)
        pieces = []
        for start in range(0, len(tokens), MAX_TOKENS):
            piece = _tokenizer.decode(tokens[start : start + MAX_TOKENS])
            if piece.strip():
                pieces.append(piece.strip())
        return pieces if pieces else [text[:500]]

    pieces: list[str] = []
    buffer_sentences: list[str] = []
    buffer_tokens = 0

    for sent in sentences:
        sent_tokens = count_tokens(sent)
        if buffer_sentences and buffer_tokens + sent_tokens > MAX_TOKENS:
            pieces.append("".join(buffer_sentences).strip())
            buffer_sentences = [sent]
            buffer_tokens = sent_tokens
        else:
            buffer_sentences.append(sent)
            buffer_tokens += sent_tokens

    if buffer_sentences:
        pieces.append("".join(buffer_sentences).strip())

    return [p for p in pieces if p]


def _build_sub_chunks(sections: list[tuple[str, str]], parent: dict) -> list[dict]:
    """
    Build sub-chunk dicts from a list of (title, text) pairs.

    Assigns alphabetic suffixes to chunk_ids (e.g. "3.4.1.2" → "3.4.1.2a", "3.4.1.2b").
    Drops sections below MIN_TOKENS. Sections that still exceed MAX_TOKENS after
    heading/paragraph splitting are further reduced via _split_long_text.
    """
    sub_chunks: list[dict] = []
    suffix_letters = "abcdefghijklmnopqrstuvwxyz"
    letter_idx = 0

    for title, text in sections:
        text = re.sub(r"[ \t]+", " ", text.strip())
        token_count = count_tokens(text)

        if token_count < MIN_TOKENS:
            continue

        # If still oversized, apply sentence-boundary / token-truncation splitting
        if token_count > MAX_TOKENS:
            pieces = _split_long_text(text)
        else:
            pieces = [text]

        for piece in pieces:
            piece_tokens = count_tokens(piece)
            if piece_tokens < MIN_TOKENS:
                continue
            suffix = suffix_letters[letter_idx] if letter_idx < len(suffix_letters) else str(letter_idx)
            sub_chunks.append({
                **parent,
                "chunk_id": f"{parent['chunk_id']}{suffix}",
                "title": title,
                "text": piece,
                "token_count": piece_tokens,
            })
            letter_idx += 1

    return sub_chunks if sub_chunks else [parent]


# ---------------------------------------------------------------------------
# Contextual enrichment
# ---------------------------------------------------------------------------

def enrich_chunk(chunk: dict, client) -> str:
    """
    Call Claude Haiku to generate 2-3 vocabulary-bridging context sentences for a chunk.
    The sentences are prepended to the chunk text before embedding, following
    Anthropic's Contextual Retrieval technique (September 2024).

    The prompt includes a 600-character excerpt of the page so the model can
    generate accurate synonyms without hallucinating content.

    Returns the context string.
    """
    # Include a truncated excerpt so the model sees the actual content,
    # not just title and section — required for accurate synonym generation.
    text_excerpt = chunk["text"][:600]
    prompt = _ENRICHMENT_PROMPT.format(
        title=chunk["title"],
        section=chunk["section"],
        text=text_excerpt,
    )
    message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=250,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


# ---------------------------------------------------------------------------
# Main processor
# ---------------------------------------------------------------------------

def process_chunks(
    raw_chunks: list[dict],
    enrich: bool = False,
) -> tuple[list[dict], int, int]:
    """
    Process all raw chunks through the full pipeline:

    1. Split chunks exceeding MAX_TOKENS at heading/paragraph boundaries
    2. Assign doe_phase, topic_tags, complexity metadata
    3. JSON-stringify linked_chunks and topic_tags (ChromaDB requires string values)
    4. Optionally prepend Claude-generated context sentences (--enrich flag)
    5. Strip raw_html from output (not needed downstream; keeps file size small)

    Returns:
        (processed_chunks, split_count, enriched_count)
    """
    client = None
    if enrich:
        from anthropic import Anthropic
        client = Anthropic()

    processed: list[dict] = []
    split_count = 0
    enriched_count = 0

    for i, raw in enumerate(raw_chunks, 1):
        progress = f"  [{i:4d}/{len(raw_chunks)}] {raw['chunk_id']}"

        # --- Split oversized chunks ---
        if raw["token_count"] > MAX_TOKENS and raw.get("raw_html"):
            sub_chunks = split_at_headings(raw["raw_html"], raw)
            if len(sub_chunks) > 1:
                split_count += 1
                print(f"{progress} → split into {len(sub_chunks)} sub-chunks")
            else:
                # Heading/paragraph split failed (e.g. pure data table with no <p>/<h2>)
                # Apply sentence-boundary / token-truncation as a final fallback
                pieces = _split_long_text(raw["text"])
                if len(pieces) > 1:
                    sub_chunks = list(_build_sub_chunks(
                        [(raw["title"], p) for p in pieces], raw
                    ))
                    split_count += 1
                    print(f"{progress} → split into {len(sub_chunks)} sub-chunks (sentence fallback)")
                else:
                    print(f"{progress} → {raw['token_count']} tokens (split failed, keeping as-is)")
        else:
            sub_chunks = [raw]

        # --- Assign metadata and optionally enrich ---
        for chunk in sub_chunks:
            chunk["doe_phase"] = assign_doe_phase(chunk["title"], chunk["text"])
            chunk["topic_tags"] = json.dumps(assign_topic_tags(chunk["title"], chunk["text"]))
            chunk["complexity"] = assign_complexity(chunk["chunk_id"])

            # linked_chunks is a Python list in raw JSON; must be a string for ChromaDB
            chunk["linked_chunks"] = json.dumps(chunk.get("linked_chunks", []))

            if enrich and client:
                try:
                    context = enrich_chunk(chunk, client)
                    chunk["text"] = context + "\n\n" + chunk["text"]
                    chunk["token_count"] = count_tokens(chunk["text"])
                    chunk["enriched"] = True
                    enriched_count += 1
                    time.sleep(ENRICHMENT_SLEEP)
                except Exception as exc:
                    print(f"\n  [WARN] Enrichment failed for {chunk['chunk_id']}: {exc}")

            chunk.pop("raw_html", None)  # Not needed downstream
            processed.append(chunk)

    return processed, split_count, enriched_count


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Process raw NIST chunks: split oversized pages, assign metadata "
            "(doe_phase, topic_tags, complexity), optionally enrich with Claude context."
        )
    )
    parser.add_argument(
        "--input",
        default="data/nist_chunks_raw.json",
        help="Input raw chunks JSON (default: data/nist_chunks_raw.json)",
    )
    parser.add_argument(
        "--output",
        default="data/nist_chunks_enriched.json",
        help="Output enriched chunks JSON (default: data/nist_chunks_enriched.json)",
    )
    parser.add_argument(
        "--enrich",
        action="store_true",
        help=(
            "Prepend Claude Haiku-generated context sentences to each chunk before embedding. "
            "Requires ANTHROPIC_API_KEY in environment. Increases chunk token count by ~50-100 tokens."
        ),
    )
    args = parser.parse_args()

    print(f"Loading raw chunks from {args.input}...")
    with open(args.input, encoding="utf-8") as f:
        raw_chunks = json.load(f)
    print(f"Loaded {len(raw_chunks)} raw chunks")

    print(f"\nProcessing (enrich={'yes' if args.enrich else 'no'})...")
    processed, split_count, enriched_count = process_chunks(raw_chunks, enrich=args.enrich)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    tokens = [c["token_count"] for c in processed]
    below_floor = sum(1 for t in tokens if t < MIN_TOKENS)
    above_ceiling = sum(1 for t in tokens if t > MAX_TOKENS)

    print(f"\n{'='*55}")
    print(f"Input chunks:          {len(raw_chunks)}")
    print(f"Chunks split:          {split_count}")
    print(f"Total output chunks:   {len(processed)}")
    print(f"Chunks enriched:       {enriched_count} / {len(processed)}")
    if tokens:
        mean = int(sum(tokens) / len(tokens))
        print(f"Token distribution:    min={min(tokens)}, mean={mean}, max={max(tokens)}")
    print(f"Below {MIN_TOKENS} tokens:        {below_floor}  (should be 0)")
    print(f"Above {MAX_TOKENS} tokens:       {above_ceiling}  (should be 0)")
    print(f"Output: {args.output}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
