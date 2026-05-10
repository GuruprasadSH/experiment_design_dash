#!/usr/bin/env python3
"""
Embeds enriched NIST chunks with text-embedding-3-small and stores them
in a persistent ChromaDB collection named 'nist_doe'.

Usage:
    python ingest/embed_and_store.py [--input PATH] [--db PATH] [--batch-size N]
"""

import argparse
import json
import time
from pathlib import Path

import chromadb
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLLECTION_NAME = "nist_doe"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
COST_PER_1K_TOKENS = 0.00002  # USD as of 2024
BATCH_SIZE = 100               # OpenAI recommends <=2048 inputs per call; 100 is safe
REQUEST_DELAY = 0.1            # Seconds between batches to respect rate limits

# Chunk IDs whose prefix should be excluded from the collection.
# 5.9.4* — the "Interaction effects matrix plot" page splits into 68 near-duplicate
# sub-chunks that saturate semantic search for any interaction/factor/confounding
# query, displacing the specific conceptual chunks that should rank. Excluded here;
# the enriched JSON retains them for reproducibility.
EXCLUDED_PREFIXES: tuple[str, ...] = ("5.9.4",)

_tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text))


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_batch(texts: list[str], client: OpenAI) -> list[list[float]]:
    """Embed a batch of texts using text-embedding-3-small. Returns list of vectors."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        dimensions=EMBEDDING_DIM,
    )
    return [item.embedding for item in response.data]


# ---------------------------------------------------------------------------
# ChromaDB storage
# ---------------------------------------------------------------------------

def build_collection(
    chunks: list[dict],
    db_path: str,
    batch_size: int,
    openai_client: OpenAI,
) -> tuple[int, int, float]:
    """
    Embed all chunks and upsert into a ChromaDB collection.

    Returns (total_chunks, total_tokens, estimated_cost_usd).
    """
    chroma = chromadb.PersistentClient(path=db_path)

    # Delete existing collection so re-runs are idempotent
    try:
        chroma.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing '{COLLECTION_NAME}' collection.")
    except Exception:
        pass

    collection = chroma.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total_tokens = 0
    embedded_count = 0

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks)...", end=" ", flush=True)

        # Prepend title so title keywords participate in semantic similarity.
        # Many NIST chunks have informative titles but sparse body text.
        texts = [c["title"] + "\n\n" + c["text"] for c in batch]
        batch_tokens = sum(count_tokens(t) for t in texts)
        total_tokens += batch_tokens

        vectors = embed_batch(texts, openai_client)

        ids = [c["chunk_id"] for c in batch]
        metadatas = [
            {
                "chunk_id": c["chunk_id"],
                "url": c["url"],
                "title": c["title"],
                "section": c["section"],
                "chapter": c.get("chapter", ""),
                "token_count": c["token_count"],
                "linked_chunks": c.get("linked_chunks", "[]"),
                "topic_tags": c.get("topic_tags", "[]"),
                "doe_phase": c.get("doe_phase", ""),
                "complexity": c.get("complexity", ""),
                "enriched": c.get("enriched", False),
            }
            for c in batch
        ]

        collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )

        embedded_count += len(batch)
        print(f"done ({batch_tokens} tokens)", flush=True)

        if batch_start + batch_size < len(chunks):
            time.sleep(REQUEST_DELAY)

    cost = (total_tokens / 1000) * COST_PER_1K_TOKENS
    return embedded_count, total_tokens, cost


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed NIST chunks with text-embedding-3-small and store in ChromaDB."
    )
    parser.add_argument(
        "--input",
        default="data/nist_chunks_enriched.json",
        help="Enriched chunks JSON (default: data/nist_chunks_enriched.json)",
    )
    parser.add_argument(
        "--db",
        default="./chroma_db",
        help="ChromaDB persistent storage path (default: ./chroma_db)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Chunks per embedding API call (default: {BATCH_SIZE})",
    )
    args = parser.parse_args()

    print(f"Loading chunks from {args.input}...")
    with open(args.input, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")

    # Apply exclusion filter
    before = len(chunks)
    chunks = [c for c in chunks if not c["chunk_id"].startswith(EXCLUDED_PREFIXES)]
    excluded = before - len(chunks)
    if excluded:
        print(f"Excluded {excluded} chunks matching prefixes {EXCLUDED_PREFIXES}")

    # Pre-flight token check — count the full embedded string (title + "\n\n" + text)
    tokens_preview = [count_tokens(c["title"] + "\n\n" + c["text"]) for c in chunks]
    over_limit = [c["chunk_id"] for c, t in zip(chunks, tokens_preview) if t > 8191]
    if over_limit:
        print(f"[WARN] {len(over_limit)} chunks exceed 8191 token embedding limit: {over_limit[:5]}")

    estimated_tokens = sum(tokens_preview)
    estimated_cost = (estimated_tokens / 1000) * COST_PER_1K_TOKENS
    print(f"Estimated tokens to embed: {estimated_tokens:,}")
    print(f"Estimated cost:            ${estimated_cost:.4f}")
    print()

    Path(args.db).mkdir(parents=True, exist_ok=True)
    client = OpenAI()

    print(f"Embedding and storing into ChromaDB at '{args.db}'...")
    embedded, total_tokens, cost = build_collection(chunks, args.db, args.batch_size, client)

    print(f"\n{'='*55}")
    print(f"Chunks embedded:    {embedded}")
    print(f"Total tokens:       {total_tokens:,}")
    print(f"Actual cost:        ${cost:.4f}")
    print(f"Collection:         {COLLECTION_NAME}")
    print(f"DB path:            {args.db}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
