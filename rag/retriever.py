#!/usr/bin/env python3
"""
Hybrid retriever combining ChromaDB semantic search with BM25 keyword search,
fused via Reciprocal Rank Fusion (RRF). Three Sprint-6 robustness features:

  1. Calibrated query expansion — rewrites plain-English questions at moderate
     specificity (not over-specialised into adjacent subfields).
  2. Query classifier — detects when a query already uses statistical vocabulary
     and skips expansion, preventing terminology drift on precise queries.
  3. Retrieval confidence gate — computes a top-1/top-2 RRF score ratio and
     falls back to no-RAG when retrieval is uncertain.

Usage (as a library):
    from rag.retriever import Retriever
    r = Retriever(db_path="./chroma_db_full")
    out = r.retrieve_with_confidence("What do I do about factors I can't control?")
    if out["confident"]:
        print(out["results"])
    else:
        print(out["low_confidence_warning"])
"""

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

import chromadb
from openai import OpenAI
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLLECTION_NAME = "nist_doe"
DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# RRF constant — 60 is the standard value from the original paper
RRF_K = 60

# Retrieval sizes (fetch more candidates than final k to improve fusion quality)
SEMANTIC_CANDIDATES = 20
BM25_CANDIDATES = 20
FINAL_TOP_K = 5

# Neighbor expansion: include up to this many linked chunks from the top-1 result
MAX_NEIGHBORS = 2

# Confidence gate threshold: ratio of top-1 to top-2 RRF score.
# Values above this indicate a clearly dominant top result; below indicates
# the retriever is uncertain and the fallback to no-RAG should fire.
# Chosen empirically at 1.15; increase to be more permissive, decrease to gate more aggressively.
CONFIDENCE_THRESHOLD = 1.15

# ---------------------------------------------------------------------------
# Statistical vocabulary for query classifier (Task 2)
# ---------------------------------------------------------------------------

# Terms that indicate a query is already using DoE/statistical vocabulary.
# When any of these appear in the query, expansion is skipped — the query
# is already precise enough to match handbook terminology.
_STATISTICAL_TERMS = {
    "factorial", "fractional factorial", "anova", "blocking", "confounding",
    "aliasing", "resolution", "replication", "center point", "response surface",
    "regression", "interaction", "main effect", "doe", "design of experiments",
    "p-value", "f-test", "residual", "curvature", "randomization", "nuisance",
    "latin square", "plackett-burman", "box-behnken", "ccd", "yates",
    "bm25", "screening design", "split-plot", "hard-to-change",
}

# ---------------------------------------------------------------------------
# Query expansion prompt (calibrated, Task 1)
# ---------------------------------------------------------------------------

# Sprint 6 revision: targets *moderate* specificity (textbook index / handbook
# section heading level), explicitly forbids over-specialisation into adjacent
# subfields (metrology, analytical chemistry, ML), and instructs the model to
# return the question unchanged when it already uses precise terminology.
_QUERY_EXPANSION_PROMPT = (
    "You are a DoE and statistics terminology translator. "
    "A practitioner has asked a question using plain, non-technical language. "
    "Rephrase it using the standard statistical and experimental design terms that would appear "
    "in a textbook index or handbook section heading — specific enough to find the right concept, "
    "but do not add sub-speciality jargon from adjacent fields (metrology, analytical chemistry, "
    "machine learning, etc.) unless the original question explicitly refers to them.\n\n"
    "If the original question already uses precise statistical terminology, return it unchanged.\n\n"
    "Output only the rephrased question, nothing else.\n\n"
    "Question: {question}"
)


# ---------------------------------------------------------------------------
# Query classifier
# ---------------------------------------------------------------------------

def needs_expansion(query: str) -> bool:
    """
    Return True if the query uses plain English and would benefit from expansion.

    Checks for statistical/DoE vocabulary in the query; if any recognised term
    is present the query is already precise enough and expansion is skipped.
    This prevents terminology drift on well-formed technical queries.
    """
    query_lower = query.lower()
    return not any(term in query_lower for term in _STATISTICAL_TERMS)


# ---------------------------------------------------------------------------
# Retriever class
# ---------------------------------------------------------------------------

class Retriever:
    """
    Hybrid semantic + BM25 retriever over the NIST DoE ChromaDB collection.

    Sprint 6 additions:
      - needs_expansion() classifier gates query expansion
      - retrieve_with_confidence() adds a confidence score and fallback path
    """

    def __init__(self, db_path: str = DB_PATH) -> None:
        self._openai = OpenAI()
        self._anthropic = None  # lazy-initialised on first expansion call

        chroma = chromadb.PersistentClient(path=db_path)
        self._collection = chroma.get_collection(COLLECTION_NAME)

        # Load all documents for BM25 index
        all_data = self._collection.get(include=["documents", "metadatas"])
        self._ids: list[str] = all_data["ids"]
        self._documents: list[str] = all_data["documents"]
        self._metadatas: list[dict] = all_data["metadatas"]

        # Build BM25 index — prepend title to document text so title keywords
        # participate in keyword matching (many NIST chunks have sparse body text
        # but informative titles, e.g. "Adding centerpoints").
        bm25_texts = [
            (meta.get("title", "") + " " + doc).lower()
            for meta, doc in zip(self._metadatas, self._documents)
        ]
        tokenized = [t.split() for t in bm25_texts]
        self._bm25 = BM25Okapi(tokenized)

        # Fast lookup by chunk_id
        self._by_id: dict[str, int] = {cid: i for i, cid in enumerate(self._ids)}

    # -----------------------------------------------------------------------
    # Public API — primary entry point
    # -----------------------------------------------------------------------

    def retrieve_with_confidence(
        self,
        query: str,
        top_k: int = FINAL_TOP_K,
        expand_neighbors: bool = True,
        expand_query: bool = True,
    ) -> dict:
        """
        Retrieve with a confidence gate. Returns a dict:

            {
                "results":               list[dict],   # retrieved chunks (empty if not confident)
                "confidence":            float,         # top-1 / top-2 RRF score ratio
                "confident":             bool,          # whether ratio exceeds CONFIDENCE_THRESHOLD
                "expanded_query":        str | None,    # expanded query if used, else None
                "expansion_fired":       bool,          # whether expansion was applied
                "low_confidence_warning": str | None,   # human-readable warning if not confident
            }

        When confident=False, results is an empty list and the caller should
        fall back to no-RAG generation with the warning shown to the user.
        """
        # Step 1: decide whether to expand
        expansion_fired = False
        expanded: str | None = None

        if expand_query and needs_expansion(query):
            expanded = self._expand_query(query)
            search_query = expanded
            expansion_fired = True
        else:
            search_query = query

        # Step 2: hybrid retrieval
        semantic_ranked = self._semantic_search(search_query, n=SEMANTIC_CANDIDATES)
        bm25_ranked = self._bm25_search(search_query, n=BM25_CANDIDATES)
        fused = _reciprocal_rank_fusion(semantic_ranked, bm25_ranked, k=RRF_K)

        # Step 3: confidence gate
        confidence = _compute_confidence(fused)
        confident = confidence >= CONFIDENCE_THRESHOLD

        if not confident:
            return {
                "results": [],
                "confidence": confidence,
                "confident": False,
                "expanded_query": expanded,
                "expansion_fired": expansion_fired,
                "low_confidence_warning": (
                    "⚠ Retrieval confidence low — answering from model knowledge "
                    "(no NIST citation available for this query)"
                ),
            }

        # Step 4: trim and expand neighbors
        top_results = fused[:top_k]
        if expand_neighbors and top_results:
            top_results = self._expand_neighbors(top_results, top_k)

        # Annotate with query provenance
        for r in top_results:
            r["original_query"] = query
            if expanded is not None:
                r["expanded_query"] = expanded

        return {
            "results": top_results,
            "confidence": confidence,
            "confident": True,
            "expanded_query": expanded,
            "expansion_fired": expansion_fired,
            "low_confidence_warning": None,
        }

    def retrieve(
        self,
        query: str,
        top_k: int = FINAL_TOP_K,
        expand_neighbors: bool = True,
        expand_query: bool = True,
    ) -> list[dict]:
        """
        Simplified retrieve() — returns the chunk list directly (no confidence
        gate). Used by the evaluator and legacy callers.

        When expand_query=True, the classifier still runs — expansion only fires
        if needs_expansion(query) is True.
        """
        expansion_fired = False
        expanded: str | None = None

        if expand_query and needs_expansion(query):
            expanded = self._expand_query(query)
            search_query = expanded
            expansion_fired = True
        else:
            search_query = query

        semantic_ranked = self._semantic_search(search_query, n=SEMANTIC_CANDIDATES)
        bm25_ranked = self._bm25_search(search_query, n=BM25_CANDIDATES)
        fused = _reciprocal_rank_fusion(semantic_ranked, bm25_ranked, k=RRF_K)
        top_results = fused[:top_k]

        if expand_neighbors and top_results:
            top_results = self._expand_neighbors(top_results, top_k)

        if expanded is not None:
            for r in top_results:
                r["original_query"] = query
                r["expanded_query"] = expanded
        elif not expansion_fired:
            for r in top_results:
                r["expansion_skipped"] = True

        return top_results

    # -----------------------------------------------------------------------
    # Query expansion
    # -----------------------------------------------------------------------

    def _expand_query(self, question: str) -> str:
        """
        Rephrase a plain-English question at moderate specificity using the
        calibrated Sprint 6 prompt. Lazy-initialises the Anthropic client.
        """
        if self._anthropic is None:
            from anthropic import Anthropic
            self._anthropic = Anthropic()

        prompt = _QUERY_EXPANSION_PROMPT.format(question=question)
        response = self._anthropic.messages.create(
            model="claude-haiku-4-5",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    # -----------------------------------------------------------------------
    # Semantic search
    # -----------------------------------------------------------------------

    def _semantic_search(self, query: str, n: int) -> list[dict]:
        """Return n chunks by cosine similarity via ChromaDB."""
        response = self._openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[query],
            dimensions=EMBEDDING_DIM,
        )
        query_vector = response.data[0].embedding

        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=min(n, len(self._ids)),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for i, cid in enumerate(results["ids"][0]):
            chunks.append({
                "chunk_id": cid,
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1.0 - results["distances"][0][i],
            })
        return chunks

    # -----------------------------------------------------------------------
    # BM25 search
    # -----------------------------------------------------------------------

    def _bm25_search(self, query: str, n: int) -> list[dict]:
        """Return n chunks by BM25 score."""
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]

        chunks = []
        for idx in top_indices:
            if scores[idx] == 0:
                break
            chunks.append({
                "chunk_id": self._ids[idx],
                "text": self._documents[idx],
                "metadata": self._metadatas[idx],
                "score": scores[idx],
            })
        return chunks

    # -----------------------------------------------------------------------
    # Neighbor expansion
    # -----------------------------------------------------------------------

    def _expand_neighbors(self, results: list[dict], top_k: int) -> list[dict]:
        """
        Fetch linked_chunks from the top-1 result and add any not already
        present, up to MAX_NEIGHBORS additional chunks.
        """
        top_result = results[0]
        linked_raw = top_result["metadata"].get("linked_chunks", "[]")
        linked_ids: list[str] = (
            json.loads(linked_raw) if isinstance(linked_raw, str) else linked_raw
        )

        existing_ids = {r["chunk_id"] for r in results}
        added = 0

        for neighbor_id in linked_ids:
            if added >= MAX_NEIGHBORS:
                break
            if neighbor_id in existing_ids or neighbor_id not in self._by_id:
                continue
            idx = self._by_id[neighbor_id]
            results.append({
                "chunk_id": neighbor_id,
                "text": self._documents[idx],
                "metadata": self._metadatas[idx],
                "score": 0.0,
            })
            existing_ids.add(neighbor_id)
            added += 1

        return results


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def _compute_confidence(fused: list[dict]) -> float:
    """
    Compute retrieval confidence as the ratio of the top-1 to top-2 RRF score.
    A high ratio means one result is clearly dominant; near 1.0 means ambiguous.
    Returns a large sentinel value (10.0) when there is only one result.
    """
    if len(fused) < 2:
        return 10.0
    top = fused[0].get("rrf_score", 0.0)
    second = fused[1].get("rrf_score", 0.0)
    if second == 0:
        return 10.0
    return top / second


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _reciprocal_rank_fusion(
    ranked_a: list[dict],
    ranked_b: list[dict],
    k: int = RRF_K,
) -> list[dict]:
    """
    Merge two ranked lists via Reciprocal Rank Fusion.

    RRF score for item i at rank r: 1 / (k + r)
    Items present in only one list still receive their single-list RRF score.
    """
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for rank, chunk in enumerate(ranked_a, start=1):
        cid = chunk["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
        chunk_map[cid] = chunk

    for rank, chunk in enumerate(ranked_b, start=1):
        cid = chunk["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
        if cid not in chunk_map:
            chunk_map[cid] = chunk

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    result = []
    for cid, score in merged:
        entry = dict(chunk_map[cid])
        entry["rrf_score"] = score
        result.append(entry)
    return result


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is a fractional factorial design?"
    print(f"Query: {query}")
    fired = needs_expansion(query)
    print(f"Expansion needed: {fired}\n")

    r = Retriever(db_path="./chroma_db_full")
    out = r.retrieve_with_confidence(query)

    if not out["confident"]:
        print(out["low_confidence_warning"])
        print(f"Confidence ratio: {out['confidence']:.3f} (threshold: {CONFIDENCE_THRESHOLD})")
    else:
        if out["expanded_query"]:
            print(f"Expanded: {out['expanded_query']}\n")
        for i, chunk in enumerate(out["results"], 1):
            meta = chunk["metadata"]
            print(f"[{i}] {chunk['chunk_id']} — {meta.get('title', '')[:60]}")
            print(f"    RRF: {chunk.get('rrf_score', 0):.4f} | confidence: {out['confidence']:.3f}")
        print()
