"""
Assembles prompts for the DoE RAG assistant.

Two modes:
  build_rag_prompt   — system prompt with retrieved NIST context embedded
  build_direct_prompt — system prompt with no context; model answers from own knowledge
"""

# ---------------------------------------------------------------------------
# System prompt templates
# ---------------------------------------------------------------------------

_RAG_SYSTEM = """\
You are a Design of Experiments (DoE) methodology assistant grounded in the NIST Statistical Handbook.

Answer the user's question using ONLY the context provided below.
Always cite the specific NIST section (e.g. "NIST 3.4.2") and URL that supports your answer.
If the context does not contain enough information to answer, say so — do not fabricate.

CONTEXT:
{context}"""

_DIRECT_SYSTEM = """\
You are a Design of Experiments (DoE) methodology assistant grounded in the NIST Statistical Handbook.

Answer from your own knowledge. If you are uncertain, say so explicitly."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_rag_prompt(question: str, chunks: list[dict]) -> tuple[str, str]:
    """
    Build a RAG prompt from retrieved chunks.

    Returns (system_prompt, user_message). The system prompt embeds the
    retrieved context; the user message is the bare question.
    Each chunk is formatted as:
        [NIST <chunk_id>] <title>
        Source: <url>
        <text>
    """
    context = _format_chunks(chunks)
    system = _RAG_SYSTEM.format(context=context)
    return system, question


def build_direct_prompt(question: str) -> tuple[str, str]:
    """
    Build a direct (no-RAG) prompt.

    Returns (system_prompt, user_message). The model answers from its own
    knowledge without any retrieved context.
    """
    return _DIRECT_SYSTEM, question


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_chunks(chunks: list[dict]) -> str:
    parts = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        chunk_id = chunk.get("chunk_id") or meta.get("chunk_id", "?")
        title = meta.get("title", "")
        url = meta.get("url", "")
        text = chunk.get("text", "")
        parts.append(
            f"[NIST {chunk_id}] {title}\n"
            f"Source: {url}\n"
            f"{text}"
        )
    return "\n\n---\n\n".join(parts)
