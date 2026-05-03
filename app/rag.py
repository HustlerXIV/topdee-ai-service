"""RAG retrieval: embed query, search Qdrant, format context block."""
from .embeddings import embed_query
from .vector_store import Hit, VectorStore


def retrieve(
    store: VectorStore,
    tenant_id: str,
    query: str,
    knowledge_base_ids: list[str] | None,
    top_k: int = 8,
) -> list[Hit]:
    if not knowledge_base_ids:
        # No KBs attached → skip RAG entirely.
        return []
    qvec = embed_query(query)
    return store.search(
        query_vector=qvec,
        tenant_id=tenant_id,
        knowledge_base_ids=knowledge_base_ids,
        top_k=top_k,
    )


def format_context(hits: list[Hit], with_sources: bool = True) -> str:
    """Render hits as a numbered block.

    When `with_sources=False`, the source filename is omitted entirely so the
    LLM has no way to leak it back to the customer. We use this on every
    real channel (LINE, Facebook, etc.) — only the dashboard playground gets
    sources, since that's where staff are testing the bot's grounding.
    """
    if not hits:
        return ""
    lines = []
    for i, h in enumerate(hits, 1):
        if with_sources and h.source:
            lines.append(f"[{i}] (source: {h.source})\n{h.text}")
        else:
            lines.append(f"[{i}]\n{h.text}")
    return "\n\n".join(lines)
