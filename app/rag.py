"""RAG retrieval: embed query, search Qdrant, format context block."""
import logging

from .embeddings import embed_query
from .vector_store import Hit, VectorStore

logger = logging.getLogger(__name__)


def retrieve(
    store: VectorStore,
    tenant_id: str,
    query: str,
    knowledge_base_ids: list[str] | None,
    top_k: int = 8,
) -> list[Hit]:
    if not knowledge_base_ids:
        # No KBs attached → skip RAG entirely.
        logger.warning("[RAG] No knowledge_base_ids provided — skipping retrieval.")
        return []
    qvec = embed_query(query)
    hits = store.search(
        query_vector=qvec,
        tenant_id=tenant_id,
        knowledge_base_ids=knowledge_base_ids,
        top_k=top_k,
    )
    if hits:
        logger.info(
            "[RAG] Retrieved %d hit(s) for tenant=%s kbs=%s | top score=%.3f | query=%r",
            len(hits), tenant_id, knowledge_base_ids, hits[0].score, query[:80],
        )
    else:
        logger.warning(
            "[RAG] No hits for tenant=%s kbs=%s | query=%r — KB may be empty or not yet ingested.",
            tenant_id, knowledge_base_ids, query[:80],
        )
    return hits


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
