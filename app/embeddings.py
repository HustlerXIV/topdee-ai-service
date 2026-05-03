"""Embedding model wrapper. Uses sentence-transformers locally so we don't
require an extra API key for embeddings. Swap to VoyageAI / OpenAI later by
changing this module + EMBEDDING_DIM in config.
"""
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from .config import settings


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vectors = _model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
