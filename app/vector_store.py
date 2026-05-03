"""Qdrant client wrapper. Single collection, multi-tenant via payload filters."""
import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .config import settings


@dataclass
class Chunk:
    text: str
    tenant_id: str
    knowledge_base_id: str
    source: str  # e.g. filename
    chunk_index: int


@dataclass
class Hit:
    text: str
    score: float
    source: str
    knowledge_base_id: str


class VectorStore:
    def __init__(self) -> None:
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=30,
        )
        self.collection = settings.qdrant_collection

    def ensure_collection(self) -> None:
        try:
            self.client.get_collection(self.collection)
            return
        except Exception:
            pass
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(
                size=settings.embedding_dim,
                distance=qm.Distance.COSINE,
            ),
        )
        # payload indexes for fast filtering
        for field in ("tenant_id", "knowledge_base_id"):
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field,
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )

    def upsert(self, vectors: list[list[float]], chunks: list[Chunk]) -> int:
        assert len(vectors) == len(chunks), "vector/chunk count mismatch"
        points = [
            qm.PointStruct(
                id=str(uuid.uuid4()),
                vector=v,
                payload={
                    "text": c.text,
                    "tenant_id": c.tenant_id,
                    "knowledge_base_id": c.knowledge_base_id,
                    "source": c.source,
                    "chunk_index": c.chunk_index,
                },
            )
            for v, c in zip(vectors, chunks)
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def search(
        self,
        query_vector: list[float],
        tenant_id: str,
        knowledge_base_ids: list[str] | None = None,
        top_k: int = 4,
    ) -> list[Hit]:
        must = [
            qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id)),
        ]
        if knowledge_base_ids:
            must.append(
                qm.FieldCondition(
                    key="knowledge_base_id",
                    match=qm.MatchAny(any=knowledge_base_ids),
                )
            )
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            query_filter=qm.Filter(must=must),
            limit=top_k,
            with_payload=True,
        )
        return [
            Hit(
                text=r.payload.get("text", ""),
                score=r.score,
                source=r.payload.get("source", ""),
                knowledge_base_id=r.payload.get("knowledge_base_id", ""),
            )
            for r in results
        ]

    def delete_knowledge_base(self, tenant_id: str, knowledge_base_id: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id)),
                        qm.FieldCondition(
                            key="knowledge_base_id",
                            match=qm.MatchValue(value=knowledge_base_id),
                        ),
                    ]
                )
            ),
        )
