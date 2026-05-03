from fastapi import APIRouter, Request

from ..config import settings

router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict:
    qdrant_ok = True
    try:
        request.app.state.vector_store.client.get_collections()
    except Exception:
        qdrant_ok = False
    return {
        "status": "ok" if qdrant_ok else "degraded",
        "model": settings.llm_model,
        "deps": {"qdrant": qdrant_ok},
    }
