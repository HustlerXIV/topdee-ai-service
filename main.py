"""FastAPI entrypoint for the AI service."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.vector_store import VectorStore
from app.routes import health, chat, ingest


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the vector store + embedding model so the first request isn't slow.
    app.state.vector_store = VectorStore()
    app.state.vector_store.ensure_collection()
    yield


app = FastAPI(
    title="Topdee AI Service",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(ingest.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=settings.ai_service_port, reload=True)
