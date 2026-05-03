"""Ingestion pipeline: file/text -> chunks -> embeddings -> Qdrant."""
import io

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .config import settings
from .embeddings import embed_texts
from .vector_store import Chunk, VectorStore


def extract_text(filename: str, content: bytes) -> str:
    """Extract plain text from a file by extension. Add docx/csv handlers as needed."""
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    # treat anything else as text/markdown/csv
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("latin-1", errors="ignore")


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [c for c in splitter.split_text(text) if c.strip()]


def _embed_with_source_context(chunks: list[str], source: str) -> list[list[float]]:
    """Embed chunks with the source/filename prepended so questions about the
    document title (e.g. "what's the project name?") can match by similarity.
    The prepended context is only used for the embedding — the chunk stored in
    the payload remains the original clean text."""
    contextualized = [f"[Document: {source}]\n{c}" for c in chunks]
    return embed_texts(contextualized)


def ingest_file(
    store: VectorStore,
    tenant_id: str,
    knowledge_base_id: str,
    filename: str,
    content: bytes,
) -> dict:
    text = extract_text(filename, content)
    chunks = chunk_text(text)
    if not chunks:
        return {"chunks": 0, "stored": 0}
    vectors = _embed_with_source_context(chunks, filename)
    payload_chunks = [
        Chunk(
            text=t,
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            source=filename,
            chunk_index=i,
        )
        for i, t in enumerate(chunks)
    ]
    stored = store.upsert(vectors, payload_chunks)
    return {"chunks": len(chunks), "stored": stored, "source": filename}


def ingest_text(
    store: VectorStore,
    tenant_id: str,
    knowledge_base_id: str,
    source: str,
    text: str,
) -> dict:
    chunks = chunk_text(text)
    if not chunks:
        return {"chunks": 0, "stored": 0}
    vectors = _embed_with_source_context(chunks, source)
    payload_chunks = [
        Chunk(
            text=t,
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            source=source,
            chunk_index=i,
        )
        for i, t in enumerate(chunks)
    ]
    stored = store.upsert(vectors, payload_chunks)
    return {"chunks": len(chunks), "stored": stored, "source": source}
