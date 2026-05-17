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


def _embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Embed chunks as plain text, matching how queries are embedded.

    Previously this prepended '[Document: filename]' to each chunk before
    embedding. That caused a vector-space mismatch: stored vectors included
    the filename prefix but query vectors did not, so even an exact FAQ
    question scored poorly against its own answer. Embedding both sides the
    same way (plain text) gives consistent cosine similarity and dramatically
    improves recall for FAQ-style knowledge bases.
    """
    return embed_texts(chunks)


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
    vectors = _embed_chunks(chunks)
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
    vectors = _embed_chunks(chunks)
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
