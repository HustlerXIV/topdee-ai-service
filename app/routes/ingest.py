from fastapi import APIRouter, File, Form, Request, UploadFile
from pydantic import BaseModel

from ..ingest import ingest_file, ingest_text

router = APIRouter()


class IngestTextRequest(BaseModel):
    tenant_id: str
    knowledge_base_id: str
    source: str
    text: str


class IngestResponse(BaseModel):
    chunks: int
    stored: int
    source: str | None = None


@router.post("/ingest/text", response_model=IngestResponse)
def ingest_text_route(req: IngestTextRequest, request: Request) -> IngestResponse:
    result = ingest_text(
        store=request.app.state.vector_store,
        tenant_id=req.tenant_id,
        knowledge_base_id=req.knowledge_base_id,
        source=req.source,
        text=req.text,
    )
    return IngestResponse(**result)


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file_route(
    request: Request,
    tenant_id: str = Form(...),
    knowledge_base_id: str = Form(...),
    file: UploadFile = File(...),
) -> IngestResponse:
    content = await file.read()
    result = ingest_file(
        store=request.app.state.vector_store,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        filename=file.filename or "upload",
        content=content,
    )
    return IngestResponse(**result)


class DeleteKBRequest(BaseModel):
    tenant_id: str
    knowledge_base_id: str


@router.post("/ingest/kb/delete")
def delete_kb(req: DeleteKBRequest, request: Request) -> dict:
    request.app.state.vector_store.delete_knowledge_base(
        tenant_id=req.tenant_id,
        knowledge_base_id=req.knowledge_base_id,
    )
    return {"deleted": True}
