from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from ..llm import generate
from ..rag import format_context, retrieve

router = APIRouter()


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    tenant_id: str
    conversation_id: str
    agent_id: str
    system_prompt: str = ""
    model: str | None = None
    temperature: float = 0.3
    history: list[ChatMessage] = Field(default_factory=list)
    message: str
    knowledge_base_ids: list[str] = Field(default_factory=list)
    top_k: int = 8
    # When False, the prompt forbids the model from naming source files and
    # we strip filenames from the RAG context before showing them to the
    # model. We also still drop the `sources` list from the response so a
    # buggy frontend can't accidentally render it.
    mention_sources: bool = True


class ChatResponse(BaseModel):
    reply: str
    sources: list[str] = Field(default_factory=list)
    tokens_used: int = 0


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request) -> ChatResponse:
    store = request.app.state.vector_store
    hits = retrieve(
        store=store,
        tenant_id=req.tenant_id,
        query=req.message,
        knowledge_base_ids=req.knowledge_base_ids,
        top_k=req.top_k,
    )
    context = format_context(hits, with_sources=req.mention_sources)
    reply, tokens = generate(
        system_prompt=req.system_prompt,
        history=[m.model_dump() for m in req.history],
        user_message=req.message,
        rag_context=context or None,
        model=req.model,
        temperature=req.temperature,
        mention_sources=req.mention_sources,
    )
    # When the caller asked us to hide sources, return an empty list — saves
    # downstream code from having to remember not to render it.
    sources = (
        sorted({h.source for h in hits if h.source}) if req.mention_sources else []
    )
    return ChatResponse(reply=reply, sources=sources, tokens_used=tokens)
