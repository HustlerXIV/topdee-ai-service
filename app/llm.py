"""LLM adapter — wraps LangChain's ChatGoogleGenerativeAI so the rest of
the app calls a small interface and can swap providers later.
"""
from typing import Iterable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from .config import settings


def _build_llm(model: str | None, temperature: float) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model or settings.llm_model,
        temperature=temperature,
        google_api_key=settings.google_api_key,
        max_output_tokens=1024,
    )


def to_lc_messages(
    system_prompt: str,
    history: Iterable[dict],
    user_message: str,
    rag_context: str | None = None,
    mention_sources: bool = True,
) -> list[BaseMessage]:
    """Convert our DTOs into LangChain BaseMessage list.

    `history` items have shape {"role": "user"|"assistant", "content": str}.
    If `rag_context` is provided, it's appended to the system prompt.

    `mention_sources` controls whether the model is allowed to talk about
    where its answers came from. Real customer channels (LINE, FB, etc.)
    pass False — we don't want to leak internal filenames like
    "company_handbook_v3.pdf" to a stranger asking about products. The
    dashboard playground passes True so staff can verify grounding while
    testing.
    """
    full_system = system_prompt or "You are a helpful assistant."
    if rag_context:
        # The "say you don't know" rule below is intentionally scoped to
        # *factual product/business questions* — questions like "what's
        # your name?" must be answered from the [Identity] block in the
        # system prompt above, NOT refused because the answer isn't in
        # the RAG context.
        if mention_sources:
            sources_directive = (
                "Each block is labelled with its source filename in `(source: ...)` — "
                "the filename itself is part of the knowledge and may answer "
                "questions about document or project titles. "
            )
        else:
            sources_directive = (
                "Treat the context as your own internal knowledge. "
                "Do NOT mention, name, quote, or refer to source files, documents, "
                "filenames, knowledge bases, or any internal storage in your reply. "
                "If a user asks where your answer came from or what document "
                "you're using, say only that it's based on the information "
                "available to you — do not name any source. "
            )
        full_system += (
            "\n\nUse the following context to answer factual questions about "
            "the business, products, services, policies, and pricing. "
            f"{sources_directive}"
            "Answer in the same language the user wrote in. "
            "If the user asks a factual question whose answer is not in the "
            "context above, say you don't know and offer to connect them with "
            "a human. This rule does NOT apply to questions about your own "
            "identity (your name, who you are, your role) — those are answered "
            "from the [Identity] section earlier in this prompt.\n\n"
            f"<context>\n{rag_context}\n</context>"
        )
    messages: list[BaseMessage] = [SystemMessage(content=full_system)]
    for h in history:
        role = h.get("role")
        content = h.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role in ("assistant", "ai", "human"):
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=user_message))
    return messages


def generate(
    system_prompt: str,
    history: Iterable[dict],
    user_message: str,
    rag_context: str | None = None,
    model: str | None = None,
    temperature: float = 0.3,
    mention_sources: bool = True,
) -> tuple[str, int]:
    """Run a chat completion. Returns (reply, tokens_used)."""
    llm = _build_llm(model, temperature)
    messages = to_lc_messages(
        system_prompt,
        history,
        user_message,
        rag_context,
        mention_sources=mention_sources,
    )
    result = llm.invoke(messages)

    tokens = 0
    meta = getattr(result, "usage_metadata", None) or {}
    if isinstance(meta, dict):
        tokens = int(meta.get("total_tokens", 0))

    return str(result.content), tokens
