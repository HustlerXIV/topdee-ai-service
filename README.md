# AI Service (Python · FastAPI · LangChain)

The brain. Owns the RAG pipeline and all LLM calls. The Go backend never touches Google's Gemini API or Qdrant directly — it talks to this service over HTTP.

## Run

```bash
cp .env.example .env
# set GOOGLE_API_KEY in .env (get one at https://aistudio.google.com)
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

First run downloads the sentence-transformers model (~80MB). Subsequent runs are fast.

## Endpoints

| Method | Path                  | Body                                                                          |
| ------ | --------------------- | ----------------------------------------------------------------------------- |
| GET    | `/health`             | —                                                                             |
| POST   | `/chat`               | `{tenant_id, conversation_id, agent_id, system_prompt, model, temperature, history, message, knowledge_base_ids}` |
| POST   | `/ingest/text`        | `{tenant_id, knowledge_base_id, source, text}`                                |
| POST   | `/ingest/file`        | multipart `file` + form fields `tenant_id`, `knowledge_base_id`               |
| POST   | `/ingest/kb/delete`   | `{tenant_id, knowledge_base_id}`                                              |

In the Shape 2 platform-agent model the Go backend always passes its env-configured `PLATFORM_SYSTEM_PROMPT`, `PLATFORM_MODEL`, and `PLATFORM_TEMPERATURE` on every `/chat` call. The `agent_id` field is kept in the schema for backward compatibility but isn't used for retrieval — only `tenant_id` and `knowledge_base_ids` decide what context gets pulled from Qdrant.

## Pipeline

```
ingest:  upload → extract_text → chunk → embed → upsert(Qdrant)
chat:    embed(query) → search(Qdrant, filter by tenant+KBs) → top-K
         → build messages [system + context + history + user]
         → ChatGoogleGenerativeAI.invoke → reply
```

Vector store layout: a single Qdrant collection `topdee_chunks` with payload `{tenant_id, knowledge_base_id, source, chunk_index, text}`. Tenant isolation is enforced via Qdrant filters on every search.

## Swapping providers

- **LLM**: change `app/llm.py` — `_build_llm()` is the only place a provider is constructed. To route across providers (Gemini today, plus OpenAI / Anthropic / Groq later), branch on the `model` string prefix and import the matching `ChatXxx` class.
- **Embeddings**: change `app/embeddings.py` — return a list of float lists. Update `EMBEDDING_DIM` and **drop the Qdrant collection** before changing dimension.

## What's stubbed

- Embeddings run inline on the request thread. For large files this should be queued (Redis + RQ/Celery worker). The endpoint already returns a usable response, just slowly for big PDFs.
- No streaming response on `/chat` — add SSE/`StreamingResponse` once the frontend wants it.
- No reranker step. For higher-quality RAG, add a cross-encoder rerank between Qdrant search and LLM call.
