"""
Knowledgebase diagnostic script.
Run from the ai-service directory:
    python diagnose_kb.py [tenant_id] [query]

With no args it prints a summary of ALL tenants and KBs in Qdrant.
"""
import sys
import os
from dotenv import load_dotenv

load_dotenv(".env")

from app.config import settings
from app.vector_store import VectorStore
from app.embeddings import embed_query

def main():
    tenant_filter = sys.argv[1] if len(sys.argv) > 1 else None
    query = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"\n{'='*60}")
    print(f"Qdrant URL  : {settings.qdrant_url}")
    print(f"Collection  : {settings.qdrant_collection}")
    print(f"Embed model : {settings.embedding_model}")
    print(f"Embed dim   : {settings.embedding_dim}")
    print(f"{'='*60}\n")

    store = VectorStore()

    # ── 1. Does the collection exist? ────────────────────────────────
    try:
        info = store.client.get_collection(settings.qdrant_collection)
        print(f"✅ Collection exists")
        print(f"   Vectors count : {info.vectors_count}")
        print(f"   Points count  : {info.points_count}")
        print(f"   Vector size   : {info.config.params.vectors.size}")
    except Exception as e:
        print(f"❌ Collection does not exist or Qdrant unreachable: {e}")
        return

    if info.points_count == 0:
        print("\n⚠️  Collection is EMPTY — no data has been ingested yet.")
        print("   Upload a file via the Knowledge Base UI to populate it.")
        return

    # ── 2. Scroll a sample to see what tenants/KBs are stored ───────
    print(f"\n--- Sample of stored points (up to 20) ---")
    results, _ = store.client.scroll(
        collection_name=settings.qdrant_collection,
        limit=20,
        with_payload=True,
        with_vectors=False,
    )
    tenants_kbs: dict[str, set] = {}
    for r in results:
        p = r.payload or {}
        tid = p.get("tenant_id", "?")
        kid = p.get("knowledge_base_id", "?")
        tenants_kbs.setdefault(tid, set()).add(kid)

    for tid, kids in tenants_kbs.items():
        print(f"  tenant_id={tid}")
        for kid in kids:
            # count points for this KB
            count = store.client.count(
                collection_name=settings.qdrant_collection,
                count_filter={
                    "must": [
                        {"key": "tenant_id",        "match": {"value": tid}},
                        {"key": "knowledge_base_id","match": {"value": kid}},
                    ]
                },
                exact=True,
            )
            print(f"    knowledge_base_id={kid}  →  {count.count} chunks")

    # ── 3. Optionally run a search ───────────────────────────────────
    if tenant_filter and query:
        print(f"\n--- Search test ---")
        print(f"  tenant_id : {tenant_filter}")
        print(f"  query     : {query!r}")
        vec = embed_query(query)
        hits = store.search(
            query_vector=vec,
            tenant_id=tenant_filter,
            knowledge_base_ids=None,   # search ALL KBs for this tenant
            top_k=5,
        )
        if not hits:
            print("  ❌ No hits returned — check tenant_id matches what's in Qdrant above")
        else:
            for i, h in enumerate(hits, 1):
                print(f"\n  Hit {i}  score={h.score:.4f}  kb={h.knowledge_base_id}")
                print(f"  Text: {h.text[:200]!r}")
    elif tenant_filter:
        print(f"\nTip: pass a query as a 3rd argument to test search:")
        print(f"  python diagnose_kb.py {tenant_filter!r} 'your question here'")
    else:
        print(f"\nTip: pass tenant_id + query to test retrieval:")
        print(f"  python diagnose_kb.py <tenant_id> 'your question here'")

if __name__ == "__main__":
    main()
