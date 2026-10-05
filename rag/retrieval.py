from rag.embeddings import embed_one
from rag.vectorstore import get_store


def retrieve(query: str, top_k: int = 5, filter: dict | None = None) -> list[dict]:
    return get_store().query(embed_one(query), top_k=top_k, filter=filter)


def dedupe(docs: list[dict]) -> list[dict]:
    seen, out = set(), []
    for d in sorted(docs, key=lambda x: -x["score"]):
        if d["id"] not in seen:
            seen.add(d["id"])
            out.append(d)
    return out
