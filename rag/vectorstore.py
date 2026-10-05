"""Pinecone (primary) and FAISS (local fallback) behind one interface."""
import time
from functools import lru_cache

from core.config import settings


class PineconeStore:
    def __init__(self):
        from pinecone import Pinecone, ServerlessSpec

        if not settings.PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY is not set")
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        names = [i["name"] for i in pc.list_indexes()]
        if settings.PINECONE_INDEX not in names:
            pc.create_index(
                name=settings.PINECONE_INDEX, dimension=settings.EMBED_DIM, metric="cosine",
                spec=ServerlessSpec(cloud=settings.PINECONE_CLOUD, region=settings.PINECONE_REGION),
            )
            while not pc.describe_index(settings.PINECONE_INDEX).status["ready"]:
                time.sleep(1)
        self.index = pc.Index(settings.PINECONE_INDEX)

    def upsert(self, chunks, vectors, namespace="default"):
        batch = []
        for c, v in zip(chunks, vectors):
            batch.append({"id": c["id"], "values": v,
                          "metadata": {**c["metadata"], "text": c["text"]}})
        for i in range(0, len(batch), 100):
            self.index.upsert(vectors=batch[i:i + 100], namespace=namespace)
        return len(batch)

    def query(self, vector, top_k=5, filter=None, namespace="default"):
        r = self.index.query(vector=vector, top_k=top_k, include_metadata=True,
                             filter=filter or None, namespace=namespace)
        out = []
        for m in r["matches"]:
            md = dict(m["metadata"])
            out.append({"id": m["id"], "score": float(m["score"]),
                        "text": md.pop("text", ""), "metadata": md})
        return out


class FaissStore:
    """In-memory cosine search (inner product on normalised vectors)."""

    def __init__(self):
        import faiss

        self.index = faiss.IndexFlatIP(settings.EMBED_DIM)
        self.items: list[dict] = []

    def upsert(self, chunks, vectors, namespace="default"):
        import numpy as np

        self.index.add(np.array(vectors, dtype="float32"))
        self.items.extend(chunks)
        return len(chunks)

    def query(self, vector, top_k=5, filter=None, namespace="default"):
        import numpy as np

        if not self.items:
            return []
        scores, ids = self.index.search(np.array([vector], dtype="float32"), min(top_k * 4, len(self.items)))
        out = []
        for s, i in zip(scores[0], ids[0]):
            it = self.items[i]
            if filter and not all(it["metadata"].get(k) == (v.get("$eq") if isinstance(v, dict) else v)
                                  for k, v in filter.items()):
                continue
            out.append({"id": it["id"], "score": float(s), "text": it["text"], "metadata": it["metadata"]})
            if len(out) >= top_k:
                break
        return out


@lru_cache(maxsize=1)
def get_store():
    return FaissStore() if settings.VECTOR_BACKEND == "faiss" else PineconeStore()
