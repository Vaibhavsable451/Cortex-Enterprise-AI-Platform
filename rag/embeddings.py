from functools import lru_cache

from core.config import settings


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.EMBED_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    return _model().encode(texts, normalize_embeddings=True, batch_size=32).tolist()


def embed_one(text: str) -> list[float]:
    return embed([text])[0]
