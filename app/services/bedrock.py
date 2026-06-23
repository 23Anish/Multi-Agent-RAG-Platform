import json
import logging
from functools import lru_cache

import boto3
from botocore.config import Config

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_RETRY_CONFIG = Config(
    region_name=settings.aws_region,
    retries={"max_attempts": 3, "mode": "adaptive"},
)


@lru_cache(maxsize=1)
def _bedrock_runtime():
    """Singleton Bedrock runtime client (thread-safe via GIL for boto3)."""
    kwargs: dict = {"config": _RETRY_CONFIG}
    if settings.aws_access_key_id:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    return boto3.client("bedrock-runtime", **kwargs)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Batch embed texts using Amazon Titan Embed v2.

    Bedrock does NOT support true batching on Titan Embed — we call it
    per-text.  For production, swap to Cohere embed-english-v3 which does
    support batch=96.

    Returns a list of float vectors (length = bedrock_embed_dimensions).
    """
    client = _bedrock_runtime()
    vectors: list[list[float]] = []

    for text in texts:
        body = json.dumps(
            {
                "inputText": text[:8192],  # Titan hard limit
                "dimensions": settings.bedrock_embed_dimensions,
                "normalize": True,
            }
        )
        try:
            resp = client.invoke_model(
                modelId=settings.bedrock_embed_model,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(resp["body"].read())
            vectors.append(result["embedding"])
        except Exception as exc:
            logger.error("Bedrock embed error for text slice: %s", exc)
            raise

    return vectors


async def embed_query(query: str) -> list[float]:
    """Convenience wrapper for a single query string."""
    vecs = await embed_texts([query])
    return vecs[0]
