import json
import logging
from typing import Any

import redis.asyncio as aioredis

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_pool: aioredis.ConnectionPool | None = None


def _get_pool() -> aioredis.ConnectionPool:
    global _pool
    if _pool is None:
        _pool = aioredis.ConnectionPool.from_url(
            settings.redis_url, max_connections=20, decode_responses=True
        )
    return _pool


def get_redis() -> aioredis.Redis:
    return aioredis.Redis(connection_pool=_get_pool())


async def cache_get(key: str) -> Any | None:
    r = get_redis()
    try:
        raw = await r.get(key)
        return json.loads(raw) if raw is not None else None
    except Exception as exc:
        logger.warning("Redis GET failed for key=%s: %s", key, exc)
        return None


async def cache_set(key: str, value: Any, ttl: int | None = None) -> None:
    r = get_redis()
    try:
        ttl = ttl or settings.cache_ttl_seconds
        await r.set(key, json.dumps(value), ex=ttl)
    except Exception as exc:
        logger.warning("Redis SET failed for key=%s: %s", key, exc)


async def cache_delete(key: str) -> None:
    r = get_redis()
    try:
        await r.delete(key)
    except Exception as exc:
        logger.warning("Redis DEL failed for key=%s: %s", key, exc)


async def cache_ping() -> bool:
    try:
        r = get_redis()
        return await r.ping()
    except Exception:
        return False
