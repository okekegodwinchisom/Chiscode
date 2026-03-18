"""
ChisCode — Redis Client
=======================
Uses redis-py (already installed via pyproject.toml as redis[hiredis]).
Connects to Upstash via the rediss:// URL — no extra packages needed.

HF Spaces Secret to set:
    UPSTASH_REDIS_REST_URL   → use the redis:// connection string from
                               Upstash Console → your DB → Connect → redis-py
                               Format: rediss://default:<token>@<host>.upstash.io:6379

The rediss:// scheme (double-s) enables TLS automatically in redis-py.
"""
import redis.asyncio as aioredis

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_redis: aioredis.Redis | None = None


# ── Lifecycle ────────────────────────────────────────────────────

async def connect() -> None:
    global _redis
    host = settings.upstash_redis_rest_url.split("@")[-1] if "@" in settings.upstash_redis_rest_url else settings.upstash_redis_rest_url
    logger.info("Connecting to Redis", host=host)

    is_tls = settings.upstash_redis_rest_url.startswith("rediss://")

    _redis = await aioredis.from_url(
        settings.upstash_redis_rest_url,
        encoding="utf-8",
        decode_responses=True,
        max_connections=10,
        socket_timeout=10,
        socket_connect_timeout=10,
        ssl_cert_reqs="none" if is_tls else None,
    )
    await _redis.ping()
    logger.info("✅ Upstash Redis connected successfully", tls=is_tls)


async def disconnect() -> None:
    global _redis
    if _redis:
        await _redis.aclose()
        _redis = None
        logger.info("Redis connection closed")


def get_redis() -> aioredis.Redis:
    if _redis is None:
        raise RuntimeError("Redis not initialised — call connect() first.")
    return _redis


# ── Rate Limiting ────────────────────────────────────────────────

RATE_LIMIT_KEY = "rate:{user_id}:{date}"


async def check_and_increment_rate_limit(
    user_id: str,
    daily_limit: int,
    date_str: str,
) -> tuple[bool, int, int]:
    """
    Atomically increment today's request counter and check against limit.
    Uses individual commands instead of pipeline — avoids AsyncPipeline.execute()
    signature incompatibility with Upstash's redis-py connection.
    Returns (allowed, current_count, daily_limit).
    """
    r = get_redis()
    key = RATE_LIMIT_KEY.format(user_id=user_id, date=date_str)
    current = int(await r.incr(key))
    await r.expire(key, 86400)  # Reset at end of day (86400s)
    return current <= daily_limit, current, daily_limit


async def get_current_usage(user_id: str, date_str: str) -> int:
    key = RATE_LIMIT_KEY.format(user_id=user_id, date=date_str)
    val = await get_redis().get(key)
    return int(val) if val else 0


# ── Token Blacklisting ───────────────────────────────────────────

async def blacklist_token(jti: str, ttl_seconds: int) -> None:
    await get_redis().setex(f"blacklist:{jti}", ttl_seconds, "1")


async def is_token_blacklisted(jti: str) -> bool:
    return bool(await get_redis().exists(f"blacklist:{jti}"))


# ── General Cache ────────────────────────────────────────────────

async def cache_set(key: str, value: str, ttl: int = 300) -> None:
    await get_redis().setex(key, ttl, value)


async def cache_get(key: str) -> str | None:
    return await get_redis().get(key)


async def cache_delete(key: str) -> None:
    await get_redis().delete(key)


# ── WebSocket Presence ───────────────────────────────────────────

async def set_user_presence(project_id: str, user_id: str, data: str) -> None:
    r = get_redis()
    await r.hset(f"presence:{project_id}", mapping={user_id: data})
    await r.expire(f"presence:{project_id}", 120)


async def remove_user_presence(project_id: str, user_id: str) -> None:
    await get_redis().hdel(f"presence:{project_id}", user_id)


async def get_project_presence(project_id: str) -> dict[str, str]:
    return await get_redis().hgetall(f"presence:{project_id}") or {}
    
