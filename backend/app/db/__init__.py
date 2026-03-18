"""
ChisCode — Database Package
============================
Exposes MongoDB and Redis clients as a single importable namespace.

Usage:
    from app.db import mongodb, redis_client          # module-level
    from app.db.mongodb import get_db, users_collection
    from app.db.redis_client import get_redis, blacklist_token

Lifecycle (called from app/main.py lifespan):
    await mongodb.connect()       # connects Motor client to Atlas, creates indexes
    await redis_client.connect()  # connects redis-py to Upstash via rediss://
    ...
    await mongodb.disconnect()
    await redis_client.disconnect()
"""

from app.db import mongodb, redis_client

# ── MongoDB ────────────────────────────────────────────────────────
from app.db.mongodb import (
    get_client,
    get_db,
    users_collection,
    projects_collection,
    project_versions_collection,
    sessions_collection,
    templates_collection,
)

# ── Redis ──────────────────────────────────────────────────────────
from app.db.redis_client import (
    get_redis,
    check_and_increment_rate_limit,
    get_current_usage,
    blacklist_token,
    is_token_blacklisted,
    cache_set,
    cache_get,
    cache_delete,
    set_user_presence,
    remove_user_presence,
    get_project_presence,
)

__all__ = [
    # modules
    "mongodb",
    "redis_client",

    # MongoDB client
    "get_client",
    "get_db",

    # MongoDB collections
    "users_collection",
    "projects_collection",
    "project_versions_collection",
    "sessions_collection",
    "templates_collection",

    # Redis client
    "get_redis",

    # Redis operations
    "check_and_increment_rate_limit",
    "get_current_usage",
    "blacklist_token",
    "is_token_blacklisted",
    "cache_set",
    "cache_get",
    "cache_delete",
    "set_user_presence",
    "remove_user_presence",
    "get_project_presence",
]
