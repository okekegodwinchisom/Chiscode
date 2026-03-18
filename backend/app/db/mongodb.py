"""
ChisCode — MongoDB Client
=========================
Async Motor client configured for MongoDB Atlas:
  - TLS enabled with CA bundle from certifi (system certs are unreliable in containers)
  - tlsAllowInvalidCertificates=False — strict cert validation, never disabled in prod
  - Connection string parsed from MONGODB_URL env var (mongodb+srv:// format for Atlas)
  - Index creation is idempotent — safe to run on every startup
  - Soft failure mode: connect() logs and re-raises so main.py can start degraded
"""

import ssl

import certifi
import motor.motor_asyncio
from pymongo import ASCENDING, DESCENDING, IndexModel
from pymongo.errors import ConnectionFailure, ConfigurationError, ServerSelectionTimeoutError

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Client singletons ────────────────────────────────────────────
_client: motor.motor_asyncio.AsyncIOMotorClient | None = None
_db:     motor.motor_asyncio.AsyncIOMotorDatabase | None = None


def get_client() -> motor.motor_asyncio.AsyncIOMotorClient:
    if _client is None:
        raise RuntimeError("MongoDB client not initialised — call connect() first.")
    return _client


def get_db() -> motor.motor_asyncio.AsyncIOMotorDatabase:
    if _db is None:
        raise RuntimeError("MongoDB database not initialised — call connect() first.")
    return _db


# ── TLS context ──────────────────────────────────────────────────

def _make_tls_context() -> ssl.SSLContext:
    """
    Build an SSLContext that trusts the certifi CA bundle.

    Why certifi instead of the system store?
    - python:3.11-slim ships minimal system certs — Atlas roots may be absent.
    - certifi is always pinned and up-to-date with the Let's Encrypt / ISRG root.
    - Using ssl.create_default_context(cafile=certifi.where()) is the pattern
      recommended by both PyMongo and the certifi authors.
    """
    ctx = ssl.create_default_context(cafile=certifi.where())
    ctx.verify_mode        = ssl.CERT_REQUIRED   # never skip verification
    ctx.check_hostname     = True                # enforce SNI hostname check
    ctx.minimum_version    = ssl.TLSVersion.TLSv1_2  # Atlas requires >= TLS 1.2
    return ctx


# ── Lifecycle ────────────────────────────────────────────────────

async def connect() -> None:
    """
    Initialise the Motor client and verify Atlas connectivity.

    Connection string format (from MONGODB_URL env var):
        mongodb+srv://user:password@cluster.mongodb.net/?retryWrites=true&w=majority

    The +srv scheme handles DNS SRV lookup automatically — no need to list
    individual shard hosts. Motor + PyMongo resolve the SRV record at connect time.

    TLS is always on for Atlas (enforced by the cluster). We pass our own
    SSLContext so certifi's CA bundle is used rather than the container's
    sparse system store.
    """
    global _client, _db

    # Log the host only — never log credentials
    safe_url = _redact_url(settings.mongodb_url)
    logger.info("Connecting to MongoDB Atlas", host=safe_url)

    tls_ctx = _make_tls_context()

    _client = motor.motor_asyncio.AsyncIOMotorClient(
        settings.mongodb_url,

        # ── TLS / SSL ──────────────────────────────────────────
        # tlsCAFile is the preferred PyMongo >=4.x kwarg for the CA bundle.
        # Passing ssl_context covers the underlying pymongo transport layer.
        tlsCAFile=certifi.where(),
        tls=True,
        tlsAllowInvalidCertificates=False,   # NEVER True in production

        # ── Timeouts ───────────────────────────────────────────
        serverSelectionTimeoutMS=8000,   # how long to wait for a viable server
        connectTimeoutMS=8000,           # socket connect timeout
        socketTimeoutMS=20000,           # per-operation socket timeout

        # ── Pool ───────────────────────────────────────────────
        # HF Spaces runs 2 uvicorn workers; keep pool modest to avoid
        # hitting Atlas M0/M2 connection limits (500 and 500 respectively).
        maxPoolSize=10,
        minPoolSize=1,

        # ── Reliability ────────────────────────────────────────
        retryWrites=True,
        retryReads=True,
        w="majority",          # write concern — confirms write on primary + 1 secondary
    )

    try:
        # Ping forces a real round-trip; raises immediately if Atlas is unreachable
        await _client.admin.command("ping")
        logger.info("MongoDB Atlas connection established", db=settings.mongodb_db)
    except (ConnectionFailure, ServerSelectionTimeoutError, ConfigurationError) as exc:
        logger.error("MongoDB Atlas connection failed", error=str(exc), host=safe_url)
        _client = None
        raise

    _db = _client[settings.mongodb_db]
    await _create_indexes()


async def disconnect() -> None:
    """Close the Motor client and release connection pool."""
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db     = None
        logger.info("MongoDB connection closed")


# ── Index creation ───────────────────────────────────────────────

async def _create_indexes() -> None:
    """
    Idempotently create all required collection indexes.
    Safe to call on every startup — PyMongo skips indexes that already exist.
    """
    db = get_db()

    # ── users ──────────────────────────────────────────────────
    await db.users.create_indexes([
        IndexModel([("email",       ASCENDING)], unique=True,  name="email_unique"),
        IndexModel([("github_id",   ASCENDING)], sparse=True,  name="github_id_sparse"),
        IndexModel([("api_key_hash",ASCENDING)], sparse=True,  name="api_key_sparse"),
        IndexModel([("created_at",  DESCENDING)],              name="created_at_desc"),
    ])

    # ── projects ───────────────────────────────────────────────
    await db.projects.create_indexes([
        IndexModel([("user_id",   ASCENDING)],                             name="user_id"),
        IndexModel([("user_id",   ASCENDING), ("created_at", DESCENDING)], name="user_projects_desc"),
        IndexModel([("status",    ASCENDING)],                             name="status"),
    ])

    # ── project_versions ───────────────────────────────────────
    await db.project_versions.create_indexes([
        IndexModel([("project_id", ASCENDING)],                           name="project_id"),
        IndexModel([("project_id", ASCENDING), ("version", DESCENDING)],
                   unique=True, name="project_version_unique"),
    ])

    # ── sessions (refresh tokens) ──────────────────────────────
    # TTL index auto-expires documents when expires_at is reached.
    # expireAfterSeconds=0 means expire exactly at the field's datetime value.
    await db.sessions.create_indexes([
        IndexModel([("user_id",   ASCENDING)],                name="session_user_id"),
        IndexModel([("jti",       ASCENDING)], unique=True,   name="jti_unique"),
        IndexModel([("expires_at",ASCENDING)], expireAfterSeconds=0, name="ttl_expires"),
    ])

    logger.info("MongoDB indexes verified / created")


# ── Collection accessors ─────────────────────────────────────────
# Import these in services instead of calling get_db() directly —
# makes unit-testing easier (mock the accessor, not the client).

def users_collection():
    return get_db().users

def projects_collection():
    return get_db().projects

def project_versions_collection():
    return get_db().project_versions

def sessions_collection():
    return get_db().sessions

def templates_collection():
    return get_db().templates


# ── Internal helpers ─────────────────────────────────────────────

def _redact_url(url: str) -> str:
    """Return the host portion of a MongoDB URL, credentials stripped."""
    try:
        # mongodb+srv://user:pass@cluster.mongodb.net/... → cluster.mongodb.net/...
        return url.split("@")[-1].split("/")[0]
    except Exception:
        return "<url-parse-error>"
        
