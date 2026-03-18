"""
ChisCode — Security Utilities
JWT token management, password hashing, API key generation, and encryption helpers.
"""
import secrets
import hashlib
import base64
from datetime import datetime, timedelta, timezone
from typing import Any

from cryptography.fernet import Fernet
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Password hashing context (bcrypt)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Fernet encryption for sensitive data (e.g., GitHub tokens)
_fernet: Fernet | None = None


def _get_fernet() -> Fernet:
    """Lazy-initialise Fernet cipher using the app SECRET_KEY."""
    global _fernet
    if _fernet is None:
        # Derive a valid 32-byte base64 key from SECRET_KEY
        key_bytes = hashlib.sha256(settings.secret_key.encode()).digest()
        fernet_key = base64.urlsafe_b64encode(key_bytes)
        _fernet = Fernet(fernet_key)
    return _fernet


# ── Password with 72-byte limit handling ───────────────────

def _prepare_password(password: str) -> str:
    """
    Prepare password for bcrypt hashing by handling the 72-byte limit.
    
    Bcrypt can only handle passwords up to 72 bytes. For longer passwords,
    we pre-hash with SHA-256 to create a fixed-length string.
    This is a standard approach that maintains security while avoiding
    the bcrypt limitation.
    """
    # Check if password exceeds 72 bytes
    password_bytes = password.encode('utf-8')
    if len(password_bytes) > 72:
        logger.debug(
            "Password exceeds 72 bytes, applying SHA-256 pre-hashing",
            original_length=len(password_bytes)
        )
        # Use SHA-256 to create a fixed-length hash (64 chars hex)
        return hashlib.sha256(password_bytes).hexdigest()
    return password


def hash_password(plain: str) -> str:
    """
    Hash a plaintext password using bcrypt.
    
    Automatically handles passwords longer than 72 bytes by pre-hashing
    with SHA-256 before applying bcrypt.
    """
    prepared = _prepare_password(plain)
    return pwd_context.hash(prepared)


def verify_password(plain: str, hashed: str) -> bool:
    """
    Verify a plaintext password against a bcrypt hash.
    
    Applies the same pre-hashing transformation if the password is
    longer than 72 bytes to ensure consistent verification.
    """
    prepared = _prepare_password(plain)
    return pwd_context.verify(prepared, hashed)


# ── JWT ──────────────────────────────────────────────────────

def create_access_token(
    subject: str,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """
    Create a signed JWT access token.

    Args:
        subject: The user ID (stored as JWT 'sub').
        extra_claims: Optional additional claims (e.g., plan, role).
    """
    now = datetime.now(tz=timezone.utc)
    expire = now + timedelta(minutes=settings.jwt_access_token_expire_minutes)

    payload: dict[str, Any] = {
        "sub": str(subject),
        "iat": now,
        "exp": expire,
        "jti": secrets.token_urlsafe(16),  # Unique token ID for blacklisting
        "type": "access",
    }
    if extra_claims:
        payload.update(extra_claims)

    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def create_refresh_token(subject: str) -> str:
    """Create a longer-lived refresh token."""
    now = datetime.now(tz=timezone.utc)
    expire = now + timedelta(days=settings.jwt_refresh_token_expire_days)

    payload = {
        "sub": str(subject),
        "iat": now,
        "exp": expire,
        "jti": secrets.token_urlsafe(16),
        "type": "refresh",
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict[str, Any]:
    """
    Decode and validate a JWT token.

    Raises:
        JWTError: If the token is invalid, expired, or tampered.
    """
    return jwt.decode(
        token,
        settings.jwt_secret_key,
        algorithms=[settings.jwt_algorithm],
    )


# ── Encryption ───────────────────────────────────────────────

def encrypt_value(plaintext: str) -> str:
    """Encrypt a sensitive string value (e.g., GitHub OAuth token)."""
    fernet = _get_fernet()
    return fernet.encrypt(plaintext.encode()).decode()


def decrypt_value(ciphertext: str) -> str:
    """Decrypt a previously encrypted string value."""
    fernet = _get_fernet()
    return fernet.decrypt(ciphertext.encode()).decode()


# ── API Keys ─────────────────────────────────────────────────

def generate_api_key() -> tuple[str, str]:
    """
    Generate a new ChisCode API key.

    Returns:
        (raw_key, hashed_key) — store hashed_key in DB, give raw_key to user.
    """
    raw = f"ck_live_{secrets.token_urlsafe(32)}"
    hashed = hash_password(raw)
    return raw, hashed


def verify_api_key(raw_key: str, hashed_key: str) -> bool:
    """Verify a raw API key against its stored hash."""
    return verify_password(raw_key, hashed_key)
