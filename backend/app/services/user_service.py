"""
ChisCode — User Service
Business logic for user creation, authentication, and profile management.
"""
from datetime import datetime, timezone
from typing import Optional

from bson import ObjectId
from pymongo.errors import DuplicateKeyError

from app.core.logging import get_logger
from app.core.security import (
    generate_api_key,
    hash_password,
    verify_api_key,
    verify_password,
)
from app.db.mongodb import users_collection
from app.schemas.user import UserInDB, UserRegisterRequest

logger = get_logger(__name__)


class UserNotFoundError(Exception):
    pass


class UserAlreadyExistsError(Exception):
    pass


class InvalidCredentialsError(Exception):
    pass


# ── Create ────────────────────────────────────────────────────

async def create_user(req: UserRegisterRequest) -> UserInDB:
    """Register a new user with email/password."""
    coll = users_collection()

    user_doc = {
        "email": req.email,
        "username": req.username,
        "hashed_password": hash_password(req.password),
        "plan": "free",
        "is_active": True,
        "is_verified": False,
        "created_at": datetime.now(tz=timezone.utc),
        "updated_at": datetime.now(tz=timezone.utc),
    }

    try:
        result = await coll.insert_one(user_doc)
        user_doc["_id"] = result.inserted_id
        logger.info("User created", user_id=str(result.inserted_id), email=req.email)
        return UserInDB(**user_doc)
    except DuplicateKeyError:
        raise UserAlreadyExistsError(f"Email '{req.email}' is already registered.")


async def upsert_github_user(
    github_id: str,
    github_username: str,
    email: str,
    avatar_url: str,
    encrypted_token: str,
) -> UserInDB:
    """Create or update a user via GitHub OAuth."""
    coll = users_collection()
    now = datetime.now(tz=timezone.utc)

    result = await coll.find_one_and_update(
        {"github_id": github_id},
        {
            "$set": {
                "github_username": github_username,
                "email": email,
                "avatar_url": avatar_url,
                "github_token_encrypted": encrypted_token,
                "last_login": now,
                "updated_at": now,
            },
            "$setOnInsert": {
                "plan": "free",
                "is_active": True,
                "is_verified": True,   # GitHub verifies email
                "username": github_username,
                "created_at": now,
            },
        },
        upsert=True,
        return_document=True,
    )
    return UserInDB(**result)


# ── Read ──────────────────────────────────────────────────────

async def get_user_by_id(user_id: str) -> UserInDB:
    coll = users_collection()
    doc = await coll.find_one({"_id": ObjectId(user_id)})
    if not doc:
        raise UserNotFoundError(f"User {user_id} not found.")
    return UserInDB(**doc)


async def get_user_by_email(email: str) -> Optional[UserInDB]:
    coll = users_collection()
    doc = await coll.find_one({"email": email})
    return UserInDB(**doc) if doc else None


async def get_user_by_api_key(raw_key: str) -> Optional[UserInDB]:
    """Find a user by their raw API key (checks hash)."""
    coll = users_collection()
    # Cannot query by hash directly — iterate Pro/Yearly users
    # In production, store a deterministic prefix for fast lookup
    cursor = coll.find(
        {"api_key_hash": {"$exists": True}, "plan": {"$in": ["pro", "yearly"]}}
    )
    async for doc in cursor:
        user = UserInDB(**doc)
        if user.api_key_hash and verify_api_key(raw_key, user.api_key_hash):
            return user
    return None


# ── Auth ──────────────────────────────────────────────────────

async def authenticate_user(email: str, password: str) -> UserInDB:
    """Verify email/password credentials."""
    user = await get_user_by_email(email)
    if not user:
        raise InvalidCredentialsError("Invalid email or password.")
    if not user.hashed_password:
        raise InvalidCredentialsError("This account uses GitHub login.")
    if not verify_password(password, user.hashed_password):
        raise InvalidCredentialsError("Invalid email or password.")
    if not user.is_active:
        raise InvalidCredentialsError("Account is deactivated.")

    # Update last_login timestamp
    await users_collection().update_one(
        {"_id": ObjectId(user.id)},
        {"$set": {"last_login": datetime.now(tz=timezone.utc)}},
    )
    return user


# ── Update ────────────────────────────────────────────────────

async def update_user_plan(user_id: str, plan: str) -> None:
    """Update a user's subscription plan (called by RevenueCat webhook)."""
    await users_collection().update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"plan": plan, "updated_at": datetime.now(tz=timezone.utc)}},
    )
    logger.info("User plan updated", user_id=user_id, plan=plan)


async def generate_user_api_key(user_id: str) -> str:
    """
    Generate and store a new API key for a Pro/Yearly user.
    Returns the raw key — it will not be retrievable again.
    """
    user = await get_user_by_id(user_id)
    if user.plan not in ("pro", "yearly"):
        raise PermissionError("API key access requires Pro or Yearly plan.")

    raw_key, hashed = generate_api_key()
    await users_collection().update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"api_key_hash": hashed, "updated_at": datetime.now(tz=timezone.utc)}},
    )
    logger.info("API key generated", user_id=user_id)
    return raw_key


async def revoke_user_api_key(user_id: str) -> None:
    await users_collection().update_one(
        {"_id": ObjectId(user_id)},
        {"$unset": {"api_key_hash": ""}, "$set": {"updated_at": datetime.now(tz=timezone.utc)}},
  )
