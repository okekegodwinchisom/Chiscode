"""
ChisCode — User Schemas
Pydantic v2 models for user data validation and serialization.
"""
from datetime import datetime
from typing import Annotated, Any, Literal, Optional

from bson import ObjectId
from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator


# ── BSON ObjectId helper ──────────────────────────────────────

class PyObjectId(str):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: Any) -> str:
        if isinstance(v, ObjectId):
            return str(v)
        if isinstance(v, str) and ObjectId.is_valid(v):
            return v
        raise ValueError(f"Invalid ObjectId: {v!r}")

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema
        return core_schema.no_info_plain_validator_function(cls.validate)


PlanType = Literal["free", "basic", "pro", "yearly"]


# ── Core User Model (stored in MongoDB) ──────────────────────

class UserInDB(BaseModel):
    """Represents a user document as stored in MongoDB."""
    model_config = {"populate_by_name": True, "arbitrary_types_allowed": True}

    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    email: EmailStr
    username: str = Field(min_length=3, max_length=50)
    hashed_password: Optional[str] = None    # None for OAuth-only users
    is_active: bool = True
    is_verified: bool = False

    # GitHub OAuth
    github_id: Optional[str] = None
    github_username: Optional[str] = None
    github_token_encrypted: Optional[str] = None  # Encrypted with Fernet

    # Subscription / Plan
    plan: PlanType = "free"
    revenuecat_customer_id: Optional[str] = None

    # API Access (Pro/Yearly only)
    api_key_hash: Optional[str] = None

    # Metadata
    avatar_url: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None


# ── Request Schemas (API input) ───────────────────────────────

class UserRegisterRequest(BaseModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("username")
    @classmethod
    def username_alphanumeric(cls, v: str) -> str:
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Username must be alphanumeric (underscores and hyphens allowed)")
        return v.lower()

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserLoginRequest(BaseModel):
    email: EmailStr
    password: str


class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8, max_length=128)


# ── Response Schemas (API output) ────────────────────────────

class UserPublic(BaseModel):
    """Safe user representation returned to clients (no secrets)."""
    model_config = {"populate_by_name": True}

    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    email: EmailStr
    username: str
    plan: PlanType
    avatar_url: Optional[str] = None
    github_username: Optional[str] = None
    is_verified: bool
    created_at: datetime
    has_api_key: bool = False

    @model_validator(mode="before")
    @classmethod
    def set_has_api_key(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data["has_api_key"] = bool(data.get("api_key_hash"))
        return data


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    user: UserPublic


class ApiKeyResponse(BaseModel):
    """Returned once when API key is generated — raw key shown once only."""
    api_key: str
    message: str = "Store this key securely — it will not be shown again."


class UsageResponse(BaseModel):
    plan: PlanType
    daily_limit: int
    used_today: int
    remaining: int
    resets_at: str   # ISO date string

    # Add these to your app/schemas/user.py

class UserUpdate(BaseModel):
    """Schema for updating user profile."""
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    avatar_url: Optional[str] = None
    full_name: Optional[str] = Field(None, max_length=100)


class UserDeleteResponse(BaseModel):
    """Response after user deletion."""
    message: str = "Account successfully deleted"
