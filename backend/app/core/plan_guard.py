"""
ChisCode — Plan Guard
======================
FastAPI dependency factories for plan + entitlement enforcement.

Deploy to: backend/app/core/plan_guard.py

The key design decision: get_current_user is resolved via FastAPI's
Depends() at request time, never imported at module load time.
This eliminates any possibility of circular imports.

Usage:

    from app.core.plan_guard import require_plan, require_feature, require_generation_quota

    @router.post("/projects/generate")
    async def generate(current_user = Depends(require_generation_quota)):
        ...

    @router.post("/projects/{id}/deploy")
    async def deploy(_: None = Depends(require_feature("deploy"))):
        ...

    @router.post("/users/me/api-key")
    async def gen_key(current_user = Depends(require_plan("pro", "yearly"))):
        ...
"""
from __future__ import annotations

from fastapi import Depends, HTTPException, status

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.user import UserInDB

logger = get_logger(__name__)


# ── Resolve get_current_user without importing app.api at module level ──
#
# Instead of:  from app.api.deps import get_current_user   ← causes circular import
#
# We replicate the exact same dependency inline. deps.py already imports
# from app.core and app.services — nothing from app.api — so there is no
# cycle when plan_guard does the same.

from app.core.security import decode_token
from app.db import redis_client
from app.services import user_service
from fastapi import Cookie, Header
from jose import JWTError


async def _get_current_user(
    access_token:  str | None = Cookie(default=None),
    authorization: str | None = Header(default=None),
    x_chiscode_api_key: str | None = Header(default=None, alias="X-ChisCode-API-Key"),
) -> UserInDB:
    """
    Inline re-implementation of get_current_user from deps.py.
    Accepts JWT cookie/header OR API key — identical logic, no cross-import.
    """
    # ── API key path ──────────────────────────────────────────
    if x_chiscode_api_key:
        user = await user_service.get_user_by_api_key(x_chiscode_api_key)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key.",
            )
        if user.plan not in ("pro", "yearly"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="API key access requires Pro or Yearly plan.",
            )
        return user

    # ── JWT path ──────────────────────────────────────────────
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    token = access_token
    if not token and authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer":
            raise credentials_exception

    if not token:
        raise credentials_exception

    try:
        payload = decode_token(token)
        user_id: str = payload.get("sub", "")
        jti:     str = payload.get("jti", "")
        if not user_id or not jti:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    if await redis_client.is_token_blacklisted(jti):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked. Please log in again.",
        )

    try:
        user = await user_service.get_user_by_id(user_id)
    except user_service.UserNotFoundError:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated.",
        )

    return user


# ── Plan gate ──────────────────────────────────────────────────

def require_plan(*plans: str):
    """Require the user to be on one of the listed plans."""
    async def _check(
        current_user: UserInDB = Depends(_get_current_user),
    ) -> UserInDB:
        if current_user.plan not in plans:
            plan_list = " or ".join(p.capitalize() for p in plans)
            upgrade_to = plans[0] if plans else None
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error":        "plan_required",
                    "message":      f"This feature requires {plan_list} plan.",
                    "current_plan": current_user.plan,
                    "upgrade_url":  f"/api/v1/billing/checkout/{upgrade_to}" if upgrade_to else None,
                },
            )
        return current_user
    return _check


# ── Feature gate ───────────────────────────────────────────────

def require_feature(feature: str):
    """
    Gate on a named feature entitlement.
    Features: 'api_key', 'deploy', 'priority'
    """
    _plan_map: dict[str, tuple[str, ...]] = {
        "api_key":  ("pro", "yearly"),
        "deploy":   ("basic", "pro", "yearly"),
        "priority": ("pro", "yearly"),
    }

    async def _check(
        current_user: UserInDB = Depends(_get_current_user),
    ) -> UserInDB:
        from app.services.billing_service import check_feature_allowed
        if not check_feature_allowed(current_user.plan, feature):
            required  = _plan_map.get(feature, ())
            upgrade   = required[0] if required else None
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error":        "feature_not_available",
                    "feature":      feature,
                    "message":      f"'{feature}' is not available on your current plan.",
                    "current_plan": current_user.plan,
                    "upgrade_url":  f"/api/v1/billing/checkout/{upgrade}" if upgrade else None,
                },
            )
        return current_user
    return _check


# ── Generation quota ───────────────────────────────────────────

async def require_generation_quota(
    current_user: UserInDB = Depends(_get_current_user),
) -> UserInDB:
    """
    Verify the user has remaining daily generation quota.
    Raises HTTP 429 with Retry-After header when the limit is hit.
    """
    from app.services.billing_service import check_generation_allowed
    allowed, reason = await check_generation_allowed(
        str(current_user.id), current_user.plan
    )
    if not allowed:
        limit      = settings.get_rate_limit(current_user.plan)
        next_plans = {"free": "basic", "basic": "pro", "pro": "yearly"}
        upgrade_to = next_plans.get(current_user.plan)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error":        "quota_exceeded",
                "message":      reason,
                "daily_limit":  limit,
                "current_plan": current_user.plan,
                "upgrade_url":  f"/api/v1/billing/checkout/{upgrade_to}" if upgrade_to else None,
            },
            headers={"Retry-After": "86400"},
        )
    return current_user


# ── Billing flag check ─────────────────────────────────────────

async def require_no_billing_issue(
    current_user: UserInDB = Depends(_get_current_user),
) -> UserInDB:
    """Block access for accounts with an unresolved billing issue."""
    from bson import ObjectId
    from app.db.mongodb import users_collection
    doc = await users_collection().find_one(
        {"_id": ObjectId(current_user.id)}, {"billing_flagged": 1}
    )
    if doc and doc.get("billing_flagged"):
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail={
                "error":      "billing_issue",
                "message":    "Your account has a billing issue. Please update your payment method.",
                "portal_url": "/billing/portal",
            },
        )
    return current_user
    
