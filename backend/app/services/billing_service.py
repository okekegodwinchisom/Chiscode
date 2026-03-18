"""
ChisCode — Billing Service (Polar.sh)
======================================
All payment logic using Polar.sh as the payments provider.
https://docs.polar.sh/api

Polar replaces RevenueCat. Key differences:
  - Webhooks use Polar's signature format (HMAC-SHA256, header: webhook-signature)
  - Checkout sessions created server-side via Polar API
  - Subscription state queried via Polar REST API
  - Events: order.created, subscription.created, subscription.updated,
            subscription.active, subscription.canceled, subscription.revoked
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import httpx
from bson import ObjectId
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import get_logger
from app.db.mongodb import users_collection
from app.db import redis_client

logger = get_logger(__name__)

_POLAR_BASE = "https://api.polar.sh/v1"

# ── Product ID → plan name ─────────────────────────────────────
def _product_plan_map() -> dict[str, str]:
    return {
        settings.polar_product_basic:  "basic",
        settings.polar_product_pro:    "pro",
        settings.polar_product_yearly: "yearly",
    }

# ── Plan display metadata ──────────────────────────────────────
PLAN_META = {
    "free": {
        "name":        "Free",
        "price":       "$0",
        "daily_limit": 5,
        "features": [
            "5 generations/day",
            "Export as ZIP",
            "GitHub push",
            "Community support",
        ],
        "missing": ["API key access", "One-click deployment", "Priority generation"],
    },
    "basic": {
        "name":        "Basic",
        "price":       "$25/mo",
        "daily_limit": 100,
        "features": [
            "100 generations/day",
            "All deployment platforms",
            "Export as ZIP",
            "GitHub push",
            "Email support",
        ],
        "missing": ["API key access", "Priority generation"],
    },
    "pro": {
        "name":        "Pro",
        "price":       "$120/mo",
        "daily_limit": 1000,
        "features": [
            "1,000 generations/day",
            "API key access",
            "Priority generation queue",
            "All deployment platforms",
            "GitHub push",
            "Priority support",
        ],
        "missing": [],
    },
    "yearly": {
        "name":        "Yearly",
        "price":       "$1,000/yr",
        "daily_limit": 1000,
        "features": [
            "1,000 generations/day",
            "API key access",
            "Priority generation queue",
            "All deployment platforms",
            "GitHub push",
            "Priority support",
            "2 months free vs monthly Pro",
        ],
        "missing": [],
    },
}


# ── Schemas ────────────────────────────────────────────────────

class CustomerInfo(BaseModel):
    user_id:         str
    plan:            str
    polar_id:        Optional[str]
    is_active:       bool
    billing_flagged: bool
    entitlements:    list[str]
    portal_url:      Optional[str]


class UsageSummary(BaseModel):
    plan:        str
    plan_name:   str
    plan_price:  str
    daily_limit: int
    used_today:  int
    remaining:   int
    resets_at:   str
    features:    list[str]
    missing:     list[str]
    upgrade_url: Optional[str]


# ── Polar API helpers ──────────────────────────────────────────

def _polar_headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.polar_access_token}",
        "Content-Type":  "application/json",
    }


async def create_checkout_session(
    product_id: str,
    user_id:    str,
    email:      str,
    success_url: str,
) -> str | None:
    """
    Create a Polar checkout session and return the checkout URL.
    Polar docs: POST /v1/checkouts/
    """
    if not settings.polar_access_token or not product_id:
        return None
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                f"{_POLAR_BASE}/checkouts/",
                headers=_polar_headers(),
                json={
                    "product_id":         product_id,
                    "customer_email":     email,
                    "metadata":  {"chiscode_user_id": user_id},
                    "success_url":        success_url,
                    "allow_discount_codes": True,
                },
            )
            r.raise_for_status()
            return r.json().get("url")
    except Exception as exc:
        logger.error("Polar checkout creation failed", error=str(exc))
        return None


async def get_polar_subscription(polar_subscription_id: str) -> dict | None:
    """Fetch a subscription from Polar API."""
    if not settings.polar_access_token:
        return None
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                f"{_POLAR_BASE}/subscriptions/{polar_subscription_id}",
                headers=_polar_headers(),
            )
            r.raise_for_status()
            return r.json()
    except Exception as exc:
        logger.warning("Polar subscription fetch failed", error=str(exc))
        return None


async def get_customer_portal_url(polar_customer_id: str) -> str | None:
    """Get Polar customer portal URL for managing subscriptions."""
    if not settings.polar_access_token or not polar_customer_id:
        return None
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                f"{_POLAR_BASE}/customer-sessions",
                headers=_polar_headers(),
                json={"customer_id": polar_customer_id},
            )
            r.raise_for_status()
            return r.json().get("customer_portal_url")
    except Exception as exc:
        logger.warning("Polar portal URL failed", error=str(exc))
        return None


# ── Plan enforcement ───────────────────────────────────────────

async def check_generation_allowed(user_id: str, plan: str) -> tuple[bool, str]:
    from datetime import date
    today = date.today().isoformat()
    used  = await redis_client.get_current_usage(user_id, today)
    limit = settings.get_rate_limit(plan)

    if used >= limit:
        next_pl = _next_plan(plan)
        return False, (
            f"Daily limit reached ({used}/{limit}). Resets at midnight UTC."
            + (f" Upgrade to {next_pl.capitalize()} for more." if next_pl else "")
        )

    doc = await users_collection().find_one(
        {"_id": ObjectId(user_id)}, {"billing_flagged": 1}
    )
    if doc and doc.get("billing_flagged"):
        return False, "Your account has a billing issue. Please update your payment method."

    return True, ""


def check_feature_allowed(plan: str, feature: str) -> bool:
    gates: dict[str, list[str]] = {
        "api_key":  ["pro", "yearly"],
        "deploy":   ["basic", "pro", "yearly"],
        "priority": ["pro", "yearly"],
    }
    return plan in gates.get(feature, [])


def _next_plan(current: str) -> str | None:
    order = ["free", "basic", "pro", "yearly"]
    idx   = order.index(current) if current in order else -1
    return order[idx + 1] if idx < len(order) - 2 else None


def get_product_id(plan: str) -> str:
    return {
        "basic":  settings.polar_product_basic,
        "pro":    settings.polar_product_pro,
        "yearly": settings.polar_product_yearly,
    }.get(plan, "")


# ── Usage summary ──────────────────────────────────────────────

async def get_usage_summary(user_id: str, plan: str) -> UsageSummary:
    from datetime import date, timedelta
    today     = date.today()
    used      = await redis_client.get_current_usage(user_id, today.isoformat())
    limit     = settings.get_rate_limit(plan)
    resets_at = datetime.combine(
        today + timedelta(days=1), datetime.min.time()
    ).isoformat() + "Z"
    meta = PLAN_META.get(plan, PLAN_META["free"])
    next_pl = _next_plan(plan)

    return UsageSummary(
        plan=plan,
        plan_name=meta["name"],
        plan_price=meta["price"],
        daily_limit=limit,
        used_today=used,
        remaining=max(0, limit - used),
        resets_at=resets_at,
        features=meta["features"],
        missing=meta["missing"],
        upgrade_url=f"/api/v1/billing/checkout/{next_pl}" if next_pl else None,
    )


# ── Billing flag ───────────────────────────────────────────────

async def flag_billing_issue(user_id: str) -> None:
    await users_collection().update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {
            "billing_flagged":    True,
            "billing_flagged_at": datetime.now(tz=timezone.utc),
            "updated_at":         datetime.now(tz=timezone.utc),
        }},
    )
    logger.warning("Billing issue flagged", user_id=user_id)


async def clear_billing_flag(user_id: str) -> None:
    await users_collection().update_one(
        {"_id": ObjectId(user_id)},
        {"$unset": {"billing_flagged": "", "billing_flagged_at": ""},
         "$set":   {"updated_at": datetime.now(tz=timezone.utc)}},
    )


# ── Customer info ──────────────────────────────────────────────

async def get_customer_info(user_id: str) -> CustomerInfo:
    doc = await users_collection().find_one({"_id": ObjectId(user_id)})
    if not doc:
        raise ValueError(f"User {user_id} not found")

    plan            = doc.get("plan", "free")
    polar_cid       = doc.get("polar_customer_id")
    portal_url      = await get_customer_portal_url(polar_cid) if polar_cid else None

    entitlements = [f for f in ("api_key", "deploy", "priority")
                    if check_feature_allowed(plan, f)]

    return CustomerInfo(
        user_id=user_id,
        plan=plan,
        polar_id=polar_cid,
        is_active=doc.get("is_active", True),
        billing_flagged=doc.get("billing_flagged", False),
        entitlements=entitlements,
        portal_url=portal_url,
    )
    
