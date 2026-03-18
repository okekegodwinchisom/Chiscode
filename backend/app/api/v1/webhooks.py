"""
ChisCode — Webhooks Router (Polar.sh)
======================================
Handles Polar.sh subscription lifecycle webhooks.
Replace backend/app/api/v1/webhooks.py with this file.

Polar webhook docs: https://docs.polar.sh/features/webhooks
Signature header:   webhook-id, webhook-timestamp, webhook-signature
Verification:       standardwebhooks spec (HMAC-SHA256)

Event → Action map:
  order.created              → one-time purchase (not subscription)
  subscription.created       → new subscription, set plan
  subscription.updated       → plan change (upgrade/downgrade)
  subscription.active        → subscription confirmed active
  subscription.canceled      → mark cancel_at_period_end, keep plan live
  subscription.revoked       → immediately downgrade to free (payment failed)
"""
import hashlib
import hmac
import json
import base64
import time

from fastapi import APIRouter, Header, HTTPException, Request, status

from app.core.config import settings
from app.core.logging import get_logger
from app.db.mongodb import users_collection
from app.services import user_service
from app.services.billing_service import (
    _product_plan_map,
    clear_billing_flag,
    flag_billing_issue,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/webhooks", tags=["webhooks"])


# ── Polar standard webhooks signature verification ─────────────
# Spec: https://www.standardwebhooks.com/

def _verify_polar_signature(
    body:      bytes,
    msg_id:    str | None,
    timestamp: str | None,
    signature: str | None,
) -> bool:
    """
    Verify Polar webhook using the Standard Webhooks spec.
    Signed message = "{webhook-id}.{webhook-timestamp}.{body}"
    """
    secret = settings.polar_webhook_secret
    if not secret:
        return not settings.is_production

    if not msg_id or not timestamp or not signature:
        return False

    # Reject timestamps older than 5 minutes (replay protection)
    try:
        ts = int(timestamp)
        if abs(time.time() - ts) > 300:
            return False
    except (ValueError, TypeError):
        return False

    # Strip "v1," prefix that Polar adds
    sig_value = signature.replace("v1,", "")

    # Decode base64 secret (Polar uses base64-encoded secret)
    try:
        secret_bytes = base64.b64decode(secret)
    except Exception:
        secret_bytes = secret.encode()

    signed_content = f"{msg_id}.{timestamp}.".encode() + body
    expected = base64.b64encode(
        hmac.new(secret_bytes, signed_content, hashlib.sha256).digest()
    ).decode()

    return hmac.compare_digest(expected, sig_value)


@router.post("/polar", status_code=status.HTTP_200_OK)
async def polar_webhook(
    request: Request,
    webhook_id:        str | None = Header(default=None, alias="webhook-id"),
    webhook_timestamp: str | None = Header(default=None, alias="webhook-timestamp"),
    webhook_signature: str | None = Header(default=None, alias="webhook-signature"),
):
    """
    Handle Polar subscription lifecycle events.
    Configure in Polar Dashboard → Settings → Webhooks
    URL: https://your-space.hf.space/api/v1/webhooks/polar
    """
    body = await request.body()

    if not _verify_polar_signature(body, webhook_id, webhook_timestamp, webhook_signature):
        logger.warning("Polar webhook: invalid signature")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook signature.",
        )

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON.")

    event_type = payload.get("type", "")
    data       = payload.get("data", {})

    logger.info("Polar webhook received", event_type=event_type)

    try:
        if event_type == "subscription.created":
            await _handle_subscription_created(data)

        elif event_type == "subscription.updated":
            await _handle_subscription_updated(data)

        elif event_type == "subscription.active":
            await _handle_subscription_active(data)

        elif event_type == "subscription.canceled":
            await _handle_subscription_canceled(data)

        elif event_type == "subscription.revoked":
            await _handle_subscription_revoked(data)

        elif event_type == "order.created":
            # One-time purchase (not recurring) — handle if needed
            logger.info("Polar order created", order_id=data.get("id"))

        else:
            logger.debug("Unhandled Polar event", event_type=event_type)

    except user_service.UserNotFoundError:
        logger.error("Polar webhook: user not found",
                     metadata=data.get("metadata", {}))
        return {"status": "ok", "note": "user not found"}
    except Exception as exc:
        logger.error("Polar webhook error", error=str(exc), event_type=event_type)
        raise HTTPException(status_code=500, detail="Webhook processing failed.")

    return {"status": "ok"}


# ── Event handlers ─────────────────────────────────────────────

def _get_user_id(data: dict) -> str:
    """
    Extract ChisCode user ID from Polar event metadata.
    We store chiscode_user_id in checkout session metadata.
    Polar propagates this to subscription/order objects.
    """
    meta = data.get("metadata") or data.get("customer_metadata") or {}
    uid  = meta.get("chiscode_user_id", "")
    if not uid:
        # Fallback: match by customer email
        raise user_service.UserNotFoundError("No chiscode_user_id in metadata")
    return uid


def _get_plan(data: dict) -> str:
    product_id  = (data.get("product") or {}).get("id", "")
    plan_map    = _product_plan_map()
    return plan_map.get(product_id, "basic")


async def _handle_subscription_created(data: dict) -> None:
    user_id   = _get_user_id(data)
    plan      = _get_plan(data)
    polar_cid = (data.get("customer") or {}).get("id", "")
    sub_id    = data.get("id", "")

    await user_service.update_user_plan(user_id, plan)
    await clear_billing_flag(user_id)

    # Store Polar customer + subscription IDs
    from datetime import datetime, timezone
    await users_collection().update_one(
        {"_id": __import__("bson").ObjectId(user_id)},
        {"$set": {
            "polar_customer_id":     polar_cid,
            "polar_subscription_id": sub_id,
            "updated_at":            datetime.now(tz=timezone.utc),
        }},
    )
    logger.info("Subscription created", user_id=user_id, plan=plan, sub_id=sub_id)


async def _handle_subscription_updated(data: dict) -> None:
    user_id = _get_user_id(data)
    plan    = _get_plan(data)
    status_ = data.get("status", "")

    if status_ in ("active", "trialing"):
        await user_service.update_user_plan(user_id, plan)
        await clear_billing_flag(user_id)
        logger.info("Subscription updated", user_id=user_id, plan=plan)
    elif status_ == "past_due":
        await flag_billing_issue(user_id)
        logger.warning("Subscription past due", user_id=user_id)


async def _handle_subscription_active(data: dict) -> None:
    user_id = _get_user_id(data)
    plan    = _get_plan(data)
    await user_service.update_user_plan(user_id, plan)
    await clear_billing_flag(user_id)
    logger.info("Subscription confirmed active", user_id=user_id, plan=plan)


async def _handle_subscription_canceled(data: dict) -> None:
    """
    Polar keeps the subscription active until the period ends (cancel_at_period_end).
    We don't downgrade yet — that happens on subscription.revoked.
    """
    user_id = _get_user_id(data)
    logger.info("Subscription canceled (active until period end)", user_id=user_id)

    from datetime import datetime, timezone
    await users_collection().update_one(
        {"_id": __import__("bson").ObjectId(user_id)},
        {"$set": {
            "subscription_cancel_at_period_end": True,
            "updated_at": datetime.now(tz=timezone.utc),
        }},
    )


async def _handle_subscription_revoked(data: dict) -> None:
    """
    Period ended or payment definitively failed — downgrade to free immediately.
    """
    user_id = _get_user_id(data)
    await user_service.update_user_plan(user_id, "free")

    from datetime import datetime, timezone
    await users_collection().update_one(
        {"_id": __import__("bson").ObjectId(user_id)},
        {"$unset": {
            "polar_subscription_id":          "",
            "subscription_cancel_at_period_end": "",
        },
         "$set": {"updated_at": datetime.now(tz=timezone.utc)}},
    )
    logger.info("Subscription revoked — downgraded to free", user_id=user_id)
    
