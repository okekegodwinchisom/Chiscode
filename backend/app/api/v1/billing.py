"""
ChisCode — Billing API Router (Polar.sh)
==========================================
GET  /billing/usage              — daily usage + plan summary
GET  /billing/plans              — all plans with Polar checkout URLs
GET  /billing/customer           — Polar customer info + portal URL
GET  /billing/checkout/{plan}    — create Polar checkout session → redirect
POST /billing/portal             — return customer portal URL
GET  /billing/entitlements       — feature flags for current user
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse

from app.api.deps import get_current_user
from app.core.logging import get_logger
from app.schemas.user import UserInDB
from app.services.billing_service import (
    PLAN_META,
    CustomerInfo,
    UsageSummary,
    check_feature_allowed,
    create_checkout_session,
    get_customer_info,
    get_product_id,
    get_usage_summary,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/billing", tags=["billing"])


@router.get("/usage", response_model=UsageSummary)
async def get_usage(current_user: UserInDB = Depends(get_current_user)):
    return await get_usage_summary(str(current_user.id), current_user.plan)


@router.get("/plans")
async def get_plans(current_user: UserInDB = Depends(get_current_user)):
    plans = []
    for plan_key, meta in PLAN_META.items():
        plans.append({
            "id":           plan_key,
            "name":         meta["name"],
            "price":        meta["price"],
            "daily_limit":  meta["daily_limit"],
            "features":     meta["features"],
            "missing":      meta["missing"],
            "checkout_url": f"/api/v1/billing/checkout/{plan_key}" if plan_key != "free" else None,
            "is_current":   current_user.plan == plan_key,
            "is_upgrade":   _is_upgrade(current_user.plan, plan_key),
        })
    return {"plans": plans, "current_plan": current_user.plan}


@router.get("/customer", response_model=CustomerInfo)
async def get_customer(current_user: UserInDB = Depends(get_current_user)):
    return await get_customer_info(str(current_user.id))


@router.get("/checkout/{plan}")
async def checkout_redirect(
    plan:    str,
    request: Request,
    current_user: UserInDB = Depends(get_current_user),
):
    """Create a Polar checkout session and redirect to it."""
    valid = ("basic", "pro", "yearly")
    if plan not in valid:
        raise HTTPException(status_code=400, detail=f"Invalid plan. Choose: {valid}")

    if not _is_upgrade(current_user.plan, plan):
        raise HTTPException(
            status_code=400,
            detail="Use the customer portal to manage your current subscription.",
        )

    product_id  = get_product_id(plan)
    if not product_id:
        raise HTTPException(status_code=503, detail="Plan not configured. Contact support.")

    success_url = str(request.base_url) + "dashboard?upgraded=1"
    checkout_url = await create_checkout_session(
        product_id=product_id,
        user_id=str(current_user.id),
        email=str(current_user.email),
        success_url=success_url,
    )
    if not checkout_url:
        raise HTTPException(status_code=503, detail="Could not create checkout session.")

    return RedirectResponse(url=checkout_url, status_code=302)


@router.post("/portal")
async def open_portal(current_user: UserInDB = Depends(get_current_user)):
    info = await get_customer_info(str(current_user.id))
    return {
        "url":     info.portal_url,
        "message": "Redirecting to subscription portal." if info.portal_url else "No active subscription.",
    }


@router.get("/entitlements")
async def get_entitlements(current_user: UserInDB = Depends(get_current_user)):
    plan = current_user.plan
    return {
        "plan": plan,
        "entitlements": {
            "api_key":   check_feature_allowed(plan, "api_key"),
            "deploy":    check_feature_allowed(plan, "deploy"),
            "priority":  check_feature_allowed(plan, "priority"),
            "rag":       True,
            "templates": True,
            "github":    True,
        },
    }


def _is_upgrade(current: str, target: str) -> bool:
    order = ["free", "basic", "pro", "yearly"]
    try:
        return order.index(target) > order.index(current)
    except ValueError:
        return False
  
