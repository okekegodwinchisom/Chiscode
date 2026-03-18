"""
ChisCode — User Routes
Profile, usage, and API key management endpoints.
"""
from datetime import date, datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query

from app.api.deps import get_current_user, require_plan
from app.core.config import settings
from app.db import redis_client
from app.schemas.user import (
    ApiKeyResponse, 
    UsageResponse, 
    UserPublic,
    UserUpdate,
    UserDeleteResponse
)
from app.services import user_service
from app.schemas import PyObjectId
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me", response_model=UserPublic, response_model_by_alias=True)
async def get_profile(current_user=Depends(get_current_user)):
    """Return the currently authenticated user's profile."""
    return UserPublic.model_validate(current_user.model_dump(by_alias=True))


@router.get("/me/usage", response_model=UsageResponse, response_model_by_alias=True)
async def get_usage(current_user=Depends(get_current_user)):
    """Return today's request usage and plan limits."""
    today = date.today()
    used = await redis_client.get_current_usage(str(current_user.id), today.isoformat())
    daily_limit = settings.get_rate_limit(current_user.plan)

    tomorrow = datetime.combine(today + timedelta(days=1), datetime.min.time())

    return UsageResponse(
        plan=current_user.plan,
        daily_limit=daily_limit,
        used_today=used,
        remaining=max(0, daily_limit - used),
        resets_at=tomorrow.isoformat() + "Z",
    )


@router.patch("/me", response_model=UserPublic, response_model_by_alias=True)
async def update_profile(
    update_data: UserUpdate,
    current_user=Depends(get_current_user)
):
    """Update user profile information (name, avatar, etc.)."""
    try:
        updated_user = await user_service.update_user(str(current_user.id), update_data)
        logger.info("User profile updated", user_id=str(current_user.id))
        return UserPublic.model_validate(updated_user.model_dump(by_alias=True))
    except user_service.UserNotFoundError:
        raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        logger.error("Failed to update user profile", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update profile")


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
async def delete_account(current_user=Depends(get_current_user)):
    """Delete user account and all associated data."""
    try:
        await user_service.delete_user(str(current_user.id))
        # Also clear any sessions/blacklist tokens
        logger.info("User account deleted", user_id=str(current_user.id))
        return None
    except Exception as e:
        logger.error("Failed to delete user account", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to delete account")


@router.post("/me/api-key", response_model=ApiKeyResponse)
async def generate_api_key(
    current_user=Depends(require_plan("pro", "yearly")),
):
    """
    Generate a new API key (Pro/Yearly plans only).
    The raw key is returned once — store it securely.
    """
    try:
        raw_key = await user_service.generate_user_api_key(str(current_user.id))
        logger.info("API key generated", user_id=str(current_user.id))
        return ApiKeyResponse(api_key=raw_key)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error("Failed to generate API key", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to generate API key")


@router.delete("/me/api-key", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    current_user=Depends(require_plan("pro", "yearly")),
):
    """Revoke the current API key."""
    try:
        await user_service.revoke_user_api_key(str(current_user.id))
        logger.info("API key revoked", user_id=str(current_user.id))
        return None
    except Exception as e:
        logger.error("Failed to revoke API key", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to revoke API key")


@router.get("/me/api-key/status")
async def get_api_key_status(
    current_user=Depends(require_plan("pro", "yearly")),
):
    """Check if user has an active API key."""
    return {
        "has_api_key": current_user.api_key_hash is not None,
        "plan": current_user.plan
    }


@router.get("/me/activity", response_model=dict)
async def get_recent_activity(
    limit: int = Query(10, ge=1, le=50),
    current_user=Depends(get_current_user)
):
    """Get recent user activity (logins, project creations, etc.)."""
    try:
        activity = await user_service.get_recent_activity(
            str(current_user.id),
            limit=limit
        )
        return {"activities": activity}
    except Exception as e:
        logger.error("Failed to fetch activity", error=str(e))
        return {"activities": []}


@router.get("/me/stats", response_model=dict)
async def get_user_stats(current_user=Depends(get_current_user)):
    """Get user statistics (project count, total generations, etc.)."""
    try:
        stats = await user_service.get_user_stats(str(current_user.id))
        return stats
    except Exception as e:
        logger.error("Failed to fetch user stats", error=str(e))
        return {
            "project_count": 0,
            "total_generations": 0,
            "success_rate": 0
        }


@router.post("/me/logout-all", status_code=status.HTTP_200_OK)
async def logout_all_devices(current_user=Depends(get_current_user)):
    """Logout from all devices by invalidating all refresh tokens."""
    try:
        # This would blacklist all tokens for the user
        await redis_client.blacklist_all_user_tokens(str(current_user.id))
        logger.info("User logged out from all devices", user_id=str(current_user.id))
        return {"message": "Logged out from all devices successfully"}
    except Exception as e:
        logger.error("Failed to logout all devices", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to logout all devices")
