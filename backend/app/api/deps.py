"""
ChisCode — FastAPI Dependencies
Reusable dependency functions for auth, rate limiting, and database access.
"""
from datetime import date, datetime, timedelta
from typing import Optional, List, Union, Dict, Tuple, Callable

from fastapi import Cookie, Depends, Header, HTTPException, Request, status, WebSocket
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

from app.core.config import settings
from app.core.logging import get_logger
from app.core.security import decode_token
from app.db import redis_client
from app.schemas.user import UserInDB
from app.services import user_service

logger = get_logger(__name__)

# Security scheme for OpenAPI docs
security = HTTPBearer(auto_error=False)


# ── Auth Dependencies ─────────────────────────────────────────

async def get_token_from_request(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    access_token: Optional[str] = Cookie(default=None),
    authorization: Optional[str] = Header(default=None),
) -> Optional[str]:
    """
    Extract JWT token from various sources:
    - Cookie (access_token)
    - Authorization header (Bearer token)
    - HTTPBearer security scheme
    
    Priority: HTTPBearer > Cookie > Header
    """
    # Check HTTPBearer (for OpenAPI docs)
    if credentials:
        return credentials.credentials
    
    # Check cookie
    if access_token:
        return access_token
    
    # Check Authorization header manually
    if authorization:
        scheme, _, token_value = authorization.partition(" ")
        if scheme.lower() == "bearer" and token_value:
            return token_value
    
    return None


async def get_current_user_from_jwt(
    token_str: Optional[str] = Depends(get_token_from_request),
) -> UserInDB:
    """
    Resolve the current user from a JWT token.
    Checks the token blacklist (logged-out tokens).
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not token_str:
        raise credentials_exception

    try:
        payload = decode_token(token_str)
        user_id: str = payload.get("sub", "")
        jti: str = payload.get("jti", "")
        token_type: str = payload.get("type", "access")
        
        if not user_id or not jti:
            raise credentials_exception
            
        # Ensure this is an access token, not refresh token
        if token_type != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type. Use access token.",
            )
            
    except JWTError as e:
        logger.warning(f"JWT validation failed: {str(e)}")
        raise credentials_exception
    except Exception as e:
        logger.error(f"Token decode error: {str(e)}")
        raise credentials_exception

    # Check token blacklist
    try:
        is_blacklisted = await redis_client.is_token_blacklisted(jti)
        if is_blacklisted:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked. Please log in again.",
            )
    except Exception as e:
        logger.warning(f"Failed to check token blacklist: {str(e)}")
        # Don't fail auth if Redis is down - just log and continue

    try:
        user = await user_service.get_user_by_id(user_id)
        if user is None:
            raise credentials_exception
    except Exception as e:
        logger.error(f"Failed to get user: {str(e)}")
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated.",
        )

    return user


async def get_current_user_from_api_key(
    x_chiscode_api_key: Optional[str] = Header(default=None, alias="X-ChisCode-API-Key"),
) -> Optional[UserInDB]:
    """Resolve user from an API key header (Pro/Yearly plans only)."""
    if not x_chiscode_api_key:
        return None
    
    try:
        user = await user_service.get_user_by_api_key(x_chiscode_api_key)
        
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key.",
            )
        
        if user.plan not in ("pro", "yearly"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="API key access requires Pro or Yearly plan.",
            )
        
        # Update last used timestamp (optional)
        try:
            await redis_client.update_api_key_last_used(str(user.id))
        except Exception as e:
            logger.warning(f"Failed to update API key timestamp: {e}")
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API key validation error: {str(e)}")
        return None


async def get_current_user(
    jwt_user: Optional[UserInDB] = Depends(get_current_user_from_jwt),
    api_key_user: Optional[UserInDB] = Depends(get_current_user_from_api_key),
) -> UserInDB:
    """
    Master auth dependency — accepts either JWT cookie/header OR API key.
    Use this on all protected endpoints.
    """
    user = api_key_user or jwt_user
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def get_optional_user(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    access_token: Optional[str] = Cookie(default=None),
) -> Optional[UserInDB]:
    """
    Optional auth dependency - returns None if not authenticated.
    Useful for endpoints that work with or without auth (e.g., public project viewing).
    """
    try:
        token = await get_token_from_request(None, access_token, authorization)
        if not token:
            return None
        
        payload = decode_token(token)
        user_id = payload.get("sub", "")
        if not user_id:
            return None
        
        user = await user_service.get_user_by_id(user_id)
        return user
    except Exception as e:
        logger.debug(f"Optional auth failed: {str(e)}")
        return None


# ── WebSocket Auth ────────────────────────────────────────────

async def get_current_user_ws(
    websocket: WebSocket,
) -> Optional[UserInDB]:
    """
    Authenticate user for WebSocket connections.
    Token can come from query parameter or cookie.
    """
    token = None
    
    # Try query parameter first
    token = websocket.query_params.get("token")
    
    # Try cookie if no query param
    if not token:
        try:
            cookie_header = websocket.headers.get("cookie", "")
            cookies: Dict[str, str] = {}
            for cookie in cookie_header.split(";"):
                if "=" in cookie:
                    key, value = cookie.strip().split("=", 1)
                    cookies[key] = value
            token = cookies.get("access_token")
        except Exception as e:
            logger.debug(f"Failed to parse cookies: {e}")
    
    if not token:
        return None
    
    try:
        payload = decode_token(token)
        user_id = payload.get("sub", "")
        if not user_id:
            return None
        
        user = await user_service.get_user_by_id(user_id)
        
        if user is None or not user.is_active:
            return None
        
        return user
    except Exception as e:
        logger.debug(f"WebSocket auth failed: {str(e)}")
        return None


# ── Rate Limiting ─────────────────────────────────────────────

async def check_rate_limit(
    request: Request,
    current_user: UserInDB = Depends(get_current_user),
) -> UserInDB:
    """
    Check and increment the user's daily request counter.
    Raises 429 if the limit is exceeded.
    """
    today = date.today().isoformat()
    daily_limit = settings.get_rate_limit(current_user.plan)

    try:
        result = await redis_client.check_and_increment_rate_limit(
            user_id=str(current_user.id),
            daily_limit=daily_limit,
            date_str=today,
        )
        
        # Handle different return formats (tuple or dict)
        if isinstance(result, tuple):
            allowed, count, limit = result
        elif isinstance(result, dict):
            allowed = result.get("allowed", True)
            count = result.get("count", 0)
            limit = result.get("limit", daily_limit)
        else:
            # Fallback if Redis is down
            allowed = True
            count = 0
            limit = daily_limit
            logger.warning("Rate limit check returned unexpected format")
            
    except Exception as e:
        # If Redis is down, allow the request but log warning
        logger.warning(f"Rate limit check failed: {e}")
        allowed = True
        count = 0
        limit = daily_limit

    if not allowed:
        plan_display = current_user.plan.capitalize()
        
        # Calculate reset time (midnight UTC next day)
        now = datetime.utcnow()
        tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        seconds_until_reset = int((tomorrow - now).total_seconds())
        
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Daily limit of {limit} requests reached for {plan_display} plan. "
                "Resets at midnight UTC."
            ),
            headers={
                "Retry-After": str(seconds_until_reset),
                "X-RateLimit-Limit": str(limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(tomorrow.timestamp())),
                "X-RateLimit-Used": str(count)
            },
        )

    # Add rate limit info to request state for middleware
    remaining = max(0, limit - count)
    tomorrow = (datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) + 
                timedelta(days=1))
    
    request.state.rate_limit = {
        "limit": limit,
        "remaining": remaining,
        "used": count,
        "reset": int(tomorrow.timestamp())
    }

    logger.debug(
        f"Rate limit check passed - user_id: {str(current_user.id)}, "
        f"plan: {current_user.plan}, count: {count}, limit: {limit}"
    )
    
    return current_user


async def check_rate_limit_optional(
    request: Request,
    current_user: Optional[UserInDB] = Depends(get_optional_user),
) -> Optional[UserInDB]:
    """
    Optional rate limiting - only applies if user is authenticated.
    Used for endpoints that work with or without auth.
    """
    if current_user is None:
        return None
    
    return await check_rate_limit(request, current_user)


# ── Plan Guards ───────────────────────────────────────────────

def require_plan(*plans: str) -> Callable:
    """Dependency factory — require the user to be on one of the given plans."""
    async def _check(current_user: UserInDB = Depends(get_current_user)) -> UserInDB:
        if current_user.plan not in plans:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This feature requires one of these plans: {', '.join(plans)}.",
            )
        return current_user
    return _check


def require_feature(feature: str) -> Callable:
    """Check if user's plan has access to a specific feature."""
    async def _check(current_user: UserInDB = Depends(get_current_user)) -> UserInDB:
        try:
            plan_features = settings.get_plan_features(current_user.plan)
            
            if feature not in plan_features:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Your {current_user.plan} plan does not include {feature}. "
                           f"Please upgrade to access this feature.",
                )
        except AttributeError:
            # If get_plan_features doesn't exist, just check plan
            if current_user.plan not in ("pro", "yearly"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"This feature requires Pro or Yearly plan.",
                )
        
        return current_user
    return _check


# ── Admin Guard ────────────────────────────────────────────────

async def require_admin(
    current_user: UserInDB = Depends(get_current_user)
) -> UserInDB:
    """Require user to be an admin."""
    if not getattr(current_user, 'is_admin', False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required."
        )
    return current_user


# ── Database Session ──────────────────────────────────────────

async def get_db_session():
    """
    Get a database session for transactions.
    For MongoDB, this would use client sessions.
    """
    # MongoDB transaction support
    # from app.db.mongodb import get_client
    # async with await get_client().start_session() as session:
    #     async with session.start_transaction():
    #         yield session
    
    yield None  # Placeholder for now


# ─── Request Helpers ──────────────────────────────────────────

async def get_client_ip(request: Request) -> str:
    """Extract client IP from request headers."""
    # Check X-Forwarded-For first (common in proxies/load balancers)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    
    # Check X-Real-IP
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    
    # Fallback to direct client
    return request.client.host if request.client else "unknown"


async def get_user_agent(request: Request) -> str:
    """Extract user agent from request headers."""
    return request.headers.get("User-Agent", "unknown")


def get_request_id(request: Request) -> str:
    """Get or generate request ID for tracing."""
    # Check if middleware added it
    if hasattr(request.state, "request_id"):
        return request.state.request_id
    
    # Check header
    request_id = request.headers.get("X-Request-ID")
    if request_id:
        return request_id
    
    # Generate one
    import uuid
    return str(uuid.uuid4())
