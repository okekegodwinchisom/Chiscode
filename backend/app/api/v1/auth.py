"""
ChisCode — Auth Routes
Handles email/password auth, GitHub OAuth, JWT refresh, and logout.
"""
from datetime import timezone
from urllib.parse import urlencode
import jwt  # PyJWT package
import time
from pathlib import Path
import os
import httpx
from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse
from jose import JWTError

from app.core.config import settings
from app.core.logging import get_logger
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decrypt_value,
    decode_token,
    encrypt_value,
)
from app.db import redis_client
from app.schemas.user import (
    TokenResponse,
    UserLoginRequest,
    UserPublic,
    UserRegisterRequest,
)
from app.services import user_service

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])

GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"


# ── Helpers ───────────────────────────────────────────────────

def _set_auth_cookies(response: Response, access_token: str, refresh_token: str) -> None:
    """Set HTTP-only secure auth cookies."""
    is_prod = settings.is_production
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=is_prod,
        samesite="none",
        max_age=settings.jwt_access_token_expire_minutes * 60,
        path="/",
    )
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=is_prod,
        samesite="none",
        max_age=settings.jwt_refresh_token_expire_days * 86400,
        path="/auth/refresh",
    )


def _build_token_response(user, access_token: str, refresh_token: str) -> TokenResponse:
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.jwt_access_token_expire_minutes * 60,
        user=UserPublic.model_validate(user.model_dump(by_alias=True)),
    )


# ── Email / Password ──────────────────────────────────────────

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(req: UserRegisterRequest, response: Response):
    """Register a new account with email and password."""
    try:
        user = await user_service.create_user(req)
    except user_service.UserAlreadyExistsError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))

    access_token = create_access_token(str(user.id), {"plan": user.plan})
    refresh_token = create_refresh_token(str(user.id))
    _set_auth_cookies(response, access_token, refresh_token)

    logger.info("New user registered", user_id=str(user.id))
    return _build_token_response(user, access_token, refresh_token)


@router.post("/login", response_model=TokenResponse)
async def login(req: UserLoginRequest, response: Response):
    """Log in with email and password, receive JWT tokens."""
    try:
        user = await user_service.authenticate_user(req.email, req.password)
    except user_service.InvalidCredentialsError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))

    access_token = create_access_token(str(user.id), {"plan": user.plan})
    refresh_token = create_refresh_token(str(user.id))
    _set_auth_cookies(response, access_token, refresh_token)

    logger.info("User logged in", user_id=str(user.id))
    return _build_token_response(user, access_token, refresh_token)


@router.post("/logout")
async def logout(
    response: Response,
    access_token: str | None = Cookie(default=None),
):
    """Invalidate the current access token and clear auth cookies."""
    if access_token:
        try:
            payload = decode_token(access_token)
            jti = payload.get("jti", "")
            exp = payload.get("exp", 0)
            import time
            ttl = max(0, int(exp - time.time()))
            if jti and ttl > 0:
                await redis_client.blacklist_token(jti, ttl)
        except JWTError:
            pass  # Token already invalid — still clear cookies

    response.delete_cookie("access_token", path="/")
    response.delete_cookie("refresh_token", path="/auth/refresh")
    return {"message": "Logged out successfully."}


@router.post("/refresh")
async def refresh_token(
    response: Response,
    refresh_token: str | None = Cookie(default=None),
):
    """Exchange a refresh token for a new access token."""
    if not refresh_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token missing.")

    try:
        payload = decode_token(refresh_token)
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type.")
        user_id = payload["sub"]
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired refresh token.")

    user = await user_service.get_user_by_id(user_id)
    access_token = create_access_token(str(user.id), {"plan": user.plan})

    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=settings.is_production,
        samesite="lax",
        max_age=settings.jwt_access_token_expire_minutes * 60,
    )
    return {"access_token": access_token, "token_type": "bearer"}


# ── GitHub OAuth ──────────────────────────────────────────────

@router.get("/github")
async def github_login(request: Request):
    """Redirect the user to GitHub's authorization page."""
    params = {
        "client_id": settings.github_client_id,
        "redirect_uri": settings.github_redirect_uri,
        "scope": "read:user user:email repo",
        "state": "chiscode_oauth",   # TODO: use PKCE / random state in production
    }
    url = f"{GITHUB_AUTHORIZE_URL}?{urlencode(params)}"
    return RedirectResponse(url)


@router.get("/github/callback")
async def github_callback(code: str, state: str, response: Response):
    """
    GitHub OAuth callback.
    Exchange the authorization code for an access token, then upsert the user.
    """
    # DEBUG: Log the incoming request
    logger.info("="*50)
    logger.info("GITHUB CALLBACK RECEIVED")
    logger.info(f"Code (first 10 chars): {code[:10] if code else 'None'}")
    logger.info(f"State: {state}")
    logger.info(f"Client ID configured: {settings.github_client_id[:5] if settings.github_client_id else 'Not set'}...")
    logger.info(f"Client Secret configured: {'Yes' if settings.github_client_secret else 'No'}")
    logger.info(f"Redirect URI: {settings.github_redirect_uri}")
    logger.info("="*50)
    async with httpx.AsyncClient() as client:
        # Exchange code for token
        token_resp = await client.post(
            GITHUB_TOKEN_URL,
            json={
                "client_id": settings.github_client_id,
                "client_secret": settings.github_client_secret,
                "code": code,
                "redirect_uri": settings.github_redirect_uri,
            },
            headers={"Accept": "application/json"},
        )
        token_data = token_resp.json()
        github_access_token = token_data.get("access_token")

        if not github_access_token:
            logger.error("GitHub OAuth token exchange failed", response=token_data)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="GitHub OAuth failed. Please try again.",
            )

        # Fetch user profile
        user_resp = await client.get(
            GITHUB_USER_URL,
            headers={"Authorization": f"Bearer {github_access_token}"},
        )
        gh_user = user_resp.json()

    # Encrypt the GitHub token before storing
    encrypted_token = encrypt_value(github_access_token)

    user = await user_service.upsert_github_user(
        github_id=str(gh_user["id"]),
        github_username=gh_user["login"],
        email=gh_user.get("email") or f"{gh_user['login']}@github.noreply",
        avatar_url=gh_user.get("avatar_url", ""),
        encrypted_token=encrypted_token,
    )

    access_token = create_access_token(str(user.id), {"plan": user.plan})
    refresh_token_val = create_refresh_token(str(user.id))

    redirect = RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    _set_auth_cookies(redirect, access_token, refresh_token_val)

    logger.info("GitHub OAuth login", user_id=str(user.id), username=gh_user["login"])
    return redirect

@router.get("/github")
async def github_login(request: Request):
    """Redirect the user to GitHub's authorization page."""
    if not settings.github_client_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GitHub OAuth is not configured.",
        )

    params = {
        "client_id":    settings.github_client_id,
        "redirect_uri": settings.github_redirect_uri,
        "scope":        "read:user user:email",
        "state":        "chiscode_oauth",
    }
    url = f"{GITHUB_AUTHORIZE_URL}?{urlencode(params)}"

    logger.info("Redirecting to GitHub OAuth", redirect_uri=settings.github_redirect_uri)

    return RedirectResponse(url, status_code=302)    
