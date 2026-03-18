"""
ChisCode — Test Suite
Foundation tests: health checks, auth endpoints, rate limiting.
"""
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.main import app


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest_asyncio.fixture(scope="session")
async def client():
    """Async test client for the FastAPI app."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


# ── Health Checks ─────────────────────────────────────────────

@pytest.mark.anyio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["app"] == settings.app_name


# ── Auth: Registration ─────────────────────────────────────────

@pytest.mark.anyio
async def test_register_success(client):
    resp = await client.post("/api/v1/auth/register", json={
        "email": "test@chiscode.dev",
        "username": "testuser",
        "password": "Test1234!",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert "access_token" in data
    assert data["user"]["email"] == "test@chiscode.dev"
    assert data["user"]["plan"] == "free"


@pytest.mark.anyio
async def test_register_duplicate_email(client):
    payload = {"email": "dupe@chiscode.dev", "username": "dupe1", "password": "Test1234!"}
    await client.post("/api/v1/auth/register", json=payload)
    resp = await client.post("/api/v1/auth/register", json={**payload, "username": "dupe2"})
    assert resp.status_code == 409


@pytest.mark.anyio
async def test_register_weak_password(client):
    resp = await client.post("/api/v1/auth/register", json={
        "email": "weak@chiscode.dev",
        "username": "weakpass",
        "password": "tooshort",  # No uppercase, no digit
    })
    assert resp.status_code == 422


# ── Auth: Login ────────────────────────────────────────────────

@pytest.mark.anyio
async def test_login_success(client):
    # Register first
    await client.post("/api/v1/auth/register", json={
        "email": "login@chiscode.dev",
        "username": "loginuser",
        "password": "Login1234!",
    })
    # Login
    resp = await client.post("/api/v1/auth/login", json={
        "email": "login@chiscode.dev",
        "password": "Login1234!",
    })
    assert resp.status_code == 200
    assert "access_token" in resp.json()


@pytest.mark.anyio
async def test_login_wrong_password(client):
    resp = await client.post("/api/v1/auth/login", json={
        "email": "login@chiscode.dev",
        "password": "wrongpassword",
    })
    assert resp.status_code == 401


# ── Auth: Protected endpoints ──────────────────────────────────

@pytest.mark.anyio
async def test_get_profile_unauthorized(client):
    resp = await client.get("/api/v1/users/me")
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_get_profile_authorized(client):
    # Register + login
    reg = await client.post("/api/v1/auth/register", json={
        "email": "profile@chiscode.dev",
        "username": "profileuser",
        "password": "Profile1!",
    })
    token = reg.json()["access_token"]

    resp = await client.get(
        "/api/v1/users/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["email"] == "profile@chiscode.dev"


# ── Security ──────────────────────────────────────────────────

@pytest.mark.anyio
async def test_invalid_jwt(client):
    resp = await client.get(
        "/api/v1/users/me",
        headers={"Authorization": "Bearer invalid.jwt.token"},
    )
    assert resp.status_code == 401


# ── Projects ──────────────────────────────────────────────────

@pytest.mark.anyio
async def test_list_projects_authenticated(client):
    reg = await client.post("/api/v1/auth/register", json={
        "email": "projects@chiscode.dev",
        "username": "projectsuser",
        "password": "Projects1!",
    })
    token = reg.json()["access_token"]

    resp = await client.get(
        "/api/v1/projects/",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.anyio
async def test_generate_project_authenticated(client):
    reg = await client.post("/api/v1/auth/register", json={
        "email": "generate@chiscode.dev",
        "username": "generateuser",
        "password": "Generate1!",
    })
    token = reg.json()["access_token"]

    resp = await client.post(
        "/api/v1/projects/generate",
        json={"prompt": "Build a simple todo list app with FastAPI and MongoDB"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 202
    data = resp.json()
    assert "project_id" in data
    assert "ws_url" in data
