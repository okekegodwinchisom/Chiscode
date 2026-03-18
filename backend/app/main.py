"""
ChisCode — FastAPI Application
Main app factory with lifespan management, middleware, and routing.

Updated for Phase 5 (RAG/Pinecone, Quality Pipeline, Templates)
         and Phase 6 (Preview, Deployment).
"""
import time
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.router import api_router
from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.db import mongodb, redis_client

setup_logging()
logger = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ChisCode starting up", env=settings.app_env, version=settings.app_version)

    # MongoDB
    try:
        await mongodb.connect()
    except Exception as e:
        logger.error("MongoDB connection failed — running degraded", error=str(e))

    # Redis
    try:
        await redis_client.connect()
    except Exception as e:
        logger.error("Redis connection failed — running degraded", error=str(e))

    # ── Phase 5: Pinecone ─────────────────────────────────────────
    try:
        from app.db import pinecone_client
        await pinecone_client.connect()
    except Exception as e:
        logger.warning("Pinecone connection failed — RAG disabled", error=str(e))

    # ── Phase 5 + 6: MongoDB TTL indexes ─────────────────────────
    try:
        db = mongodb.get_db()

        # Templates: text search index on name + description + tags
        await db["templates"].create_index(
            [("name", "text"), ("description", "text"), ("tags", "text")],
            name="templates_text_search",
            background=True,
        )
        await db["templates"].create_index("app_type",  background=True)
        await db["templates"].create_index("is_active", background=True)
        await db["templates"].create_index("use_count", background=True)

        # Phase 6: previews TTL index (auto-expire after expires_at)
        await db["previews"].create_index(
            "expires_at",
            expireAfterSeconds=0,
            name="previews_ttl",
            background=True,
        )

        logger.info("MongoDB indexes ensured")
    except Exception as e:
        logger.warning("Index creation failed (non-fatal)", error=str(e))

    logger.info("All connections established. ChisCode is ready.")
    yield

    # ── Shutdown ──────────────────────────────────────────────────
    logger.info("ChisCode shutting down...")
    await mongodb.disconnect()
    await redis_client.disconnect()
    try:
        from app.db import pinecone_client
        await pinecone_client.disconnect()
    except Exception:
        pass
    logger.info("Shutdown complete.")


# ── App Factory ───────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="AI-powered agent builder — natural language to production-ready apps",
        docs_url="/docs"         if settings.debug else None,
        redoc_url="/redoc"       if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        lifespan=lifespan,
    )

    # ── Middleware ────────────────────────────────────────────────
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_base_url, "https://huggingface.co"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Project-Id"]
    )

    @app.middleware("http")
    async def add_request_timing(request: Request, call_next):
        start = time.perf_counter()
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            method=request.method, path=request.url.path
        )
        response     = await call_next(request)
        duration_ms  = round((time.perf_counter() - start) * 1000, 2)
        response.headers["X-Process-Time"]        = str(duration_ms)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"]        = "DENY"
        logger.info("Request handled",
                    status_code=response.status_code, duration_ms=duration_ms)
        return response

    # ── Static files & templates ──────────────────────────────────
    _here       = Path(__file__).resolve().parent
    _candidates = [
        _here.parent / "frontend",
        _here.parent.parent / "frontend",
        Path("/app/frontend"),
    ]
    _frontend = next((p for p in _candidates if p.is_dir()), None)

    templates = None
    if _frontend:
        logger.info("Frontend found", path=str(_frontend))
        _static = _frontend / "static"
        _tmpl   = _frontend / "templates"
        if _static.is_dir():
            app.mount("/static", StaticFiles(directory=str(_static)), name="static")
        if _tmpl.is_dir():
            templates = Jinja2Templates(directory=str(_tmpl))
    else:
        logger.warning("Frontend directory not found — HTML routes disabled")

    # ── API routes ────────────────────────────────────────────────
    app.include_router(api_router, prefix="/api/v1")

    # ── Health checks ─────────────────────────────────────────────
    @app.get("/health", tags=["system"])
    async def health_check():
        return {
            "status":  "ok",
            "app":     settings.app_name,
            "version": settings.app_version,
        }

    @app.get("/health/detailed", tags=["system"])
    async def detailed_health():
        checks: dict = {}

        try:
            await mongodb.get_client().admin.command("ping")
            checks["mongodb"] = "ok"
        except Exception as e:
            checks["mongodb"] = f"error: {e}"

        try:
            await redis_client.get_redis().ping()
            checks["redis"] = "ok"
        except Exception as e:
            checks["redis"] = f"error: {e}"

        # Phase 5: Pinecone availability
        try:
            from app.db.pinecone_client import is_available
            checks["pinecone"] = "ok" if is_available() else "disabled"
        except Exception as e:
            checks["pinecone"] = f"error: {e}"

        all_ok = all(v in ("ok", "disabled") for v in checks.values())
        return JSONResponse(
            status_code=status.HTTP_200_OK if all_ok else status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "ok" if all_ok else "degraded", "checks": checks},
        )

    # ── Frontend HTML routes ──────────────────────────────────────
    if templates:

        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        async def index(request: Request):
            return templates.TemplateResponse(
                "index.html", {"request": request, "settings": settings}
            )

        @app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
        async def dashboard(request: Request):
            return templates.TemplateResponse(
                "dashboard/index.html", {"request": request}
            )

        @app.get("/login", response_class=HTMLResponse, include_in_schema=False)
        async def login_page(request: Request):
            return templates.TemplateResponse("auth/login.html", {"request": request})

        @app.get("/register", response_class=HTMLResponse, include_in_schema=False)
        async def register_page(request: Request):
            return templates.TemplateResponse(
                "auth/register.html", {"request": request}
            )

        @app.get("/projects/{project_id}", response_class=HTMLResponse, include_in_schema=False)
        async def project_detail_page(request: Request, project_id: str):
            return templates.TemplateResponse(
                "projects/detail.html", {"request": request}
            )

        # ── Dashboard: All projects page ─────────────────────────
        @app.get("/dashboard/projects", response_class=HTMLResponse, include_in_schema=False)
        async def all_projects_page(request: Request):
            return templates.TemplateResponse(
                "dashboard/projects.html", {"request": request}
            )

        # ── Phase 5: Templates browser ────────────────────────────
        @app.get("/templates", response_class=HTMLResponse, include_in_schema=False)
        async def templates_page(request: Request):
            return templates.TemplateResponse(
                "templates/index.html", {"request": request}
            )

        # ── Phase 6: Deploy panel ─────────────────────────────────
        @app.get("/projects/{project_id}/deploy", response_class=HTMLResponse,
                 include_in_schema=False)
        async def deploy_page(request: Request, project_id: str):
            return templates.TemplateResponse(
                "projects/deploy.html",
                {"request": request, "project_id": project_id},
            )

        # ── Phase 7: Pricing / billing page ───────────────────
        @app.get("/pricing", response_class=HTMLResponse, include_in_schema=False)
        async def pricing_page(request: Request):
            return templates.TemplateResponse("pricing.html", {"request": request})

        @app.get("/favicon.ico", response_class=HTMLResponse, include_in_schema=False)
        async def favicon():
            return HTMLResponse("")  # Or serve an actual favicon

    # ── Exception handlers ────────────────────────────────────────
    @app.exception_handler(404)
    async def not_found(request: Request, exc):
        if request.url.path.startswith("/api/"):
            return JSONResponse(status_code=404, content={"detail": "Not found."})
        if templates:
            return templates.TemplateResponse(
                "404.html", {"request": request,"settings": settings}, status_code=404
            )
        return JSONResponse(status_code=404, content={"detail": "Not found."})

    @app.exception_handler(500)
    async def server_error(request: Request, exc):
        logger.error("Unhandled server error", exc_info=exc)
        return JSONResponse(
            status_code=500, content={"detail": "Internal server error."}
        )

    return app

    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc):
        logger.error(
            "Validation error",
            path=str(request.url.path),
            errors=str(exc.errors()),
        )
        return JSONResponse(status_code=422, content={"detail": exc.errors()})


app = create_app()
      
