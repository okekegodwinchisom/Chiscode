"""
ChisCode — API Router
Aggregates all versioned sub-routers into a single api_router.

Phase history:
  Phase 0–1  users, auth, webhooks
  Phase 2–4  projects, mcp_server
  Phase 5    templates
  Phase 6    deploy (preview + deployment)
  Phase 7    billing (usage, plans, checkout, portal)
"""
from fastapi import APIRouter

from app.api.v1.auth      import router as auth_router
from app.api.v1.projects  import router as projects_router
from app.api.v1.users     import router as users_router
from app.api.v1.webhooks  import router as webhooks_router
from app.api.mcp_server   import router as mcp_router
from app.api.v1.templates import router as templates_router
from app.api.v1.deploy    import router as deploy_router
from app.api.v1.billing   import router as billing_router

api_router = APIRouter()

api_router.include_router(auth_router)
api_router.include_router(users_router)
api_router.include_router(projects_router)
api_router.include_router(webhooks_router)
api_router.include_router(mcp_router)
api_router.include_router(templates_router)
api_router.include_router(deploy_router)
api_router.include_router(billing_router)
