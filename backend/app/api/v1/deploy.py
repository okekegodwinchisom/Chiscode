"""
ChisCode — Deploy & Preview API Router (Phase 6)
=================================================
POST /api/v1/projects/{id}/deploy          — start deployment (SSE stream)
POST /api/v1/projects/{id}/preview         — generate/refresh preview
GET  /api/v1/preview/{id}                  — serve live preview HTML
GET  /api/v1/projects/{id}/preview/card    — get preview card data
GET  /api/v1/projects/{id}/deploy/configs  — get all generated config files
"""
from __future__ import annotations

from typing import Optional
import json

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from app.api.deps import get_current_user
from app.core.logging import get_logger
from app.db.mongodb import get_db, projects_collection
from app.schemas.user import UserInDB
from app.services.deployment_service import DeployConfig, deploy_project
from app.services.preview_service import (
    generate_preview,
    get_preview_card,
    get_preview_html,
)

logger = get_logger(__name__)
router = APIRouter(tags=["deploy"])

# ── Deploy request ─────────────────────────────────────────────

class DeployRequest(BaseModel):
    platform:        str
    # Optional platform tokens (user provides these in UI)
    vercel_token:    Optional[str] = None
    netlify_token:   Optional[str] = None
    render_token:    Optional[str] = None
    cf_api_token:    Optional[str] = None
    cf_account_id:   Optional[str] = None


# ── Endpoints ──────────────────────────────────────────────────

@router.post("/projects/{project_id}/deploy")
async def deploy_endpoint(
    project_id:   str,
    req:          DeployRequest,
    current_user: UserInDB = Depends(get_current_user),
):
    """Stream deployment progress as SSE."""
    from bson import ObjectId

    doc = await projects_collection().find_one({
        "_id":     ObjectId(project_id),
        "user_id": current_user.id,
    })
    if not doc:
        raise HTTPException(status_code=404, detail="Project not found.")
    if doc.get("status") != "complete":
        raise HTTPException(status_code=400, detail="Project must be complete before deploying.")

    cfg = DeployConfig(
        platform=req.platform,
        project_name=doc.get("name", "chiscode-project"),
        project_id=project_id,
        user_id=str(current_user.id),
        stack=doc.get("stack", {}),
        file_tree=doc.get("file_tree", {}),
        vercel_token=req.vercel_token,
        netlify_token=req.netlify_token,
        render_token=req.render_token,
        cf_api_token=req.cf_api_token,
        cf_account_id=req.cf_account_id,
        github_token=doc.get("github_token_encrypted"),   # retrieved from user record
        github_username=current_user.github_username,
    )

    async def stream():
        # Track deploy in MongoDB
        await projects_collection().update_one(
            {"_id": ObjectId(project_id)},
            {"$push": {"deploy_log": {
                "platform": req.platform,
                "started_at": __import__("datetime").datetime.utcnow().isoformat(),
            }}},
        )

        config_files: dict[str, str] = {}

        async for event in deploy_project(cfg):
            # Save config files to project document
            if event.get("event") == "config_ready":
                config_files.update(event.get("config_files", {}))
                if config_files:
                    merged_tree = {**doc.get("file_tree", {}), **config_files}
                    await projects_collection().update_one(
                        {"_id": ObjectId(project_id)},
                        {"$set": {"file_tree": merged_tree}},
                    )

            # Save deploy URL
            if event.get("event") == "deploy_done" and event.get("url"):
                await projects_collection().update_one(
                    {"_id": ObjectId(project_id)},
                    {"$set": {f"deploy_urls.{req.platform}": event["url"]}},
                )

            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"X-Accel-Buffering": "no",
                                      "Cache-Control":     "no-cache"})


@router.post("/projects/{project_id}/preview")
async def create_preview(
    project_id:   str,
    current_user: UserInDB = Depends(get_current_user),
):
    """Generate or refresh preview for a project."""
    from bson import ObjectId

    doc = await projects_collection().find_one({
        "_id":     ObjectId(project_id),
        "user_id": current_user.id,
    })
    if not doc:
        raise HTTPException(status_code=404, detail="Project not found.")

    info = await generate_preview(
        project_id=project_id,
        file_tree=doc.get("file_tree", {}),
        stack=doc.get("stack", {}),
        project_name=doc.get("name", ""),
    )
    return info.model_dump()


@router.get("/preview/{project_id}", response_class=HTMLResponse)
async def serve_preview(project_id: str):
    """Serve the live preview HTML (no auth — token in URL not needed for iframe)."""
    html = await get_preview_html(project_id)
    if not html:
        return HTMLResponse(
            content="""<!DOCTYPE html>
<html><head><meta charset="UTF-8"></head>
<body style="background:#07090f;color:#6b7f95;font-family:monospace;display:flex;align-items:center;justify-content:center;height:100vh">
<div style="text-align:center"><p>Preview expired or not available.</p>
<p style="font-size:.8rem">Generate a new preview from the project page.</p></div>
</body></html>""",
            status_code=404,
        )

    return HTMLResponse(
        content=html,
        headers={
            "Content-Security-Policy": (
                "default-src 'self' 'unsafe-inline' 'unsafe-eval' "
                "https://cdnjs.cloudflare.com https://unpkg.com https://fonts.googleapis.com "
                "https://fonts.gstatic.com data: blob:;"
            ),
            "X-Frame-Options": "SAMEORIGIN",
        },
    )


@router.get("/projects/{project_id}/preview/card")
async def get_card(
    project_id:   str,
    current_user: UserInDB = Depends(get_current_user),
):
    card = await get_preview_card(project_id)
    if not card:
        # Generate on the fly
        from bson import ObjectId
        doc = await projects_collection().find_one({"_id": ObjectId(project_id)})
        if not doc:
            raise HTTPException(status_code=404, detail="Project not found.")
        info = await generate_preview(
            project_id=project_id,
            file_tree=doc.get("file_tree", {}),
            stack=doc.get("stack", {}),
            project_name=doc.get("name", ""),
        )
        card = info.card_data

    return card or {}


@router.get("/projects/{project_id}/deploy/configs")
async def get_deploy_configs(
    project_id:   str,
    current_user: UserInDB = Depends(get_current_user),
):
    """Return all platform config files that have been generated for this project."""
    from bson import ObjectId
    doc = await projects_collection().find_one({
        "_id":     ObjectId(project_id),
        "user_id": current_user.id,
    })
    if not doc:
        raise HTTPException(status_code=404, detail="Project not found.")

    file_tree = doc.get("file_tree", {})
    config_names = ("vercel.json", "netlify.toml", "render.yaml",
                    "fly.toml", ".cloudflare", "Dockerfile")
    return {
        "configs": {k: v for k, v in file_tree.items()
                    if any(c in k for c in config_names)},
        "deploy_urls": doc.get("deploy_urls", {}),
    }
    
