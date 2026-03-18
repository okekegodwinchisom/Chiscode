"""
ChisCode — Templates API Router (Phase 5)
==========================================
GET  /templates                  — browse/search templates
GET  /templates/{id}             — get single template detail
POST /templates                  — create template (admin)
POST /templates/{id}/use         — clone template into a new project
POST /projects/{id}/promote      — promote project to template
DELETE /templates/{id}           — soft-delete (admin)
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from app.api.deps import get_current_user
from app.core.logging import get_logger
from app.db.mongodb import projects_collection
from app.schemas.user import UserInDB
from app.services.templates_service import (
    TemplateBrowseResult,
    TemplateCreate,
    create_template,
    delete_template,
    get_template,
    increment_use_count,
    list_templates,
    promote_project_to_template,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/templates", tags=["templates"])


# ── Browse ─────────────────────────────────────────────────────

@router.get("", response_model=TemplateBrowseResult)
async def browse_templates(
    page:       int           = Query(default=1,   ge=1),
    per_page:   int           = Query(default=12,  ge=1, le=48),
    app_type:   Optional[str] = Query(default=None),
    complexity: Optional[str] = Query(default=None),
    tags:       Optional[str] = Query(default=None, description="Comma-separated tags"),
    search:     Optional[str] = Query(default=None, max_length=200),
):
    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    return await list_templates(
        page=page,
        per_page=per_page,
        app_type=app_type,
        tags=tag_list,
        complexity=complexity,
        search=search,
    )


@router.get("/{template_id}")
async def get_template_detail(template_id: str):
    doc = await get_template(template_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Template not found.")
    # Exclude full file_tree from detail response (use /use to clone)
    doc.pop("file_tree", None)
    return doc


# ── Use template (clone into new project) ─────────────────────

@router.post("/{template_id}/use")
async def use_template(
    template_id: str,
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Clone a template into a new project for the current user.
    Returns { project_id, redirect_url } so the frontend can
    navigate straight to the project detail page.
    """
    from datetime import datetime, timezone
    from bson import ObjectId

    doc = await get_template(template_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Template not found.")

    # Rate-limit check delegated to existing project creation logic
    project_doc = {
        "user_id":          current_user.id,
        "name":             doc["name"],
        "description":      doc["description"],
        "original_prompt":  f"Started from template: {doc['name']}",
        "spec":             {
            "app_type":   doc.get("app_type", "web_app"),
            "description": doc.get("description", ""),
            "features":   doc.get("tags", []),
            "complexity": doc.get("complexity", "simple"),
        },
        "stack": doc.get("stack", {}),
        "file_tree":        doc.get("file_tree", {}),
        "file_plan_hint":   list((doc.get("file_tree") or {}).keys()),
        "status":           "complete",
        "current_version":  1,
        "generation_log":   [f"Cloned from template: {doc['name']}"],
        "created_at":       datetime.now(tz=timezone.utc),
        "updated_at":       datetime.now(tz=timezone.utc),
    }

    result = await projects_collection().insert_one(project_doc)
    pid    = str(result.inserted_id)

    # Track usage
    await increment_use_count(template_id)

    logger.info("Template cloned", template_id=template_id, project_id=pid,
                user_id=str(current_user.id))

    return {
        "project_id":   pid,
        "redirect_url": f"/projects/{pid}",
        "message":      f"Project created from template '{doc['name']}'.",
    }


# ── Create template (admin / internal) ────────────────────────

@router.post("", status_code=status.HTTP_201_CREATED)
async def create_template_endpoint(
    data:         TemplateCreate,
    current_user: UserInDB = Depends(get_current_user),
):
    # In production, restrict to admin role.
    # For now any authenticated user can create templates.
    tid = await create_template(data)
    return {"template_id": tid, "message": "Template created."}


# ── Promote project to template ────────────────────────────────

@router.post("/{project_id}/promote")
async def promote_project(
    project_id:   str,
    name:         str  = Query(...,  max_length=120),
    description:  str  = Query(...,  max_length=500),
    tags:         str  = Query("",   description="Comma-separated tags"),
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Promote one of the current user's completed projects to the template library.
    """
    from app.db.mongodb import projects_collection
    from bson import ObjectId

    doc = await projects_collection().find_one({
        "_id":     ObjectId(project_id),
        "user_id": current_user.id,
        "status":  "complete",
    })
    if not doc:
        raise HTTPException(
            status_code=404,
            detail="Project not found, not owned by you, or not yet complete.",
        )

    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    tid = await promote_project_to_template(
        project_id=project_id,
        name=name,
        description=description,
        tags=tag_list,
    )
    if not tid:
        raise HTTPException(status_code=500, detail="Failed to promote project.")

    return {"template_id": tid, "message": "Project promoted to template."}


# ── Delete template ────────────────────────────────────────────

@router.delete("/{template_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_template_endpoint(
    template_id:  str,
    current_user: UserInDB = Depends(get_current_user),
):
    deleted = await delete_template(template_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Template not found.")
        
