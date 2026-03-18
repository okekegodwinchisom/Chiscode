"""
ChisCode — Project Routes (Phase 3)
=====================================
Generation (HITL SSE), GitHub confirm/push, iteration PR, version control.

Two-phase generation flow:
  1. POST /projects/generate         → analyze + stack suggestion (SSE)
  2. POST /projects/{id}/select-stack → user picks stack (JSON)
  3. POST /projects/{id}/generate/run → generate files (SSE)
  4. POST /projects/{id}/confirm      → push to GitHub (SSE)
"""
import json
import re
from datetime import datetime, timezone

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agents.generation_agent import (
    node_analyze_stream,
    node_generate_stream,
    node_github_stream,
    node_iterate_stream,
)
from app.api.deps import check_rate_limit, get_current_user
from app.core.logging import get_logger
from app.db.mongodb import project_versions_collection, projects_collection
from app.core.security import decrypt_value
from app.schemas.project import (
    ConfirmProjectRequest,
    GenerateProjectRequest,
    IterateProjectRequest,
    ProjectDetail,
    ProjectInDB,
    ProjectPublic,
    ProjectVersionPublic,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/projects", tags=["projects"], redirect_slashes=False)

_active_connections: dict[str, list[WebSocket]] = {}


# ── WebSocket (compat) ────────────────────────────────────────────

async def ws_broadcast(project_id: str, message: dict) -> None:
    for ws in list(_active_connections.get(project_id, [])):
        try:
            await ws.send_text(json.dumps(message))
        except Exception:
            _active_connections[project_id].remove(ws)


@router.websocket("/ws/{project_id}")
async def project_ws(websocket: WebSocket, project_id: str):
    await websocket.accept()
    _active_connections.setdefault(project_id, []).append(websocket)
    try:
        while True:
            if await websocket.receive_text() == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        if websocket in _active_connections.get(project_id, []):
            _active_connections[project_id].remove(websocket)


# ── CRUD ──────────────────────────────────────────────────────────

@router.get("/", response_model=list[ProjectPublic], response_model_by_alias=True)
async def list_projects(
    current_user=Depends(get_current_user),
    skip: int = 0, limit: int = 20,
):
    cursor = projects_collection().find(
        {"user_id": str(current_user.id)},
        sort=[("created_at", -1)], skip=skip, limit=limit,
    )
    out = []
    async for doc in cursor:
        p = ProjectInDB(**doc)
        out.append(ProjectPublic(**p.model_dump(by_alias=True), file_count=len(p.file_tree)))
    return out


@router.get("/{project_id}", response_model=ProjectDetail, response_model_by_alias=True)
async def get_project(project_id: str, current_user=Depends(get_current_user)):
    doc = await projects_collection().find_one(
        {"_id": ObjectId(project_id), "user_id": str(current_user.id)}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Project not found.")
    p = ProjectInDB(**doc)
    return ProjectDetail(**p.model_dump(by_alias=True), file_count=len(p.file_tree))


@router.delete("/{project_id}", status_code=204)
async def delete_project(project_id: str, current_user=Depends(get_current_user)):
    r = await projects_collection().delete_one(
        {"_id": ObjectId(project_id), "user_id": str(current_user.id)}
    )
    if r.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Project not found.")


# ── Phase 3 Generation Flow ───────────────────────────────────────

@router.post("/generate")
async def start_generation(
    req: GenerateProjectRequest,
    current_user=Depends(check_rate_limit),
):
    """
    STEP 1 — Analyze prompt + suggest tech stacks (SSE).
    Ends with 'stack_suggestion' event. Client shows picker.
    Client then calls POST /{id}/select-stack, then /{id}/generate/run.
    """
    project_name = req.project_name or f"project-{ObjectId()}"

    doc = {
        "user_id":            str(current_user.id),
        "name":               project_name,
        "description":        req.prompt[:200],
        "original_prompt":    req.prompt,
        "status":             "pending",
        "file_tree":          {},
        "generation_log":     ["Agent starting..."],
        "self_heal_attempts": 0,
        "current_version":    0,
        "created_at":         datetime.now(tz=timezone.utc),
        "updated_at":         datetime.now(tz=timezone.utc),
    }
    if req.preferred_stack:
        doc["stack"] = req.preferred_stack.model_dump()

    result     = await projects_collection().insert_one(doc)
    project_id = str(result.inserted_id)
    logger.info("Generation started", project_id=project_id)

    response = StreamingResponse(
        node_analyze_stream(project_id, req.prompt, project_name),
        media_type="text/event-stream",
    )
    response.headers["X-Project-Id"]      = project_id
    response.headers["Cache-Control"]     = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["Transfer-Encoding"] = "chunked"
    return response


class SelectStackRequest(BaseModel):
    option_id:  str                    # "option_a" | "option_b" | "option_c"
    custom_stack: dict | None = None   # {frontend, backend, database, extras}


@router.post("/{project_id}/select-stack")
async def select_stack(
    project_id: str,
    req: SelectStackRequest,
    current_user=Depends(get_current_user),
):
    """
    STEP 2 — HITL: user picks a stack from the suggestions.
    Saves chosen stack to DB and advances status to 'stack_selected'.
    Client then calls /{id}/generate/run.
    """
    doc = await projects_collection().find_one(
        {"_id": ObjectId(project_id), "user_id": str(current_user.id)}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Project not found.")
    if doc.get("status") != "awaiting_stack_selection":
        raise HTTPException(status_code=400,
            detail=f"Expected status 'awaiting_stack_selection', got '{doc.get('status')}'")

    # Find the chosen option from stack_options
    chosen = None
    if req.custom_stack:
        chosen = req.custom_stack
    else:
        for opt in doc.get("stack_options", []):
            if opt.get("id") == req.option_id:
                chosen = {
                    "frontend": opt.get("frontend", ""),
                    "backend":  opt.get("backend", ""),
                    "database": opt.get("database", ""),
                    "extras":   opt.get("extras", []),
                }
                break

    if not chosen:
        raise HTTPException(status_code=400, detail=f"Stack option '{req.option_id}' not found.")

    await projects_collection().update_one(
        {"_id": ObjectId(project_id)},
        {"$set": {
            "stack":      chosen,
            "status":     "stack_selected",
            "updated_at": datetime.now(tz=timezone.utc),
        }},
    )
    logger.info("Stack selected", project_id=project_id, stack=chosen)
    return {"message": "Stack selected. Call /generate/run to start code generation.", "stack": chosen}


@router.post("/{project_id}/generate/run")
async def run_generation(
    project_id: str,
    current_user=Depends(get_current_user),
):
    """
    STEP 3 — Generate files + quality check (SSE).
    Requires status='stack_selected'. Ends with 'complete' event.
    """
    doc = await projects_collection().find_one(
        {"_id": ObjectId(project_id), "user_id": str(current_user.id)}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Project not found.")
    if doc.get("status") != "stack_selected":
        raise HTTPException(status_code=400,
            detail=f"Select a stack first. Current status: '{doc.get('status')}'")

    response = StreamingResponse(
        node_generate_stream(project_id),
        media_type="text/event-stream",
    )
    response.headers["Cache-Control"]     = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["Transfer-Encoding"] = "chunked"
    return response


@router.post("/{project_id}/confirm")
async def confirm_project(
    project_id: str,
    req: ConfirmProjectRequest,
    current_user=Depends(get_current_user),
):
    """
    STEP 4 — Push to GitHub (SSE).
    Creates repo, pushes files, saves repo URL. Requires GitHub OAuth.
    """
    doc = await projects_collection().find_one(
        {"_id": ObjectId(project_id), "user_id": str(current_user.id)}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Project not found.")
    if doc.get("status") != "awaiting_confirmation":
        raise HTTPException(status_code=400, detail="Project must be awaiting confirmation.")

    # Require GitHub token if push requested
    if req.push_to_github:
        encrypted = current_user.github_token_encrypted
        if not encrypted:
            raise HTTPException(
                status_code=400,
                detail="GitHub account not connected. Log in via GitHub OAuth first.",
            )
        github_token = decrypt_value(encrypted)
    else:
        github_token = None

    if not req.push_to_github or not github_token:
        # Just mark complete without GitHub push
        await projects_collection().update_one(
            {"_id": ObjectId(project_id)},
            {"$set": {"status": "complete", "updated_at": datetime.now(tz=timezone.utc)}},
        )
        return {"message": "Project confirmed (no GitHub push)."}

    response = StreamingResponse(
        node_github_stream(
            project_id=project_id,
            github_token=github_token,
            commit_message=req.commit_message or f"Initial commit — generated by ChisCode",
            private_repo=False,
        ),
        media_type="text/event-stream",
    )
    response.headers["Cache-Control"]     = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["Transfer-Encoding"] = "chunked"
    return response


@router.post("/{project_id}/cancel")
async def cancel_project(project_id: str, current_user=Depends(get_current_user)):
    r = await projects_collection().update_one(
        {"_id": ObjectId(project_id), "user_id": str(current_user.id)},
        {"$set": {"status": "cancelled", "updated_at": datetime.now(tz=timezone.utc)}},
    )
    if r.matched_count == 0:
        raise HTTPException(status_code=404, detail="Project not found.")
    return {"message": "Project cancelled."}


# ── Iteration ─────────────────────────────────────────────────────

@router.post("/{project_id}/iterate")
async def iterate_project(
    project_id: str,
    req: IterateProjectRequest,
    current_user=Depends(check_rate_limit),
):
    """
    Refine an existing project. Pushes changed files to a PR branch (SSE).
    """
    doc = await projects_collection().find_one(
        {"_id": ObjectId(project_id), "user_id": str(current_user.id)}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Project not found.")
    if doc.get("status") not in ("complete", "awaiting_confirmation"):
        raise HTTPException(status_code=400, detail="Project must be complete before iterating.")

    encrypted = current_user.github_token_encrypted
    if not encrypted:
        raise HTTPException(status_code=400, detail="GitHub account not connected.")
    github_token = decrypt_value(encrypted)

    next_version = doc.get("current_version", 1) + 1

    response = StreamingResponse(
        node_iterate_stream(
            project_id=project_id,
            github_token=github_token,
            iterate_prompt=req.prompt,
            version=next_version,
        ),
        media_type="text/event-stream",
    )
    response.headers["Cache-Control"]     = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["Transfer-Encoding"] = "chunked"
    return response


# ── Version Control ───────────────────────────────────────────────

@router.get("/{project_id}/versions", response_model=list[ProjectVersionPublic], response_model_by_alias=True)
async def list_versions(project_id: str, current_user=Depends(get_current_user)):
    proj = await projects_collection().find_one(
        {"_id": ObjectId(project_id), "user_id": str(current_user.id)}
    )
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found.")
    cursor = project_versions_collection().find({"project_id": project_id}, sort=[("version", -1)])
    return [ProjectVersionPublic(**doc) async for doc in cursor]


@router.post("/{project_id}/rollback/{version}")
async def rollback_to_version(
    project_id: str, version: int, current_user=Depends(get_current_user),
):
    """Restore project to a saved version snapshot and push a revert commit."""
    proj = await projects_collection().find_one(
        {"_id": ObjectId(project_id), "user_id": str(current_user.id)}
    )
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found.")

    ver_doc = await project_versions_collection().find_one(
        {"project_id": project_id, "version": version}
    )
    if not ver_doc:
        raise HTTPException(status_code=404, detail=f"Version {version} not found.")

    await projects_collection().update_one(
        {"_id": ObjectId(project_id)},
        {"$set": {
            "file_tree":       ver_doc["file_snapshot"],
            "current_version": version,
            "status":          "complete",
            "updated_at":      datetime.now(tz=timezone.utc),
        }},
    )

    # If project has GitHub, push revert commit
    encrypted = current_user.github_token_encrypted
    repo_name = proj.get("github_repo_name")
    owner     = proj.get("github_owner")

    if encrypted and repo_name and owner:
        from app.services.github_service import GitHubService
        gh = GitHubService(decrypt_value(encrypted))
        try:
            await gh.push_files(
                owner=owner,
                repo=repo_name,
                file_tree=ver_doc["file_snapshot"],
                commit_message=f"revert: rollback to version {version}",
            )
        except Exception as exc:
            logger.warning("GitHub rollback push failed", error=str(exc))

    return {"message": f"Rolled back to version {version}."}
