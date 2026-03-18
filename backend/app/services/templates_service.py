"""
ChisCode — Template Service (Phase 5)
======================================
CRUD + search for the template library.
Templates are curated project blueprints users can start from.

Two sources:
  1. Admin-curated: manually inserted into MongoDB
  2. Auto-promoted: high-quality generated projects (score ≥ 85) offered
     to the user for promotion after generation

Each template has:
  - MongoDB document (full metadata + file_tree)
  - Pinecone vector (for semantic search)
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from bson import ObjectId
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.db.mongodb import get_db
from app.db.pinecone_client import (
    search_templates,
    upsert_template,
    is_available as pinecone_available,
)

logger = get_logger(__name__)


# ── Schemas ────────────────────────────────────────────────────

class TemplateStack(BaseModel):
    frontend: str = ""
    backend:  str = ""
    database: str = ""
    extras:   list[str] = Field(default_factory=list)


class TemplateCreate(BaseModel):
    name:        str
    description: str
    tags:        list[str] = Field(default_factory=list)
    app_type:    str = "web_app"
    complexity:  str = "simple"
    stack:       TemplateStack = Field(default_factory=TemplateStack)
    file_tree:   dict[str, str] = Field(default_factory=dict)
    preview_url: Optional[str] = None
    # If promoted from a project
    source_project_id: Optional[str] = None


class TemplatePublic(BaseModel):
    id:          str
    name:        str
    description: str
    tags:        list[str]
    app_type:    str
    complexity:  str
    stack:       TemplateStack
    preview_url: Optional[str]
    file_count:  int
    use_count:   int
    created_at:  datetime


class TemplateBrowseResult(BaseModel):
    templates: list[TemplatePublic]
    total:     int
    page:      int
    per_page:  int


# ── Collection helper ──────────────────────────────────────────

def _col():
    return get_db()["templates"]


# ── CRUD ──────────────────────────────────────────────────────

async def create_template(data: TemplateCreate) -> str:
    """Insert a new template. Returns the new template _id string."""
    doc = {
        **data.model_dump(),
        "stack": data.stack.model_dump(),
        "use_count":  0,
        "file_count": len(data.file_tree),
        "created_at": datetime.now(tz=timezone.utc),
        "updated_at": datetime.now(tz=timezone.utc),
        "is_active":  True,
    }
    result = await _col().insert_one(doc)
    tid = str(result.inserted_id)

    # Index in Pinecone
    if pinecone_available():
        await upsert_template(tid, doc)

    logger.info("Template created", id=tid, name=data.name)
    return tid


async def get_template(template_id: str) -> dict | None:
    doc = await _col().find_one({"_id": ObjectId(template_id), "is_active": True})
    if not doc:
        return None
    doc["id"] = str(doc.pop("_id"))
    return doc


async def list_templates(
    page:      int = 1,
    per_page:  int = 12,
    app_type:  str | None = None,
    tags:      list[str] | None = None,
    complexity: str | None = None,
    search:    str | None = None,
) -> TemplateBrowseResult:
    """
    Browse templates with optional filters.
    If `search` is provided and Pinecone is available, uses semantic search.
    Otherwise falls back to MongoDB text/filter query.
    """
    # Semantic search path
    if search and pinecone_available():
        return await _semantic_browse(search, page, per_page, app_type)

    # MongoDB filter path
    query: dict = {"is_active": True}
    if app_type:
        query["app_type"] = app_type
    if complexity:
        query["complexity"] = complexity
    if tags:
        query["tags"] = {"$in": tags}
    if search:
        query["$text"] = {"$search": search}

    total = await _col().count_documents(query)
    cursor = _col().find(query, {"file_tree": 0}) \
                   .sort("use_count", -1) \
                   .skip((page - 1) * per_page) \
                   .limit(per_page)

    templates = []
    async for doc in cursor:
        templates.append(_doc_to_public(doc))

    return TemplateBrowseResult(
        templates=templates,
        total=total,
        page=page,
        per_page=per_page,
    )


async def _semantic_browse(
    query:    str,
    page:     int,
    per_page: int,
    app_type: str | None,
) -> TemplateBrowseResult:
    """Search Pinecone, hydrate full docs from MongoDB."""
    hits = await search_templates(query, top_k=per_page * 2, app_type=app_type)
    ids  = [h["template_id"] for h in hits if "template_id" in h]

    if not ids:
        return TemplateBrowseResult(templates=[], total=0, page=page, per_page=per_page)

    oid_list = []
    for i in ids:
        try:
            oid_list.append(ObjectId(i))
        except Exception:
            pass

    docs: list[dict] = []
    async for doc in _col().find({"_id": {"$in": oid_list}, "is_active": True}, {"file_tree": 0}):
        docs.append(doc)

    # Re-sort by Pinecone score order
    id_order = {i: rank for rank, i in enumerate(ids)}
    docs.sort(key=lambda d: id_order.get(str(d["_id"]), 999))

    start = (page - 1) * per_page
    page_docs = docs[start: start + per_page]

    return TemplateBrowseResult(
        templates=[_doc_to_public(d) for d in page_docs],
        total=len(docs),
        page=page,
        per_page=per_page,
    )


async def increment_use_count(template_id: str) -> None:
    await _col().update_one(
        {"_id": ObjectId(template_id)},
        {"$inc": {"use_count": 1}, "$set": {"updated_at": datetime.now(tz=timezone.utc)}},
    )


async def promote_project_to_template(
    project_id: str,
    name:       str,
    description: str,
    tags:       list[str],
) -> str | None:
    """
    Auto-promote a high-quality project as a public template.
    Returns template_id or None on failure.
    """
    from app.db.mongodb import projects_collection
    doc = await projects_collection().find_one({"_id": ObjectId(project_id)})
    if not doc:
        return None

    stack_raw  = doc.get("stack", {}) or {}
    file_tree  = doc.get("file_tree", {}) or {}
    spec       = doc.get("spec", {}) or {}

    data = TemplateCreate(
        name=name,
        description=description,
        tags=tags,
        app_type=spec.get("app_type", "web_app"),
        complexity=spec.get("complexity", "moderate"),
        stack=TemplateStack(**{
            "frontend": stack_raw.get("frontend", ""),
            "backend":  stack_raw.get("backend", ""),
            "database": stack_raw.get("database", ""),
            "extras":   stack_raw.get("extras", []),
        }),
        file_tree=file_tree,
        source_project_id=project_id,
    )
    tid = await create_template(data)
    logger.info("Project promoted to template", project_id=project_id, template_id=tid)
    return tid


async def delete_template(template_id: str) -> bool:
    """Soft-delete a template."""
    result = await _col().update_one(
        {"_id": ObjectId(template_id)},
        {"$set": {"is_active": False, "updated_at": datetime.now(tz=timezone.utc)}},
    )
    return result.modified_count > 0


# ── Helper ─────────────────────────────────────────────────────

def _doc_to_public(doc: dict) -> TemplatePublic:
    stack_raw = doc.get("stack", {})
    return TemplatePublic(
        id=str(doc["_id"]),
        name=doc.get("name", ""),
        description=doc.get("description", ""),
        tags=doc.get("tags", []),
        app_type=doc.get("app_type", "web_app"),
        complexity=doc.get("complexity", "simple"),
        stack=TemplateStack(
            frontend=stack_raw.get("frontend", ""),
            backend=stack_raw.get("backend", ""),
            database=stack_raw.get("database", ""),
            extras=stack_raw.get("extras", []),
        ),
        preview_url=doc.get("preview_url"),
        file_count=doc.get("file_count", len(doc.get("file_tree", {}))),
        use_count=doc.get("use_count", 0),
        created_at=doc.get("created_at", datetime.now(tz=timezone.utc)),
    )
    
