"""
ChisCode — Pinecone Client (Phase 5)
=====================================
Async wrapper around the Pinecone SDK.
Handles index connection, upsert, similarity search, and deletion.

Index layout:
  name:      from settings.pinecone_index  (default: "chiscode-embeddings")
  dims:      1024   (mistral-embed output)
  metric:    cosine
  metadata:  app_type, stack_json, features, description, complexity, user_id
"""
from __future__ import annotations

import json
from typing import Any

import structlog
from pinecone import Pinecone, ServerlessSpec

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Module-level client ────────────────────────────────────────
_pc:    Pinecone | None = None
_index: Any             = None   # pinecone.Index


async def connect() -> None:
    """Initialise Pinecone client and ensure index exists."""
    global _pc, _index
    if not settings.pinecone_api_key:
        logger.warning("PINECONE_API_KEY not set — RAG disabled")
        return

    try:
        _pc = Pinecone(api_key=settings.pinecone_api_key)

        existing = [i.name for i in _pc.list_indexes()]
        if settings.pinecone_index not in existing:
            logger.info("Creating Pinecone index", name=settings.pinecone_index)
            _pc.create_index(
                name=settings.pinecone_index,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=settings.pinecone_environment,
                ),
            )

        _index = _pc.Index(settings.pinecone_index)
        stats  = _index.describe_index_stats()
        logger.info("Pinecone connected",
                    index=settings.pinecone_index,
                    total_vectors=stats.total_vector_count)
    except Exception as exc:
        logger.error("Pinecone connection failed", error=str(exc))
        _pc    = None
        _index = None


async def disconnect() -> None:
    global _pc, _index
    _pc    = None
    _index = None
    logger.info("Pinecone disconnected")


def get_index() -> Any | None:
    return _index


def is_available() -> bool:
    return _index is not None


# ── Embed via Mistral ──────────────────────────────────────────

async def embed_text(text: str) -> list[float] | None:
    """
    Generate a 1024-dim embedding using Mistral's embed model.
    Returns None on failure so callers can gracefully skip RAG.
    """
    if not settings.codestral_api_key:
        return None
    try:
        import httpx
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                "https://api.mistral.ai/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {settings.codestral_api_key}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model": "mistral-embed",
                    "input": [text[:8000]],   # trim to model limit
                },
            )
            r.raise_for_status()
            return r.json()["data"][0]["embedding"]
    except Exception as exc:
        logger.warning("Embedding failed", error=str(exc))
        return None


# ── Upsert ─────────────────────────────────────────────────────

async def upsert_project(
    project_id: str,
    spec:        dict,
    stack:       dict,
    user_id:     str,
) -> bool:
    """
    Embed the project spec and upsert into Pinecone.
    Called by the finalize node after successful generation.
    Returns True on success, False on failure (caller should not crash).
    """
    if not is_available():
        return False

    # Build text to embed — rich description for similarity matching
    parts = [
        spec.get("description", ""),
        spec.get("app_type", ""),
        " ".join(spec.get("features", [])),
    ]
    text    = " | ".join(p for p in parts if p)
    vector  = await embed_text(text)
    if not vector:
        return False

    # Flatten stack for metadata (Pinecone doesn't support nested dicts)
    try:
        _index.upsert(vectors=[{
            "id":     f"project-{project_id}",
            "values": vector,
            "metadata": {
                "project_id":  project_id,
                "user_id":     user_id,
                "app_type":    spec.get("app_type", ""),
                "description": spec.get("description", "")[:500],
                "features":    spec.get("features", [])[:10],
                "complexity":  spec.get("complexity", "moderate"),
                "stack_json":  json.dumps({
                    "frontend": stack.get("frontend", ""),
                    "backend":  stack.get("backend",  ""),
                    "database": stack.get("database", ""),
                }),
            },
        }])
        logger.info("Pinecone upsert OK", project_id=project_id)
        return True
    except Exception as exc:
        logger.error("Pinecone upsert failed", error=str(exc))
        return False


async def upsert_template(template_id: str, template: dict) -> bool:
    """Upsert a curated template into Pinecone."""
    if not is_available():
        return False

    text   = f"{template.get('name','')} {template.get('description','')} {' '.join(template.get('tags',[]))}"
    vector = await embed_text(text)
    if not vector:
        return False

    try:
        _index.upsert(vectors=[{
            "id":     f"template-{template_id}",
            "values": vector,
            "metadata": {
                "type":        "template",
                "template_id": template_id,
                "name":        template.get("name", ""),
                "description": template.get("description", "")[:500],
                "tags":        template.get("tags", [])[:10],
                "app_type":    template.get("app_type", ""),
                "complexity":  template.get("complexity", "simple"),
                "stack_json":  json.dumps(template.get("stack", {})),
            },
        }])
        return True
    except Exception as exc:
        logger.error("Template upsert failed", error=str(exc))
        return False


# ── Query ──────────────────────────────────────────────────────

async def search_similar_projects(
    prompt:      str,
    top_k:       int = 3,
    filter_dict: dict | None = None,
) -> list[dict]:
    """
    Find similar past projects to use as few-shot RAG context.
    Returns list of metadata dicts (empty list on failure).
    """
    if not is_available():
        return []

    vector = await embed_text(prompt)
    if not vector:
        return []

    try:
        query_filter = filter_dict or {}
        # Exclude template entries
        query_filter["type"] = {"$ne": "template"}

        results = _index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            filter=query_filter if query_filter else None,
        )
        return [
            {**m.metadata, "score": m.score}
            for m in results.matches
            if m.score >= 0.65   # similarity threshold
        ]
    except Exception as exc:
        logger.warning("Pinecone search failed", error=str(exc))
        return []


async def search_templates(
    prompt:  str,
    top_k:   int = 5,
    app_type: str | None = None,
) -> list[dict]:
    """Search curated templates by semantic similarity."""
    if not is_available():
        return []

    vector = await embed_text(prompt)
    if not vector:
        return []

    try:
        flt = {"type": {"$eq": "template"}}
        if app_type:
            flt["app_type"] = {"$eq": app_type}

        results = _index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            filter=flt,
        )
        return [
            {**m.metadata, "score": m.score}
            for m in results.matches
            if m.score >= 0.5
        ]
    except Exception as exc:
        logger.warning("Template search failed", error=str(exc))
        return []


async def delete_project(project_id: str) -> bool:
    """Remove a project's vector when the project is deleted."""
    if not is_available():
        return False
    try:
        _index.delete(ids=[f"project-{project_id}"])
        return True
    except Exception as exc:
        logger.warning("Pinecone delete failed", error=str(exc))
        return False
        
