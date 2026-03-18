"""
ChisCode — Generation Agent (Phase 4)
======================================
Thin SSE wrappers that delegate to the orchestrator's LangGraph workflows.
Each endpoint is a clean one-liner that wires state and runs the right graph.
"""
from __future__ import annotations

from typing import AsyncGenerator

from app.agents.orchestrator import (
    ProjectState,
    build_analyze_graph,
    build_generate_graph,
    build_github_graph,
    build_iterate_graph,
    run_workflow,
)
from app.db.mongodb import projects_collection
from bson import ObjectId


async def analyze_stream(
    project_id:   str,
    prompt:       str,
    project_name: str,
) -> AsyncGenerator[str, None]:
    """Analyze prompt + suggest stacks. Pauses at HITL."""
    state: ProjectState = {
        "project_id":   project_id,
        "prompt":       prompt,
        "project_name": project_name,
        "logs":         [],
    }
    async for chunk in run_workflow(build_analyze_graph(), state):
        yield chunk


async def generate_stream(project_id: str) -> AsyncGenerator[str, None]:
    """Generate files + quality check after HITL stack selection."""
    doc = await projects_collection().find_one({"_id": ObjectId(project_id)})
    if not doc:
        yield 'data: {"event":"error","message":"Project not found"}\n\n'
        return

    state: ProjectState = {
        "project_id":   project_id,
        "prompt":       doc.get("original_prompt", ""),
        "project_name": doc.get("name", "my-app"),
        "spec":         doc.get("spec", {}),
        "stack":        doc.get("stack", {}),
        "file_plan":    doc.get("file_plan_hint", []),
        "file_tree":    doc.get("file_tree", {}),
        "logs":         [],
    }
    async for chunk in run_workflow(build_generate_graph(), state):
        yield chunk


async def github_stream(
    project_id:     str,
    github_token:   str,
    commit_message: str,
) -> AsyncGenerator[str, None]:
    """Push generated files to GitHub."""
    doc = await projects_collection().find_one({"_id": ObjectId(project_id)})
    if not doc:
        yield 'data: {"event":"error","message":"Project not found"}\n\n'
        return

    state: ProjectState = {
        "project_id":     project_id,
        "project_name":   doc.get("name", "my-app"),
        "spec":           doc.get("spec", {}),
        "file_tree":      doc.get("file_tree", {}),
        "github_token":   github_token,
        "commit_message": commit_message,
        "logs":           [],
    }
    async for chunk in run_workflow(build_github_graph(), state):
        yield chunk


async def iterate_stream(
    project_id:     str,
    github_token:   str,
    iterate_prompt: str,
    version:        int,
) -> AsyncGenerator[str, None]:
    """Iterate on existing project — patch files + open PR."""
    doc = await projects_collection().find_one({"_id": ObjectId(project_id)})
    if not doc:
        yield 'data: {"event":"error","message":"Project not found"}\n\n'
        return

    state: ProjectState = {
        "project_id":       project_id,
        "project_name":     doc.get("name", "my-app"),
        "prompt":           doc.get("original_prompt", ""),
        "spec":             doc.get("spec", {}),
        "stack":            doc.get("stack", {}),
        "file_tree":        dict(doc.get("file_tree", {})),
        "github_token":     github_token,
        "github_owner":     doc.get("github_owner", ""),
        "github_repo_name": doc.get("github_repo_name", ""),
        "iterate_prompt":   iterate_prompt,
        "version":          version,
        "logs":             [],
    }
    async for chunk in run_workflow(build_iterate_graph(), state):
        yield chunk


# ── Legacy compat aliases for projects.py ─────────────────────
node_analyze_stream  = analyze_stream
node_generate_stream = generate_stream
node_github_stream   = lambda project_id, github_token, commit_message, **_: \
    github_stream(project_id, github_token, commit_message)
node_iterate_stream  = lambda project_id, github_token, iterate_prompt, version: \
    iterate_stream(project_id, github_token, iterate_prompt, version)
  
