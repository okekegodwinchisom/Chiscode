"""
ChisCode — MCP Tool Server
===========================
Exposes all agent capabilities as MCP-compatible JSON-RPC tools.
Mounted at /api/mcp on the main FastAPI app.

Every tool follows the same contract:
  POST /api/mcp/tools/{tool_name}
  Body:  { "params": { ... } }
  Reply: { "result": { ... }, "error": null }
         { "result": null,   "error": "message" }

The orchestrator and all agents call these endpoints — they never
import each other directly. This means:
  - Tools are independently testable
  - New tools can be added without touching agents
  - Agents can run on separate workers in future
"""
from __future__ import annotations

import ast
import asyncio
import json
import re
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/mcp", tags=["mcp"])


# ── Tool envelope ────────────────────────────────────────────────

class ToolRequest(BaseModel):
    params: dict[str, Any] = {}


class ToolResponse(BaseModel):
    result: Any = None
    error:  str | None = None


def ok(data: Any) -> ToolResponse:
    return ToolResponse(result=data)


def err(msg: str) -> ToolResponse:
    logger.warning("MCP tool error", error=msg)
    return ToolResponse(error=msg)


# ── Tool registry (for discovery) ───────────────────────────────

TOOL_REGISTRY = {
    "code_generator": {
        "description": "Generate a single source file using Codestral",
        "params": ["filename", "system_prompt", "user_prompt"],
    },
    "stack_advisor": {
        "description": "Suggest 3 ranked tech stacks for an app description",
        "params": ["prompt", "app_type", "complexity", "features"],
    },
    "quality_checker": {
        "description": "Lint and validate a file_tree dict",
        "params": ["file_tree", "file_plan"],
    },
    "file_scaffold": {
        "description": "Generate a file plan list from a stack + spec",
        "params": ["spec", "stack"],
    },
    "github_push": {
        "description": "Create GitHub repo and push all files",
        "params": ["github_token", "repo_name", "description", "file_tree", "commit_message", "private"],
    },
    "github_pr": {
        "description": "Push changed files to a branch and open a PR",
        "params": ["github_token", "owner", "repo", "branch_name", "file_tree", "commit_message", "pr_title", "pr_body"],
    },
    "project_read": {
        "description": "Read a project document from MongoDB",
        "params": ["project_id"],
    },
    "project_write": {
        "description": "Update fields on a project document",
        "params": ["project_id", "fields"],
    },
    "project_log": {
        "description": "Append a log line to a project document",
        "params": ["project_id", "message"],
    },
    "search_web": {
        "description": "DuckDuckGo search — returns top snippets",
        "params": ["query", "max_results"],
    },
    "analyze_prompt": {
        "description": "Parse a natural language app description into a structured spec",
        "params": ["prompt", "project_name"],
    },
}


@router.get("/tools")
async def list_tools():
    """Discover all registered MCP tools."""
    return {"tools": TOOL_REGISTRY}


# ═══════════════════════════════════════════════════════════════
# TOOL: code_generator
# ═══════════════════════════════════════════════════════════════

@router.post("/tools/code_generator", response_model=ToolResponse)
async def tool_code_generator(req: ToolRequest):
    """
    Generate a single file via Codestral.
    Returns { content: str, filename: str, chars: int }
    """
    p = req.params
    filename      = p.get("filename", "")
    system_prompt = p.get("system_prompt", "You are an expert developer.")
    user_prompt   = p.get("user_prompt", f"Generate file: {filename}")

    if not filename:
        return err("filename is required")

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_mistralai import ChatMistralAI

        llm = ChatMistralAI(
            model=settings.codestral_model,
            api_key=settings.codestral_api_key,
            base_url=settings.codestral_base_url,
            temperature=p.get("temperature", 0.2),
            max_tokens=8192,
        )
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        content = _strip_fences(response.content)
        return ok({"content": content, "filename": filename, "chars": len(content)})

    except Exception as exc:
        return err(f"code_generator failed for {filename}: {exc}")


# ═══════════════════════════════════════════════════════════════
# TOOL: stack_advisor
# ═══════════════════════════════════════════════════════════════

@router.post("/tools/stack_advisor", response_model=ToolResponse)
async def tool_stack_advisor(req: ToolRequest):
    """Suggest 3 tech stacks. Returns { options: [...] }"""
    p = req.params
    try:
        from app.agents.stack_advisor import suggest_stacks
        options = await suggest_stacks(
            prompt     = p.get("prompt", ""),
            app_type   = p.get("app_type", "web_app"),
            complexity = p.get("complexity", "moderate"),
            features   = p.get("features", []),
        )
        return ok({"options": options})
    except Exception as exc:
        return err(f"stack_advisor failed: {exc}")


# ═══════════════════════════════════════════════════════════════
# TOOL: quality_checker
# ═══════════════════════════════════════════════════════════════

@router.post("/tools/quality_checker", response_model=ToolResponse)
async def tool_quality_checker(req: ToolRequest):
    """
    Lint and validate generated files.
    Returns { issues: [...], passed: bool }
    """
    p         = req.params
    file_tree = p.get("file_tree", {})
    file_plan = p.get("file_plan", list(file_tree.keys()))
    issues    = []

    # Missing files
    missing = [f for f in file_plan if f not in file_tree]
    if missing:
        issues.append(f"Missing files: {', '.join(missing)}")

    for path, content in file_tree.items():
        # Empty file
        if not content or len(content.strip()) < 20:
            issues.append(f"{path}: file is empty")
            continue

        # Python syntax
        if path.endswith(".py"):
            try:
                ast.parse(content)
            except SyntaxError as e:
                issues.append(f"{path}: SyntaxError line {e.lineno} — {e.msg}")

        # HTML structure
        if path.endswith(".html"):
            low = content.lower()
            if "<!doctype" not in low and "<html" not in low:
                issues.append(f"{path}: missing DOCTYPE or <html> root")

        # JSON validity
        if path.endswith(".json"):
            try:
                json.loads(content)
            except json.JSONDecodeError as e:
                issues.append(f"{path}: invalid JSON — {e.msg}")

        # package.json sanity
        if path == "package.json":
            try:
                pkg = json.loads(content)
                if "name" not in pkg:
                    issues.append("package.json: missing 'name' field")
            except Exception:
                pass

    if not any(p.lower() == "readme.md" for p in file_tree):
        issues.append("README.md is missing")

    return ok({"issues": issues, "passed": len(issues) == 0, "file_count": len(file_tree)})


# ═══════════════════════════════════════════════════════════════
# TOOL: file_scaffold
# ═══════════════════════════════════════════════════════════════

@router.post("/tools/file_scaffold", response_model=ToolResponse)
async def tool_file_scaffold(req: ToolRequest):
    """
    Generate the file plan (list of paths) for a given stack + spec.
    Returns { files: [...] }
    """
    p    = req.params
    spec = p.get("spec", {})
    stack = p.get("stack", {})

    frontend = (stack.get("frontend") or "").lower()
    backend  = (stack.get("backend")  or "").lower()
    database = (stack.get("database") or "").lower()
    extras   = [e.lower() for e in stack.get("extras", [])]
    complexity = spec.get("complexity", "simple")

    files = ["README.md"]

    # Frontend
    if "next.js" in frontend or "next" in frontend:
        files += ["src/app/page.tsx", "src/app/layout.tsx",
                  "src/components/ui/button.tsx", "package.json",
                  "next.config.js", "tailwind.config.js", "tsconfig.json"]
    elif "react" in frontend:
        files += ["src/App.jsx", "src/main.jsx", "src/components/Layout.jsx",
                  "src/index.css", "index.html", "package.json", "vite.config.js"]
    elif "vue" in frontend or "nuxt" in frontend:
        files += ["src/App.vue", "src/main.js", "package.json"]
    elif "svelte" in frontend or "sveltekit" in frontend:
        files += ["src/routes/+page.svelte", "src/routes/+layout.svelte",
                  "package.json", "svelte.config.js"]
    else:
        # Vanilla HTML
        files += ["index.html", "css/style.css", "js/app.js"]

    # Backend
    if "fastapi" in backend or "python" in backend:
        files += ["main.py", "requirements.txt"]
        if complexity in ("moderate", "complex"):
            files += ["app/routes/__init__.py", "app/models.py",
                      "app/config.py", "app/database.py"]
        if spec.get("auth_required"):
            files.append("app/auth.py")
    elif "express" in backend or "node" in backend:
        files += ["server.js", "package.json", "routes/index.js"]
        if complexity in ("moderate", "complex"):
            files += ["middleware/auth.js", "models/index.js"]
    elif "rust" in backend or "axum" in backend:
        files += ["src/main.rs", "Cargo.toml"]
        if complexity in ("moderate", "complex"):
            files += ["src/routes.rs", "src/models.rs", "src/db.rs"]
    elif "go" in backend or "gin" in backend:
        files += ["main.go", "go.mod", "handlers/handlers.go"]

    # Database
    if "prisma" in extras or "prisma" in database:
        files.append("prisma/schema.prisma")
    if "alembic" in extras or "sqlalchemy" in backend:
        files += ["alembic.ini", "migrations/env.py"]

    # Config / infra
    files.append(".env.example")
    if complexity == "complex":
        files += ["Dockerfile", "docker-compose.yml", ".github/workflows/ci.yml"]

    # Deduplicate preserving order
    seen, unique = set(), []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)

    return ok({"files": unique, "count": len(unique)})


# ═══════════════════════════════════════════════════════════════
# TOOL: github_push
# ═══════════════════════════════════════════════════════════════

@router.post("/tools/github_push", response_model=ToolResponse)
async def tool_github_push(req: ToolRequest):
    """Create GitHub repo + push all files. Returns { repo_url, commit_sha, owner }"""
    p = req.params
    token          = p.get("github_token", "")
    repo_name      = p.get("repo_name", "")
    description    = p.get("description", "")
    file_tree      = p.get("file_tree", {})
    commit_message = p.get("commit_message", "Initial commit — generated by ChisCode")
    private        = p.get("private", False)

    if not token:
        return err("github_token is required")
    if not repo_name:
        return err("repo_name is required")
    if not file_tree:
        return err("file_tree is empty — nothing to push")

    try:
        from app.services.github_service import GitHubService, GitHubError
        gh     = GitHubService(token)
        result = await gh.create_repo_and_push(
            repo_name=repo_name,
            description=description[:255],
            file_tree=file_tree,
            commit_message=commit_message,
            private=private,
        )
        return ok(result)
    except Exception as exc:
        return err(f"github_push failed: {exc}")


# ═══════════════════════════════════════════════════════════════
# TOOL: github_pr
# ═══════════════════════════════════════════════════════════════

@router.post("/tools/github_pr", response_model=ToolResponse)
async def tool_github_pr(req: ToolRequest):
    """Push to feature branch + open PR. Returns { pr_url, commit_sha, branch }"""
    p = req.params
    token          = p.get("github_token", "")
    owner          = p.get("owner", "")
    repo           = p.get("repo", "")
    branch_name    = p.get("branch_name", "chiscode/update")
    file_tree      = p.get("file_tree", {})
    commit_message = p.get("commit_message", "feat: ChisCode iteration")
    pr_title       = p.get("pr_title", "ChisCode Update")
    pr_body        = p.get("pr_body", "_Generated by ChisCode_")

    if not all([token, owner, repo]):
        return err("github_token, owner, repo are all required")

    try:
        from app.services.github_service import GitHubService
        gh     = GitHubService(token)
        result = await gh.push_iteration_pr(
            owner=owner, repo=repo, branch_name=branch_name,
            file_tree=file_tree, commit_message=commit_message,
            pr_title=pr_title, pr_body=pr_body,
        )
        return ok(result)
    except Exception as exc:
        return err(f"github_pr failed: {exc}")


# ═══════════════════════════════════════════════════════════════
# TOOL: project_read / project_write / project_log
# ═══════════════════════════════════════════════════════════════

@router.post("/tools/project_read", response_model=ToolResponse)
async def tool_project_read(req: ToolRequest):
    """Read a project from MongoDB. Returns the full doc (no _id)."""
    project_id = req.params.get("project_id", "")
    if not project_id:
        return err("project_id is required")
    try:
        from bson import ObjectId
        from app.db.mongodb import projects_collection
        doc = await projects_collection().find_one({"_id": ObjectId(project_id)})
        if not doc:
            return err(f"Project {project_id} not found")
        doc["_id"] = str(doc["_id"])
        doc["user_id"] = str(doc.get("user_id", ""))
        return ok(doc)
    except Exception as exc:
        return err(f"project_read failed: {exc}")


@router.post("/tools/project_write", response_model=ToolResponse)
async def tool_project_write(req: ToolRequest):
    """Update fields on a project. Returns { updated: bool }"""
    p          = req.params
    project_id = p.get("project_id", "")
    fields     = p.get("fields", {})
    if not project_id:
        return err("project_id is required")
    if not fields:
        return err("fields dict is empty")
    try:
        from datetime import datetime, timezone
        from bson import ObjectId
        from app.db.mongodb import projects_collection
        fields["updated_at"] = datetime.now(tz=timezone.utc)
        r = await projects_collection().update_one(
            {"_id": ObjectId(project_id)}, {"$set": fields}
        )
        return ok({"updated": r.modified_count > 0})
    except Exception as exc:
        return err(f"project_write failed: {exc}")


@router.post("/tools/project_log", response_model=ToolResponse)
async def tool_project_log(req: ToolRequest):
    """Append a log line. Returns { appended: bool }"""
    p          = req.params
    project_id = p.get("project_id", "")
    message    = p.get("message", "")
    if not project_id or not message:
        return err("project_id and message are required")
    try:
        from datetime import datetime, timezone
        from bson import ObjectId
        from app.db.mongodb import projects_collection
        await projects_collection().update_one(
            {"_id": ObjectId(project_id)},
            {"$push": {"generation_log": message},
             "$set":  {"updated_at": datetime.now(tz=timezone.utc)}},
        )
        return ok({"appended": True})
    except Exception as exc:
        return err(f"project_log failed: {exc}")


# ═══════════════════════════════════════════════════════════════
# TOOL: search_web
# ═══════════════════════════════════════════════════════════════

@router.post("/tools/search_web", response_model=ToolResponse)
async def tool_search_web(req: ToolRequest):
    """
    DuckDuckGo search — useful for agents to look up API docs,
    package versions, or framework patterns before generating code.
    Returns { results: [{title, url, snippet}] }
    """
    p           = req.params
    query       = p.get("query", "")
    max_results = int(p.get("max_results", 5))

    if not query:
        return err("query is required")

    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title":   r.get("title", ""),
                    "url":     r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        return ok({"results": results, "count": len(results)})
    except Exception as exc:
        return err(f"search_web failed: {exc}")


# ═══════════════════════════════════════════════════════════════
# TOOL: analyze_prompt
# ═══════════════════════════════════════════════════════════════

@router.post("/tools/analyze_prompt", response_model=ToolResponse)
async def tool_analyze_prompt(req: ToolRequest):
    """
    Parse a natural language app description into a structured spec dict.
    Returns { spec: {...}, file_plan: [...] }
    """
    p            = req.params
    prompt       = p.get("prompt", "")
    project_name = p.get("project_name", "my-app")

    if not prompt:
        return err("prompt is required")

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_mistralai import ChatMistralAI

        llm = ChatMistralAI(
            model=settings.codestral_model,
            api_key=settings.codestral_api_key,
            base_url=settings.codestral_base_url,
            temperature=0.1, max_tokens=2048,
        )
        response = await llm.ainvoke([
            SystemMessage(content="""Analyze the app idea. Return ONLY valid JSON:
{
  "app_type": "web_app|api|cli|mobile_web|dashboard|landing_page|e_commerce|real_time_app|data_app|game|ai_app",
  "app_name": "snake_case_name",
  "description": "one clear sentence",
  "features": ["feature1", "feature2"],
  "auth_required": false,
  "database_needed": false,
  "api_needed": false,
  "mobile_responsive": true,
  "complexity": "simple|moderate|complex"
}"""),
            HumanMessage(content=f"App idea: {prompt}\nProject name hint: {project_name}"),
        ])
        raw  = _strip_fences(response.content)
        spec = json.loads(raw)
        return ok({"spec": spec})
    except Exception as exc:
        return err(f"analyze_prompt failed: {exc}")


# ── Helper ───────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:\w+)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()
    
