"""
ChisCode — MCP Orchestrator
============================
LangGraph StateGraph workflows that wire MCP tools together.

Every node in the graph calls an MCP tool via _call_tool().
This means the graph is purely a coordination layer — all
actual work happens inside the MCP server tools.

Three workflows:
  1. generate_workflow  — full project generation with HITL
  2. iterate_workflow   — iterate on existing project, open PR
  3. github_workflow    — standalone repo creation + push

State flows through a typed ProjectState TypedDict.
SSE events are yielded from run_workflow() as strings.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.core.logging import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
# State
# ═══════════════════════════════════════════════════════════════

class ProjectState(TypedDict, total=False):
    # Inputs
    project_id:   str
    user_id:      str
    prompt:       str
    project_name: str
    github_token: str

    # Analysis
    spec:         dict
    stack:        dict
    stack_options: list[dict]
    file_plan:    list[str]

    # Generation
    file_tree:    dict[str, str]
    issues:       list[str]

    # GitHub
    repo_url:     str
    commit_sha:   str
    pr_url:       str

    # Control flow
    status:       str
    error:        str | None
    logs:         list[str]

    # SSE queue — nodes push events here for the stream to yield
    _sse_queue:   asyncio.Queue


# ═══════════════════════════════════════════════════════════════
# MCP Client helper
# ═══════════════════════════════════════════════════════════════

async def _call_tool(tool: str, params: dict) -> dict:
    """
    Call a local MCP tool via HTTP.
    Returns the result dict or raises on error.
    """
    import httpx
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"http://localhost:7860/api/v1/mcp/tools/{tool}",
            json={"params": params},
        )
        r.raise_for_status()
        data = r.json()
        if data.get("error"):
            raise RuntimeError(f"MCP tool '{tool}' error: {data['error']}")
        return data["result"]


def _sse(event: str, **data) -> str:
    return f"data: {json.dumps({'event': event, **data})}\n\n"


async def _push(state: ProjectState, event: str, **data) -> None:
    """Push SSE event to queue + append to logs."""
    msg = data.get("message", "")
    if msg:
        state.setdefault("logs", []).append(msg)
    q = state.get("_sse_queue")
    if q:
        await q.put(_sse(event, **data))


# ═══════════════════════════════════════════════════════════════
# WORKFLOW 1 — Generate
# Node: analyze
# ═══════════════════════════════════════════════════════════════

async def node_analyze(state: ProjectState) -> ProjectState:
    await _push(state, "status", status="analyzing", message="🔍 Analyzing requirements...")

    try:
        # Write initial DB state
        await _call_tool("project_write", {
            "project_id": state["project_id"],
            "fields":     {"status": "analyzing"},
        })

        # Analyze prompt
        result = await _call_tool("analyze_prompt", {
            "prompt":       state["prompt"],
            "project_name": state.get("project_name", "my-app"),
        })
        spec = result["spec"]

        # Suggest stacks
        await _push(state, "log", message="🧠 Evaluating best tech stacks...")
        stacks = await _call_tool("stack_advisor", {
            "prompt":     state["prompt"],
            "app_type":   spec.get("app_type", "web_app"),
            "complexity": spec.get("complexity", "moderate"),
            "features":   spec.get("features", []),
        })

        state["spec"]          = spec
        state["stack_options"] = stacks["options"]
        state["status"]        = "awaiting_stack_selection"

        # Send once only, with correct key name
        await _push(state, "stack_suggestion",
                    stacks=stacks["options"],          # ← was options=
                    message="Pick your tech stack to continue",
                    project_id=state["project_id"])
        logger.info("✅ stack_suggestion event sent")

        await _call_tool("project_write", {
            "project_id": state["project_id"],
            "fields": {
                "spec":          spec,
                "stack_options": stacks["options"],
                "status":        "awaiting_stack_selection",
            },
        })
    except Exception as exc:
        state["error"]  = str(exc)
        state["status"] = "failed"
        await _push(state, "error", message=f"Analysis failed: {exc}")
        await _call_tool("project_write", {
            "project_id": state["project_id"],
            "fields": {"status": "failed", "error_message": str(exc)},
        })

    return state


# ═══════════════════════════════════════════════════════════════
# Node: scaffold_files
# ═══════════════════════════════════════════════════════════════

async def node_scaffold(state: ProjectState) -> ProjectState:
    """Generate file plan from stack + spec."""
    try:
        result = await _call_tool("file_scaffold", {
            "spec":  state.get("spec", {}),
            "stack": state.get("stack", {}),
        })
        state["file_plan"] = result["files"]
        await _push(state, "log",
                    message=f"📋 {result['count']} files planned: {', '.join(result['files'][:5])}{'...' if result['count'] > 5 else ''}")
        await _call_tool("project_write", {
            "project_id": state["project_id"],
            "fields": {"file_plan_hint": result["files"], "status": "generating"},
        })
    except Exception as exc:
        state["error"]  = str(exc)
        state["status"] = "failed"
        await _push(state, "error", message=f"Scaffold failed: {exc}")
    return state


# ═══════════════════════════════════════════════════════════════
# Node: generate_files (parallel)
# ═══════════════════════════════════════════════════════════════

async def node_generate(state: ProjectState) -> ProjectState:
    """Generate all files in parallel via MCP code_generator tool."""
    spec       = state.get("spec", {})
    stack      = state.get("stack", {})
    file_plan  = state.get("file_plan", [])

    if not file_plan:
        state["error"]  = "No file plan — run scaffold first"
        state["status"] = "failed"
        return state

    # Build stack description for system prompt
    parts = [v for k, v in stack.items()
             if k in ("frontend", "backend", "database") and v and v.lower() != "none"]
    extras = stack.get("extras", [])
    if extras:
        parts.extend(extras[:3])
    stack_desc = " + ".join(parts) if parts else "HTML/CSS/JS"

    system_prompt = (
        f"You are an expert {stack_desc} developer.\n"
        f"App: {spec.get('app_name', 'App')} — {spec.get('description', '')}\n"
        f"Features: {', '.join(spec.get('features', []))}\n"
        f"Stack: {stack_desc}\n\n"
        f"Rules:\n"
        f"- Output ONLY raw file content — no markdown fences, no explanation\n"
        f"- Write complete, production-ready code — no placeholders\n"
        f"- Use modern best practices for every language in the stack\n"
        f"- HTML: full <!DOCTYPE html>, responsive, styled\n"
        f"- Python: syntactically valid, typed\n"
        f"- JS/TS: ES2022+, proper imports/exports\n"
    )

    await _push(state, "status", status="generating",
                message=f"⚡ Generating {len(file_plan)} files in parallel...")

    file_tree: dict[str, str] = {}
    queue: asyncio.Queue = asyncio.Queue()

    async def _gen_one(filename: str) -> None:
        try:
            result = await _call_tool("code_generator", {
                "filename":      filename,
                "system_prompt": system_prompt,
                "user_prompt":   f"Generate file: {filename}",
            })
            await queue.put(("ok", filename, result["content"]))
        except Exception as exc:
            await queue.put(("error", filename, str(exc)))

    tasks = [asyncio.create_task(_gen_one(f)) for f in file_plan]
    completed = 0

    while completed < len(file_plan):
        flag, filename, payload = await queue.get()
        completed += 1

        if flag == "ok":
            file_tree[filename] = payload
            await _push(state, "file", filename=filename,
                        size=len(payload), progress=f"{completed}/{len(file_plan)}")
            # Persist incrementally
            await _call_tool("project_write", {
                "project_id": state["project_id"],
                "fields":     {"file_tree": file_tree},
            })
        else:
            await _push(state, "log", message=f"❌ {filename} failed: {payload}")

    await asyncio.gather(*tasks, return_exceptions=True)

    state["file_tree"] = file_tree
    return state


# ═══════════════════════════════════════════════════════════════
# Node: quality_check
# ═══════════════════════════════════════════════════════════════

async def node_quality(state: ProjectState) -> ProjectState:
    await _push(state, "status", status="quality_check", message="🔎 Running quality checks...")

    try:
        result = await _call_tool("quality_checker", {
            "file_tree": state.get("file_tree", {}),
            "file_plan": state.get("file_plan", []),
        })
        issues = result.get("issues", [])
        state["issues"] = issues

        if issues:
            await _push(state, "issues", issues=issues)
            await _push(state, "log", message=f"⚠ {len(issues)} quality note(s)")
        else:
            await _push(state, "log", message="✅ Quality check passed")

        file_count = result.get("file_count", len(state.get("file_tree", {})))
        state["status"] = "complete"

        await _call_tool("project_write", {
            "project_id": state["project_id"],
            "fields": {
                "status":    "complete",
                "file_tree": state.get("file_tree", {}),
            },
        })
        await _push(state, "complete", file_count=file_count,
                    message=f"{file_count} files ready — review and confirm to push to GitHub")

    except Exception as exc:
        state["error"]  = str(exc)
        state["status"] = "failed"
        await _push(state, "error", message=f"Quality check failed: {exc}")

    return state


# ═══════════════════════════════════════════════════════════════
# Node: github_push
# ═══════════════════════════════════════════════════════════════

async def node_github_push(state: ProjectState) -> ProjectState:
    await _push(state, "status", status="committing", message="📡 Pushing to GitHub...")

    try:
        import re
        project_name = state.get("project_name", "chiscode-project")
        repo_name    = re.sub(r"[^a-zA-Z0-9_.-]", "-", project_name.lower())
        repo_name    = re.sub(r"-+", "-", repo_name).strip("-")[:100]

        spec         = state.get("spec", {})
        await _push(state, "log", message=f"📦 Creating repository: {repo_name}...")

        result = await _call_tool("github_push", {
            "github_token":   state["github_token"],
            "repo_name":      repo_name,
            "description":    spec.get("description", "")[:255],
            "file_tree":      state.get("file_tree", {}),
            "commit_message": state.get("commit_message",
                                        "Initial commit — generated by ChisCode"),
            "private":        False,
        })

        state["repo_url"]   = result["repo_url"]
        state["commit_sha"] = result["commit_sha"]
        state["status"]     = "complete"

        await _call_tool("project_write", {
            "project_id": state["project_id"],
            "fields": {
                "status":           "complete",
                "github_repo_url":  result["repo_url"],
                "github_repo_name": repo_name,
                "github_owner":     result["owner"],
                "current_version":  1,
            },
        })

        await _push(state, "log",         message=f"✅ Repository: {result['repo_url']}")
        await _push(state, "github_done", repo_url=result["repo_url"],
                    commit_sha=result["commit_sha"],
                    message="Code pushed to GitHub successfully!")

    except Exception as exc:
        state["error"]  = str(exc)
        state["status"] = "failed"
        await _push(state, "error", message=f"GitHub push failed: {exc}")
        await _call_tool("project_write", {
            "project_id": state["project_id"],
            "fields": {"status": "failed", "error_message": str(exc)},
        })

    return state


# ═══════════════════════════════════════════════════════════════
# Node: iterate (open PR with changes)
# ═══════════════════════════════════════════════════════════════

async def node_iterate(state: ProjectState) -> ProjectState:
    """Plan + generate changed files, push as PR."""
    iterate_prompt = state.get("iterate_prompt", "")
    version        = state.get("version", 2)

    await _push(state, "status", status="generating", message="🔄 Planning changes...")

    # Ask Codestral which files need changing
    try:
        file_tree = state.get("file_tree", {})
        plan_result = await _call_tool("code_generator", {
            "filename": "_plan.json",
            "system_prompt": (
                "You are a code editor. List which files need changes.\n"
                "Return ONLY valid JSON: {\"files\": [\"path1\", \"path2\"]}"
            ),
            "user_prompt": (
                f"Existing files: {list(file_tree.keys())}\n"
                f"Change request: {iterate_prompt}\n"
                f"Return JSON with files that need updating."
            ),
        })
        raw           = plan_result.get("content", "")
        files_to_change = json.loads(raw).get("files", list(file_tree.keys())[:3])
    except Exception:
        files_to_change = list(state.get("file_tree", {}).keys())[:3]

    await _push(state, "log", message=f"📋 Files to update: {', '.join(files_to_change)}")

    spec       = state.get("spec", {})
    stack      = state.get("stack", {})
    parts      = [v for k, v in stack.items()
                  if k in ("frontend", "backend", "database") and v and v.lower() != "none"]
    stack_desc = " + ".join(parts) if parts else "HTML/CSS/JS"

    changed_files: dict[str, str] = {}
    queue: asyncio.Queue = asyncio.Queue()

    async def _patch_one(filename: str) -> None:
        existing = file_tree.get(filename, "")
        try:
            result = await _call_tool("code_generator", {
                "filename":      filename,
                "system_prompt": (
                    f"You are an expert {stack_desc} developer.\n"
                    f"Modify the file according to the change request.\n"
                    f"Output ONLY the complete updated file content.\n"
                ),
                "user_prompt": (
                    f"File: {filename}\n"
                    f"Change request: {iterate_prompt}\n\n"
                    f"Current content:\n{existing[:3000]}\n\n"
                    f"Return the complete updated file."
                ),
            })
            await queue.put(("ok", filename, result["content"]))
        except Exception as exc:
            await queue.put(("error", filename, str(exc)))

    tasks = [asyncio.create_task(_patch_one(f)) for f in files_to_change]
    completed = 0
    while completed < len(files_to_change):
        flag, filename, payload = await queue.get()
        completed += 1
        if flag == "ok" and len(payload.strip()) > 10:
            changed_files[filename] = payload
            file_tree[filename]     = payload
            await _push(state, "file", filename=filename, size=len(payload))
        else:
            await _push(state, "log", message=f"❌ {filename} failed")

    await asyncio.gather(*tasks, return_exceptions=True)

    if not changed_files:
        state["error"]  = "No files were updated"
        state["status"] = "failed"
        await _push(state, "error", message="No files were updated")
        return state

    # Quality check on changed files
    qc = await _call_tool("quality_checker", {
        "file_tree": changed_files,
        "file_plan": list(changed_files.keys()),
    })
    if qc.get("issues"):
        await _push(state, "issues", issues=qc["issues"])

    # Push PR
    await _push(state, "status", status="committing", message="Opening Pull Request...")
    branch_name = f"chiscode/iteration-v{version}"
    try:
        result = await _call_tool("github_pr", {
            "github_token":   state["github_token"],
            "owner":          state.get("github_owner", ""),
            "repo":           state.get("github_repo_name", ""),
            "branch_name":    branch_name,
            "file_tree":      changed_files,
            "commit_message": f"feat: {iterate_prompt[:72]}",
            "pr_title":       f"ChisCode v{version}: {iterate_prompt[:60]}",
            "pr_body":        (
                f"## Changes\n{iterate_prompt}\n\n"
                f"**Files changed:** {', '.join(changed_files.keys())}\n\n"
                f"_Generated by ChisCode_"
            ),
        })

        state["pr_url"]    = result["pr_url"]
        state["file_tree"] = file_tree
        state["status"]    = "awaiting_confirmation"

        await _call_tool("project_write", {
            "project_id": state["project_id"],
            "fields": {"status": "awaiting_confirmation", "file_tree": file_tree},
        })
        await _push(state, "github_done",
                    pr_url=result["pr_url"], commit_sha=result["commit_sha"],
                    message="PR opened — review and merge on GitHub")

    except Exception as exc:
        state["error"] = str(exc)
        await _push(state, "error", message=f"PR failed: {exc}")

    return state


# ═══════════════════════════════════════════════════════════════
# Graph builders
# ═══════════════════════════════════════════════════════════════

def _route_after_analyze(state: ProjectState) -> Literal["scaffold", "end"]:
    """After analyze, stop at HITL — graph resumes in generate_workflow."""
    return "end" if state.get("error") else "end"  # Always pause for HITL


def _route_after_generate(state: ProjectState) -> Literal["quality", "end"]:
    return "end" if state.get("error") else "quality"


def _route_after_quality(state: ProjectState) -> Literal["end"]:
    return "end"  # Always pause for user confirm before GitHub push


def build_analyze_graph() -> StateGraph:
    """Phase 1: analyze prompt → suggest stacks. Pauses for HITL."""
    g = StateGraph(ProjectState)
    g.add_node("analyze", node_analyze)
    g.add_edge(START, "analyze")
    g.add_edge("analyze", END)
    return g.compile()


def build_generate_graph() -> StateGraph:
    """Phase 2: scaffold → parallel generate → quality check."""
    g = StateGraph(ProjectState)
    g.add_node("scaffold",  node_scaffold)
    g.add_node("generate",  node_generate)
    g.add_node("quality",   node_quality)
    g.add_edge(START,      "scaffold")
    g.add_edge("scaffold", "generate")
    g.add_conditional_edges("generate", _route_after_generate,
                            {"quality": "quality", "end": END})
    g.add_edge("quality", END)
    return g.compile()


def build_github_graph() -> StateGraph:
    """Phase 3: push files to GitHub."""
    g = StateGraph(ProjectState)
    g.add_node("github_push", node_github_push)
    g.add_edge(START, "github_push")
    g.add_edge("github_push", END)
    return g.compile()


def build_iterate_graph()
    """Iterate: plan changes → parallel patch → PR."""
    g = StateGraph(ProjectState)
    g.add_node("iterate", node_iterate)
    g.add_edge(START, "iterate")
    g.add_edge("iterate", END)
    return g.compile()


# ═══════════════════════════════════════════════════════════════
# Public SSE runner
# ═══════════════════════════════════════════════════════════════

async def run_workflow(
    graph:        StateGraph,
    initial_state: ProjectState,
) -> AsyncGenerator[str, None]:
    """
    Run a LangGraph workflow, yielding SSE strings as events arrive.
    Injects a queue into state so nodes can push events mid-execution.
    """
    queue: asyncio.Queue = asyncio.Queue()
    initial_state["_sse_queue"] = queue

    async def _run():
        try:
            await graph.ainvoke(initial_state)
        except Exception as exc:
            await queue.put(_sse("error", message=f"Workflow error: {exc}"))
        finally:
            await queue.put(None)  # sentinel

    task = asyncio.create_task(_run())

    while True:
        item = await queue.get()
        if item is None:
            break
        yield item

    await task
  
