"""
ChisCode — Deployment Service (Phase 6)
========================================
Handles one-click deployment of generated projects to external platforms.

Supported platforms:
  - vercel      (via Vercel REST API — no CLI needed)
  - netlify     (via Netlify REST API)
  - render      (via Render REST API)
  - fly         (config generation only — fly.io requires CLI)
  - cloudflare  (via Cloudflare Pages API)
  - github_pages (via GitHub Pages API — already have token)

For HF Spaces (no shell access to run CLI tools):
  All deployments use REST APIs only.
  fly.io/AWS/GCP generate config files for manual deploy.

Each platform deploy:
  1. Validates project is complete
  2. Generates platform config file(s)
  3. Calls platform API (or generates ready-to-deploy archive)
  4. Returns { status, url, deploy_id, logs }
"""
from __future__ import annotations

import asyncio
import base64
import json
import zipfile
import io
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Literal, Optional

import httpx
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

Platform = Literal[
    "vercel", "netlify", "render", "fly", "cloudflare",
    "github_pages", "download"
]


# ── Schemas ────────────────────────────────────────────────────

class DeployConfig(BaseModel):
    platform:     Platform
    project_name: str
    project_id:   str
    user_id:      str
    # Optional platform-specific tokens (provided by user in UI)
    vercel_token:    Optional[str] = None
    netlify_token:   Optional[str] = None
    render_token:    Optional[str] = None
    cf_api_token:    Optional[str] = None
    cf_account_id:   Optional[str] = None
    github_token:    Optional[str] = None  # for github_pages
    github_username: Optional[str] = None
    # From project
    stack:        dict = {}
    file_tree:    dict[str, str] = {}


class DeployResult(BaseModel):
    platform:  str
    status:    Literal["success", "failed", "pending", "config_only"]
    deploy_id: Optional[str] = None
    url:       Optional[str] = None
    logs:      list[str]     = []
    config_files: dict[str, str] = {}   # generated config files to commit
    error:     Optional[str] = None


# ═══════════════════════════════════════════════════════════════
# Main dispatch
# ═══════════════════════════════════════════════════════════════

async def deploy_project(cfg: DeployConfig) -> AsyncGenerator[dict, None]:
    """
    Stream deployment progress events.
    Yields SSE-style dicts: { event, message, ... }
    """
    dispatchers = {
        "vercel":       _deploy_vercel,
        "netlify":      _deploy_netlify,
        "render":       _deploy_render,
        "fly":          _generate_fly_config,
        "cloudflare":   _deploy_cloudflare,
        "github_pages": _deploy_github_pages,
        "download":     _prepare_download,
    }
    fn = dispatchers.get(cfg.platform)
    if not fn:
        yield {"event": "error", "message": f"Unknown platform: {cfg.platform}"}
        return

    yield {"event": "deploy_start", "platform": cfg.platform, "message": f"Starting {cfg.platform} deployment…"}

    try:
        async for event in fn(cfg):
            yield event
    except Exception as exc:
        logger.error("Deploy failed", platform=cfg.platform, error=str(exc))
        yield {"event": "error", "message": str(exc)}


# ═══════════════════════════════════════════════════════════════
# Platform: Vercel (REST API)
# ═══════════════════════════════════════════════════════════════

async def _deploy_vercel(cfg: DeployConfig) -> AsyncGenerator[dict, None]:
    token = cfg.vercel_token
    if not token:
        yield {"event": "error", "message": "Vercel token required. Set it in your deployment settings."}
        return

    yield {"event": "log", "message": "📦 Preparing files for Vercel…"}

    # Generate vercel.json
    vercel_json = _make_vercel_config(cfg)
    files_to_deploy = {**cfg.file_tree, "vercel.json": json.dumps(vercel_json, indent=2)}

    # Build Vercel deployment files payload
    files_payload = []
    for path, content in files_to_deploy.items():
        encoded = base64.b64encode(content.encode()).decode()
        files_payload.append({"file": path, "data": encoded, "encoding": "base64"})

    yield {"event": "log", "message": f"🚀 Deploying {len(files_payload)} files to Vercel…"}

    async with httpx.AsyncClient(timeout=120) as client:
        # Create deployment
        resp = await client.post(
            "https://api.vercel.com/v13/deployments",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={
                "name":   _sanitize_name(cfg.project_name),
                "files":  files_payload,
                "target": "production",
                "projectSettings": {"framework": _detect_vercel_framework(cfg.stack)},
            },
        )

        if resp.status_code not in (200, 201):
            yield {"event": "error", "message": f"Vercel API error {resp.status_code}: {resp.text[:300]}"}
            return

        data      = resp.json()
        deploy_id = data.get("id", "")
        url       = data.get("url", "")

        yield {"event": "log",    "message": f"✅ Deployment created: {deploy_id}"}
        yield {"event": "log",    "message": f"⏳ Waiting for build to complete…"}

        # Poll for completion (up to 3 minutes)
        for _ in range(36):
            await asyncio.sleep(5)
            poll = await client.get(
                f"https://api.vercel.com/v13/deployments/{deploy_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            state = poll.json().get("readyState", "")
            if state == "READY":
                yield {"event": "deploy_done", "url": f"https://{url}",
                       "message": f"🎉 Live at https://{url}"}
                return
            elif state in ("ERROR", "CANCELED"):
                yield {"event": "error", "message": f"Vercel build {state.lower()}"}
                return
            yield {"event": "log", "message": f"  Build state: {state}…"}

        yield {"event": "error", "message": "Vercel deployment timed out (3 min)"}


def _make_vercel_config(cfg: DeployConfig) -> dict:
    framework = _detect_vercel_framework(cfg.stack)
    base: dict[str, Any] = {}
    if framework:
        base["framework"] = framework
    # SPA routing
    if any(p.endswith(".html") for p in cfg.file_tree):
        base["rewrites"] = [{"source": "/((?!api/).*)", "destination": "/index.html"}]
    return base


def _detect_vercel_framework(stack: dict) -> str | None:
    fe = (stack.get("frontend") or "").lower()
    if "next"    in fe: return "nextjs"
    if "react"   in fe: return "create-react-app"
    if "vue"     in fe: return "vue"
    if "nuxt"    in fe: return "nuxtjs"
    if "svelte"  in fe: return "svelte"
    if "astro"   in fe: return "astro"
    return None


# ═══════════════════════════════════════════════════════════════
# Platform: Netlify (REST API)
# ═══════════════════════════════════════════════════════════════

async def _deploy_netlify(cfg: DeployConfig) -> AsyncGenerator[dict, None]:
    token = cfg.netlify_token
    if not token:
        yield {"event": "error", "message": "Netlify token required."}
        return

    yield {"event": "log", "message": "📦 Building Netlify deploy archive…"}

    # Generate netlify.toml
    netlify_toml = _make_netlify_toml(cfg)
    files_to_deploy = {**cfg.file_tree, "netlify.toml": netlify_toml}

    # Zip everything
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, content in files_to_deploy.items():
            zf.writestr(path, content)
    zip_buf.seek(0)
    zip_bytes = zip_buf.read()

    yield {"event": "log", "message": f"🚀 Uploading to Netlify ({len(zip_bytes)//1024}KB)…"}

    async with httpx.AsyncClient(timeout=120) as client:
        # Create site first (or reuse)
        site_resp = await client.post(
            "https://api.netlify.com/api/v1/sites",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"name": _sanitize_name(cfg.project_name)},
        )
        if site_resp.status_code not in (200, 201):
            yield {"event": "error", "message": f"Netlify site create error: {site_resp.text[:300]}"}
            return

        site_id = site_resp.json().get("id")

        # Deploy zip
        deploy_resp = await client.post(
            f"https://api.netlify.com/api/v1/sites/{site_id}/deploys",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/zip"},
            content=zip_bytes,
        )
        if deploy_resp.status_code not in (200, 201):
            yield {"event": "error", "message": f"Netlify deploy error: {deploy_resp.text[:300]}"}
            return

        data = deploy_resp.json()
        url  = data.get("deploy_ssl_url") or data.get("url", "")

        yield {"event": "deploy_done", "url": url,
               "message": f"🎉 Live at {url}"}


def _make_netlify_toml(cfg: DeployConfig) -> str:
    be  = (cfg.stack.get("backend") or "").lower()
    cmd = "npm run build" if "react" in be or "vue" in be or "next" in be else ""
    pub = "dist" if cmd else "."
    return f"""[build]
  command = "{cmd}"
  publish = "{pub}"

[[redirects]]
  from = "/*"
  to   = "/index.html"
  status = 200
"""


# ═══════════════════════════════════════════════════════════════
# Platform: Render (REST API)
# ═══════════════════════════════════════════════════════════════

async def _deploy_render(cfg: DeployConfig) -> AsyncGenerator[dict, None]:
    token = cfg.render_token
    if not token:
        yield {"event": "error", "message": "Render API key required."}
        return

    yield {"event": "log", "message": "📦 Creating Render service…"}

    be = (cfg.stack.get("backend") or "").lower()
    service_type = "web_service" if "python" in be or "node" in be or "fastapi" in be else "static_site"

    # Render requires a GitHub repo — generate render.yaml instead
    render_yaml = _make_render_yaml(cfg)

    yield {"event": "config_ready",
           "config_files": {"render.yaml": render_yaml},
           "message": (
               "⚠ Render requires a GitHub repository.\n"
               "render.yaml has been generated and added to your project.\n"
               "Push to GitHub, then connect the repo at dashboard.render.com."
           )}

    # Generate deploy button URL for convenience
    yield {"event": "deploy_done",
           "url": "https://dashboard.render.com/new",
           "status": "config_only",
           "message": "render.yaml ready — connect your GitHub repo at dashboard.render.com"}


def _make_render_yaml(cfg: DeployConfig) -> str:
    be   = (cfg.stack.get("backend") or "").lower()
    db   = (cfg.stack.get("database") or "").lower()
    name = _sanitize_name(cfg.project_name)

    services = f"""services:
  - type: web
    name: {name}
    env: {"python" if "python" in be or "fastapi" in be else "node"}
    buildCommand: {"pip install -r requirements.txt" if "python" in be else "npm install && npm run build"}
    startCommand: {"uvicorn main:app --host 0.0.0.0 --port $PORT" if "fastapi" in be else "node server.js"}
    envVars:
      - key: PORT
        value: 10000
"""
    if "postgres" in db or "pg" in db:
        services += """
databases:
  - name: {name}-db
    databaseName: {name}
    user: {name}
""".format(name=name)

    return services


# ═══════════════════════════════════════════════════════════════
# Platform: Fly.io (config generation)
# ═══════════════════════════════════════════════════════════════

async def _generate_fly_config(cfg: DeployConfig) -> AsyncGenerator[dict, None]:
    yield {"event": "log", "message": "📦 Generating Fly.io configuration…"}

    fly_toml    = _make_fly_toml(cfg)
    dockerfile  = _make_dockerfile_if_missing(cfg)
    configs     = {"fly.toml": fly_toml}
    if dockerfile:
        configs["Dockerfile"] = dockerfile

    yield {"event": "log", "message": "✅ fly.toml generated"}
    yield {"event": "config_ready", "config_files": configs,
           "message": (
               "fly.toml added to your project.\n"
               "Deploy with: flyctl launch --copy-config && flyctl deploy"
           )}
    yield {"event": "deploy_done",
           "url": "https://fly.io/docs",
           "status": "config_only",
           "message": "Config ready — run `flyctl deploy` in your project directory"}


def _make_fly_toml(cfg: DeployConfig) -> str:
    name = _sanitize_name(cfg.project_name)
    be   = (cfg.stack.get("backend") or "").lower()
    port = 8000 if "fastapi" in be or "python" in be else 3000
    return f"""app = "{name}"
primary_region = "iad"

[build]

[http_service]
  internal_port = {port}
  force_https   = true
  auto_stop_machines  = true
  auto_start_machines = true

[[vm]]
  memory = "256mb"
  cpu_kind = "shared"
  cpus = 1
"""


def _make_dockerfile_if_missing(cfg: DeployConfig) -> str | None:
    if "Dockerfile" in cfg.file_tree or "dockerfile" in cfg.file_tree:
        return None
    be = (cfg.stack.get("backend") or "").lower()
    if "python" in be or "fastapi" in be:
        return """FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    if "node" in be or "express" in be or "next" in be:
        return """FROM node:20-slim
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
EXPOSE 3000
CMD ["node", "server.js"]
"""
    return None


# ═══════════════════════════════════════════════════════════════
# Platform: Cloudflare Pages (REST API)
# ═══════════════════════════════════════════════════════════════

async def _deploy_cloudflare(cfg: DeployConfig) -> AsyncGenerator[dict, None]:
    token      = cfg.cf_api_token
    account_id = cfg.cf_account_id
    if not token or not account_id:
        yield {"event": "error", "message": "Cloudflare API token and Account ID required."}
        return

    yield {"event": "log", "message": "📦 Creating Cloudflare Pages project…"}

    name = _sanitize_name(cfg.project_name)

    async with httpx.AsyncClient(timeout=60) as client:
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        # Create project
        proj_resp = await client.post(
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}/pages/projects",
            headers=headers,
            json={"name": name, "production_branch": "main"},
        )
        if proj_resp.status_code not in (200, 201):
            yield {"event": "error",
                   "message": f"CF Pages project create error: {proj_resp.text[:300]}"}
            return

        yield {"event": "log", "message": "📤 Uploading files…"}

        # Upload files via multipart
        form = httpx.MultipartUpload
        # Build form data for direct upload
        files_data = {}
        for path, content in cfg.file_tree.items():
            files_data[path] = content

        # CF Pages direct upload (simplified — production should chunk large projects)
        manifest = {p: "" for p in files_data}
        upload_resp = await client.post(
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}/pages/projects/{name}/deployments",
            headers={"Authorization": f"Bearer {token}"},
            files={
                "manifest": ("manifest.json", json.dumps(manifest), "application/json"),
                **{
                    p: (p, c.encode(), "text/plain")
                    for p, c in list(files_data.items())[:200]   # CF limit
                },
            },
        )

        if upload_resp.status_code in (200, 201):
            url = f"https://{name}.pages.dev"
            yield {"event": "deploy_done", "url": url,
                   "message": f"🎉 Live at {url}"}
        else:
            yield {"event": "error",
                   "message": f"CF Pages upload error: {upload_resp.text[:300]}"}


# ═══════════════════════════════════════════════════════════════
# Platform: GitHub Pages (uses existing GitHub token)
# ═══════════════════════════════════════════════════════════════

async def _deploy_github_pages(cfg: DeployConfig) -> AsyncGenerator[dict, None]:
    token    = cfg.github_token
    username = cfg.github_username
    if not token or not username:
        yield {"event": "error", "message": "GitHub token and username required."}
        return

    repo_name = _sanitize_name(cfg.project_name)
    yield {"event": "log", "message": f"📦 Deploying to GitHub Pages ({username}/{repo_name})…"}

    async with httpx.AsyncClient(timeout=60) as client:
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept":        "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        # Ensure repo exists (create if not)
        repo_resp = await client.get(
            f"https://api.github.com/repos/{username}/{repo_name}",
            headers=headers,
        )
        if repo_resp.status_code == 404:
            create_resp = await client.post(
                "https://api.github.com/user/repos",
                headers=headers,
                json={"name": repo_name, "auto_init": True, "private": False},
            )
            if create_resp.status_code not in (200, 201):
                yield {"event": "error", "message": f"Repo create failed: {create_resp.text[:200]}"}
                return
            yield {"event": "log", "message": f"📁 Repository {repo_name} created"}
            await asyncio.sleep(2)  # let GitHub settle

        # Commit all files to gh-pages branch
        files_committed = 0
        for path, content in cfg.file_tree.items():
            b64 = base64.b64encode(content.encode()).decode()
            # Check if file exists (get its SHA)
            file_resp = await client.get(
                f"https://api.github.com/repos/{username}/{repo_name}/contents/{path}",
                headers=headers, params={"ref": "gh-pages"},
            )
            put_body: dict[str, Any] = {
                "message": f"Deploy {path}",
                "content": b64,
                "branch":  "gh-pages",
            }
            if file_resp.status_code == 200:
                put_body["sha"] = file_resp.json().get("sha", "")

            await client.put(
                f"https://api.github.com/repos/{username}/{repo_name}/contents/{path}",
                headers=headers,
                json=put_body,
            )
            files_committed += 1
            if files_committed % 5 == 0:
                yield {"event": "log", "message": f"  Committed {files_committed}/{len(cfg.file_tree)} files…"}

        # Enable GitHub Pages
        await client.post(
            f"https://api.github.com/repos/{username}/{repo_name}/pages",
            headers=headers,
            json={"source": {"branch": "gh-pages", "path": "/"}},
        )

        url = f"https://{username}.github.io/{repo_name}"
        yield {"event": "deploy_done", "url": url,
               "message": f"🎉 Deploying to {url} (may take 1–2 min to go live)"}


# ═══════════════════════════════════════════════════════════════
# Platform: Download (ZIP)
# ═══════════════════════════════════════════════════════════════

async def _prepare_download(cfg: DeployConfig) -> AsyncGenerator[dict, None]:
    """Signal frontend to trigger JSZip download — no server-side needed."""
    yield {"event": "deploy_done",
           "status": "success",
           "url": None,
           "download": True,
           "message": "✅ Use the 'Export .zip' button to download your project."}



# ── Helpers ────────────────────────────────────────────────────

def _sanitize_name(name: str) -> str:
    import re
    slug = re.sub(r"[^a-z0-9\-]", "-", name.lower().strip())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug[:50] or "chiscode-project"
    
