"""
ChisCode — Preview Service (Phase 6)
======================================
Generates a static in-browser preview of the generated project.

HF Spaces constraint: no gVisor, no subprocess execution.
Strategy:
  1. HTML/CSS/JS projects  → serve files directly via a signed preview URL
  2. React/Vue/Next apps   → server-side render an index preview skeleton
  3. Python/Node backends  → generate a visual "architecture card" summary
  4. All projects          → file tree + code stats as a rich preview card

Preview types:
  - "live"    : static HTML served from /preview/{project_id}
  - "card"    : rich metadata card (framework, files, structure)
  - "iframe"  : embed-ready URL returned to frontend
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import PurePosixPath
from typing import Optional

from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import get_logger
from app.db.mongodb import get_db

logger = get_logger(__name__)


# ── Schemas ────────────────────────────────────────────────────

class PreviewInfo(BaseModel):
    project_id:   str
    preview_type: str    # "live" | "card"
    iframe_url:   Optional[str] = None
    card_data:    Optional[dict] = None
    expires_at:   datetime
    file_count:   int
    primary_lang: str
    entry_point:  Optional[str] = None


# ── Preview generation ─────────────────────────────────────────

async def generate_preview(
    project_id: str,
    file_tree:  dict[str, str],
    stack:      dict,
    project_name: str = "",
) -> PreviewInfo:
    """
    Analyse the file tree and generate the best possible preview.
    Stores preview HTML in MongoDB for serving.
    """
    analysis  = _analyse_project(file_tree, stack)
    expires   = datetime.now(tz=timezone.utc) + timedelta(hours=6)

    if analysis["has_html"]:
        # Live preview: inject files into a single-page iframe shell
        preview_html = _build_live_preview(file_tree, analysis, project_name)
        preview_type = "live"
        iframe_url   = f"/api/v1/preview/{project_id}"
        card_data    = None
    else:
        # Rich card preview
        preview_html = None
        preview_type = "card"
        iframe_url   = None
        card_data    = _build_card_data(analysis, project_name, stack)

    # Store in MongoDB (TTL index handles expiry)
    await get_db()["previews"].replace_one(
        {"project_id": project_id},
        {
            "project_id":   project_id,
            "preview_type": preview_type,
            "preview_html": preview_html,
            "card_data":    card_data,
            "analysis":     analysis,
            "expires_at":   expires,
            "created_at":   datetime.now(tz=timezone.utc),
        },
        upsert=True,
    )

    logger.info("Preview generated", project_id=project_id, type=preview_type,
                files=len(file_tree))

    return PreviewInfo(
        project_id=project_id,
        preview_type=preview_type,
        iframe_url=iframe_url,
        card_data=card_data,
        expires_at=expires,
        file_count=len(file_tree),
        primary_lang=analysis["primary_lang"],
        entry_point=analysis.get("entry_point"),
    )


async def get_preview_html(project_id: str) -> str | None:
    """Retrieve stored preview HTML for serving."""
    doc = await get_db()["previews"].find_one(
        {"project_id": project_id, "expires_at": {"$gt": datetime.now(tz=timezone.utc)}}
    )
    return doc.get("preview_html") if doc else None


async def get_preview_card(project_id: str) -> dict | None:
    doc = await get_db()["previews"].find_one({"project_id": project_id})
    return doc.get("card_data") if doc else None


# ── Analysis ──────────────────────────────────────────────────

def _analyse_project(file_tree: dict[str, str], stack: dict) -> dict:
    ext_counts: dict[str, int] = {}
    total_lines = 0

    for path, content in file_tree.items():
        ext = PurePosixPath(path).suffix.lstrip(".").lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
        total_lines += content.count("\n")

    # Detect entry point
    entry = None
    for candidate in ("index.html", "index.js", "main.py", "app.py",
                       "src/index.html", "src/index.jsx", "src/main.tsx",
                       "public/index.html"):
        if candidate in file_tree:
            entry = candidate
            break

    # Primary language
    lang_map = {"py": "Python", "js": "JavaScript", "ts": "TypeScript",
                "jsx": "React", "tsx": "React/TypeScript", "html": "HTML",
                "rs": "Rust", "go": "Go", "rb": "Ruby", "java": "Java"}
    primary_lang = "Unknown"
    for ext, label in lang_map.items():
        if ext_counts.get(ext, 0) > 0:
            primary_lang = label
            break

    has_html = bool(ext_counts.get("html", 0)) or bool(ext_counts.get("htm", 0))

    # Structure detection
    dirs: set[str] = set()
    for path in file_tree:
        parts = path.split("/")
        for i in range(1, len(parts)):
            dirs.add("/".join(parts[:i]))

    return {
        "ext_counts":    ext_counts,
        "total_files":   len(file_tree),
        "total_lines":   total_lines,
        "primary_lang":  primary_lang,
        "has_html":      has_html,
        "entry_point":   entry,
        "directories":   sorted(list(dirs))[:20],
        "has_tests":     any("test" in p.lower() or "spec" in p.lower() for p in file_tree),
        "has_docker":    "Dockerfile" in file_tree,
        "has_readme":    any("readme" in p.lower() for p in file_tree),
        "has_env":       any(".env" in p for p in file_tree),
    }


# ── Live preview builder ───────────────────────────────────────

def _build_live_preview(
    file_tree: dict[str, str],
    analysis:  dict,
    project_name: str,
) -> str:
    """
    Wrap the HTML project in a sandboxed iframe shell.
    Inlines all CSS and JS for static serving.
    """
    entry = analysis.get("entry_point") or _find_html_entry(file_tree)
    if not entry:
        return _fallback_preview_html(analysis, project_name)

    html_content = file_tree.get(entry, "")

    # Inline referenced CSS and JS files
    import re

    def inline_css(m: re.Match) -> str:
        href = m.group(1)
        # Try relative path from entry's directory
        base_dir = "/".join(entry.split("/")[:-1])
        candidates = [
            href.lstrip("/"),
            f"{base_dir}/{href}".lstrip("/"),
            href.lstrip("./"),
        ]
        for candidate in candidates:
            if candidate in file_tree:
                return f"<style>\n{file_tree[candidate]}\n</style>"
        return m.group(0)

    def inline_js(m: re.Match) -> str:
        src = m.group(1)
        # Skip CDN scripts
        if src.startswith("http") or src.startswith("//"):
            return m.group(0)
        base_dir = "/".join(entry.split("/")[:-1])
        candidates = [
            src.lstrip("/"),
            f"{base_dir}/{src}".lstrip("/"),
            src.lstrip("./"),
        ]
        for candidate in candidates:
            if candidate in file_tree:
                return f"<script>\n{file_tree[candidate]}\n</script>"
        return m.group(0)

    html_content = re.sub(
        r'<link[^>]+href=["\']([^"\']+\.css)["\'][^>]*/?>',
        inline_css, html_content
    )
    html_content = re.sub(
        r'<script[^>]+src=["\']([^"\']+\.js)["\'][^>]*></script>',
        inline_js, html_content
    )

    # Inject a tiny banner
    banner = f"""<div style="
        position:fixed;top:0;right:0;z-index:99999;
        background:#07090f;color:#00e5ff;font-family:monospace;
        font-size:11px;padding:4px 10px;border-bottom-left-radius:6px;
        border:1px solid #1e2d45;border-top:none;border-right:none;
        ">ChisCode Preview · {project_name}</div>"""

    if "<body" in html_content.lower():
        html_content = re.sub(
            r"(<body[^>]*>)", r"\1" + banner, html_content, count=1, flags=re.IGNORECASE
        )
    else:
        html_content = banner + html_content

    return html_content


def _find_html_entry(file_tree: dict[str, str]) -> str | None:
    for candidate in ("index.html", "public/index.html", "dist/index.html",
                       "src/index.html", "static/index.html"):
        if candidate in file_tree:
            return candidate
    # Any .html at the root
    for path in sorted(file_tree.keys()):
        if path.endswith(".html") and "/" not in path:
            return path
    return None


def _fallback_preview_html(analysis: dict, project_name: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{project_name} Preview</title>
<style>
  body{{margin:0;font-family:system-ui,sans-serif;background:#07090f;color:#eef4ff;display:flex;align-items:center;justify-content:center;min-height:100vh}}
  .card{{background:#111827;border:1px solid #1e2d45;border-radius:12px;padding:2rem;max-width:480px;text-align:center}}
  .badge{{display:inline-block;background:#1e2d45;border-radius:4px;padding:.2rem .5rem;font-size:.75rem;margin:.25rem}}
</style></head>
<body><div class="card">
  <div style="font-size:2.5rem;margin-bottom:.5rem">🚀</div>
  <h2 style="color:#00e5ff;margin:.5rem 0">{project_name}</h2>
  <p style="color:#6b7f95;font-size:.85rem">This project requires a build step for live preview.</p>
  <p style="margin:1rem 0">
    <span class="badge">{analysis['primary_lang']}</span>
    <span class="badge">{analysis['total_files']} files</span>
    <span class="badge">{analysis['total_lines']:,} lines</span>
  </p>
  <p style="color:#6b7f95;font-size:.8rem">Download the project to run it locally.</p>
</div></body></html>"""


# ── Card data builder ──────────────────────────────────────────

def _build_card_data(analysis: dict, project_name: str, stack: dict) -> dict:
    ext_counts = analysis["ext_counts"]

    # Language breakdown
    lang_map = {"py": "Python", "js": "JavaScript", "ts": "TypeScript",
                "jsx": "React", "tsx": "React/TSX", "html": "HTML",
                "css": "CSS", "json": "JSON", "md": "Markdown",
                "yaml": "YAML", "yml": "YAML", "sql": "SQL",
                "sh": "Shell", "rs": "Rust", "go": "Go"}
    languages = [
        {"name": lang_map.get(ext, ext.upper()), "files": count}
        for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1])
        if count > 0 and ext in lang_map
    ][:6]

    features = []
    if analysis["has_tests"]:   features.append("Tests included")
    if analysis["has_docker"]:  features.append("Docker ready")
    if analysis["has_readme"]:  features.append("README included")
    if analysis["has_env"]:     features.append("Env config")

    return {
        "name":       project_name,
        "stack":      stack,
        "stats": {
            "files":       analysis["total_files"],
            "lines":       analysis["total_lines"],
            "directories": len(analysis["directories"]),
        },
        "languages":  languages,
        "features":   features,
        "directories": analysis["directories"][:10],
        "primary_lang": analysis["primary_lang"],
    }
    
