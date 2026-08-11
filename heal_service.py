It's like this heal service is over engineered,it all most stripped all content when trying to fix the depend issue,let me provide the file so to know what each function does and to know what is actually needed and which should be removed or modified,
"""
ChisCode — Preview Self-Heal Service
======================================
ADK-powered self-healing pipeline for E2B sandbox previews.

Architecture (ADK Phases A + B + C)
--------------------------------------
Phase A — install_and_heal() delegates to an ADK LoopAgent pipeline:
    InstallRunnerAgent  → DevServerAgent → ProbeAgent
    → HealExitChecker  → [fixer block]  → ApplyOrRetryAgent

Phase B — Generator-Critic-Checker replaces the two-shot plan+fix pattern:
    HealFixerAgent (LlmAgent) — classifies errors + proposes fixes (no hardcoded enum)
    HealCriticAgent (LlmAgent) — reflects on fix before it is uploaded
    ApplyOrRetryAgent (BaseAgent) — applies only critic-approved fixes

Phase C — ParallelFileFixAgent wraps Fixer + Critic in a ParallelAgent
    so multiple broken files are fixed concurrently.

Bug fixes applied
------------------
BUG1 — files_to_fix never validated against file_tree keys (phantom fixes)
BUG2 — probe_live_url only tried once; now uses a mini 2-iteration LoopAgent
BUG3 — START_WAIT fixed sleep replaced by _wait_for_server() poll loop

Session state keys (heal:*)
-----------------------------
heal:file_tree       — working file tree (replaces current_tree dict mutation)
heal:error_lines     — errors from install or server log
heal:stack_logs      — raw log tail for LLM context
heal:probe_errors    — classified errors from localhost probe
heal:probe_status    — HTTP status int from probe
heal:proposed_fixes  — JSON str: {filepath: fixed_content} from HealFixer
heal:critique        — JSON str: {approved, reason} from HealCritic
heal:done            — bool: HealExitChecker sets True to escalate loop
heal:stack           — stack dict
heal:start_cmd       — dev server command string
heal:port            — port int
heal:sandbox_id      — sandbox ID for logging
heal:project_id      — project ID for Phoenix spans
heal:is_node         — bool: project has package.json
heal:is_python       — bool: project has requirements.txt
heal:has_node        — bool: node binary available in sandbox
heal:stack_desc      — human-readable stack string
heal:attempt         — current iteration number (informational)
"""
from __future__ import annotations
import os
from pathlib import Path
from pydantic import BaseModel  # ← ADD THIS
import asyncio
import json
import re
import time
from typing import TYPE_CHECKING, AsyncGenerator

from app.core.config  import settings
from app.core.logging import get_logger
from app.core.monitoring import (
    heal_span, probe_span, sandbox_span,
    record_probe_result, record_heal_result,
    get_tracer, OTEL_ATTRS,
)

# ── ADK ───────────────────────────────────────────────────────────────────────
from google.adk.agents import (
    BaseAgent,
    LlmAgent,
    LoopAgent,
    ParallelAgent,
)
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types
from typing_extensions import override
# Add these imports
from app.services.skill_trajectory_service import (
    log_heal_trajectory,
    log_failure_signature,
    log_skill_snapshot,
    log_generation_outcome,
    log_skillopt_training_result,
)
from app.services.skill_optimizer_service import SkillOptimizer
# ─────────────────────────────────────────────────────────────────────────────

# ── NEW: Tenacity, Instructor, Tree-sitter imports ──────────────────────────
from app.core.retry import _call_tool_with_retry
from app.llm.structured import (
    generate_structured,
    HealFixResult,
    FileEditAction,
)
from app.parser.tree_sitter_wrapper import get_parser
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.parser.ast_grep_wrapper import get_ast_grep
from app.llm.token_compactor import get_token_compactor, get_context_manager
from app.llm.tool_schemas import ToolCallValidator

if TYPE_CHECKING:
    from e2b import Sandbox

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_ATTEMPTS         = 3
INSTALL_TIMEOUT      = 180
SERVER_POLL_INTERVAL = 2      # seconds between readiness polls  (BUG3)
SERVER_POLL_MAX      = 30     # maximum seconds to wait for server (BUG3)
_HEAL_SKILL_CACHE: str | None = None
_HEAL_SKILL_VERSION: str = "v1.0.0"


def _make_codestral_model():
    """
    Returns the model identifier for Codestral LlmAgent calls.
    Uses the "codestral/" LiteLLM provider prefix so requests are
    routed to codestral.mistral.ai with the correct API key.
    Set env vars before startup:
      CODESTRAL_API_KEY  = settings.codestral_api_key
      CODESTRAL_API_BASE = settings.codestral_base_url
    LiteLLM reads these automatically from the environment.
    """
    import os
    os.environ.setdefault("CODESTRAL_API_KEY",
                          getattr(settings, "codestral_api_key", ""))
    os.environ.setdefault("CODESTRAL_API_BASE",
                          getattr(settings, "codestral_base_url",
                                  "https://codestral.mistral.ai/v1"))
    return "codestral/codestral-latest"

# ── Sandbox reference registry ────────────────────────────────────────────────
# ADK agents are pure-Python dataclasses — they cannot hold live E2B Sandbox
# objects as fields (not JSON-serialisable).  We park the sandbox in a
# module-level dict keyed by sandbox_id and look it up inside each agent.
_SANDBOX_REGISTRY: dict[str, "Sandbox"] = {}


def _register_sandbox(sandbox: "Sandbox") -> None:
    _SANDBOX_REGISTRY[sandbox.sandbox_id] = sandbox


def _get_sandbox(sandbox_id: str) -> "Sandbox":
    sb = _SANDBOX_REGISTRY.get(sandbox_id)
    if sb is None:
        raise RuntimeError(f"Sandbox {sandbox_id} not in registry")
    return sb


def _deregister_sandbox(sandbox_id: str) -> None:
    _SANDBOX_REGISTRY.pop(sandbox_id, None)


def _load_heal_skill() -> str:
    """Load the SkillOpt-optimized heal skill, if one exists on disk."""
    global _HEAL_SKILL_CACHE, _HEAL_SKILL_VERSION

    if os.getenv("SKILL_REFRESH"):
        _HEAL_SKILL_CACHE = None

    if _HEAL_SKILL_CACHE is None:
        skill_dir = Path(__file__).parent.parent / "agents" / "skills"
        optimized = skill_dir / "chiscode_heal_skill_optimized.md"
        base = skill_dir / "chiscode_heal_skill.md"

        if optimized.exists():
            _HEAL_SKILL_CACHE = optimized.read_text()
            _HEAL_SKILL_VERSION = _extract_heal_version(_HEAL_SKILL_CACHE)
        elif base.exists():
            _HEAL_SKILL_CACHE = base.read_text()
            _HEAL_SKILL_VERSION = "v1.0.0"
        else:
            _HEAL_SKILL_CACHE = ""  # nothing to add — existing prompts still work fine
            _HEAL_SKILL_VERSION = "none"

    return _HEAL_SKILL_CACHE


def _extract_heal_version(skill_content: str) -> str:
    for line in skill_content.splitlines():
        if "Version:" in line:
            return line.split("Version:")[-1].strip()
    return "v1.0.0"

def _categorize_error(error: str) -> str:
    """Categorize an error for pattern mining."""
    error_lower = error.lower()
    
    if "validationerror" in error_lower or "field required" in error_lower:
        return "config"
    elif "modulenotfound" in error_lower or "no module named" in error_lower:
        return "build"
    elif "import" in error_lower and "error" in error_lower:
        return "build"
    elif "timeout" in error_lower or "connection refused" in error_lower:
        return "network"
    elif "localstorage" in error_lower or "document" in error_lower:
        return "runtime"
    elif "syntaxerror" in error_lower or "indentation" in error_lower:
        return "syntax"
    elif "command not found" in error_lower:
        return "environment"
    else:
        return "unknown"
# ═══════════════════════════════════════════════════════════════════════════════
# Error patterns  (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════════

SERVER_ERROR_PATTERNS = [
    r"ERROR\s", r"error TS\d+", r"SyntaxError",
    r"Cannot find module", r"Module not found", r"ENOENT",
    r"EADDRINUSE", r"npm ERR!", r"failed to compile",
    r"Build failed", r"error\[E", r"ImportError",
    r"ModuleNotFoundError", r"Failed to load url",
    r"Could not resolve", r"does not provide an export",
    r"command not found", r"No such file or directory",
    r"Permission denied",
]

BROWSER_ERROR_PATTERNS = [
    r"ReferenceError", r"TypeError", r"SyntaxError",
    r"is not defined", r"is not a function",
    r"Cannot read propert", r"500\s*Internal",
    r"Internal Error", r"<h1>500</h1>",
    r"vite-error-overlay", r"Failed to fetch",
    r"NetworkError", r"net::ERR_",
]

SERVER_ERROR_RE  = re.compile("|".join(SERVER_ERROR_PATTERNS),  re.IGNORECASE)
BROWSER_ERROR_RE = re.compile("|".join(BROWSER_ERROR_PATTERNS), re.IGNORECASE)


# ═══════════════════════════════════════════════════════════════════════════════
# Exception
# ═══════════════════════════════════════════════════════════════════════════════

class PreviewHealError(Exception):
    def __init__(self, message: str, partial_tree: dict[str, str] | None = None):
        super().__init__(message)
        self.partial_tree = partial_tree or {}

# ═══════════════════════════════════════════════════════════════════════════════
# Pure helper functions  (unchanged from original — no ADK dependency)
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_errors(
    log: str,
    pattern: re.Pattern,
    max_lines: int = 20,
) -> list[str]:
    errors = []
    for line in log.splitlines():
        if "Requires-Python" in line and "Ignored" in line:
            continue
        if "notice" in line.lower():
            continue
        if pattern.search(line):
            errors.append(line.strip())
        if len(errors) >= max_lines:
            break
    return errors


def _read_log(sandbox: "Sandbox", timeout: int = 5) -> str:
    try:
        result = sandbox.commands.run(
            "cat /tmp/app.log 2>/dev/null | tail -100 || echo ''",
            timeout=timeout, user="user",
        )
        return result.stdout or ""
    except Exception:
        return ""


def _read_full_log(sandbox: "Sandbox", timeout: int = 5) -> str:
    """Read the full app.log without truncation (last 200 lines)."""
    try:
        result = sandbox.commands.run(
            "cat /tmp/app.log 2>/dev/null | tail -200 || echo ''",
            timeout=timeout, user="user",
        )
        return result.stdout or ""
    except Exception:
        return ""


def _read_stderr(sandbox: "Sandbox", timeout: int = 5) -> str:
    """Read the stderr log for silent crashes."""
    try:
        result = sandbox.commands.run(
            "cat /tmp/app.err 2>/dev/null | tail -100 || echo ''",
            timeout=timeout, user="user",
        )
        return result.stdout or ""
    except Exception:
        return ""


def _extract_validation_error(log: str) -> str:
    """
    Extract the full Pydantic ValidationError traceback including
    the field-level details that say which env vars are missing.
    """
    lines = log.splitlines()
    capture = False
    captured = []
    for line in lines:
        if ("ValidationError" in line or "validation error" in line
                or "pydantic_core" in line):
            capture = True
        if capture:
            captured.append(line)
            # Stop at the next blank line after capturing details
            if len(captured) > 2 and line.strip() == "":
                break
            # Also stop at a new traceback
            if "Traceback" in line and len(captured) > 3:
                break
    return "\n".join(captured) if captured else ""

def _inject_missing_env_fields(sandbox: "Sandbox", fields: list[str]) -> None:
    """Read .env, append missing fields, preserve ALL existing fields."""
    try:
        r = sandbox.commands.run(
            "cat /home/user/.env 2>/dev/null || echo ''", timeout=5, user="user")
        current = r.stdout or ""
    except Exception:
        current = ""

    # Parse existing fields to preserve them exactly
    existing_lines = current.splitlines()
    existing_keys = set()
    preserved_lines = []
    for line in existing_lines:
        if "=" in line and not line.startswith("#"):
            key = line.split("=")[0].strip()
            existing_keys.add(key)
            preserved_lines.append(line)
        elif line.strip():
            preserved_lines.append(line)

    FIELD_DEFAULTS = {
        "DATABASE_URL":          "sqlite:///./preview.db",
        "SECRET_KEY":            "changeme-preview-secret",
        "ALGORITHM":             "HS256",
        "ALLOWED_ORIGINS":       '["http://localhost:8000"]',
        "ACCESS_TOKEN_EXPIRE_MINUTES": "30",
        "DEBUG":                 "false",
        "APP_NAME":              "preview-app",
    }

    additions = []
    for field in fields:
        if field not in existing_keys:
            default = FIELD_DEFAULTS.get(field, "placeholder")
            additions.append(f"{field}={default}")

    if not additions:
        return

    # Preserve all existing lines + append new ones
    new_env = "\n".join(preserved_lines).rstrip() + "\n" + "\n".join(additions) + "\n"
    sandbox.files.write("/home/user/.env", new_env)
    logger.info("_inject_missing_env_fields: appended to .env",
                added=additions, preserved=list(existing_keys))
    
def _restart_dev_server(
    sandbox:   "Sandbox",
    start_cmd: str,
    port:      int,
) -> None:
    """Kill all processes on the target port and restart the dev server."""
    try:
        # STEP 1: Aggressively kill all processes on the port
        # Multiple methods to ensure cleanup
        kill_commands = [
            # Kill by port using fuser
            f"fuser -k {port}/tcp 2>/dev/null || true",
            # Kill node processes specifically
            "pkill -9 -f 'node' 2>/dev/null || true",
            "pkill -9 -f 'next' 2>/dev/null || true", 
            "pkill -9 -f 'vite' 2>/dev/null || true",
            "pkill -9 -f 'webpack' 2>/dev/null || true",
            "pkill -9 -f 'react-scripts' 2>/dev/null || true",
            # Kill by port using lsof if available
            "lsof -ti:{port} | xargs kill -9 2>/dev/null || true",
            # Alternative: find and kill by port using netstat
            "netstat -tlnp 2>/dev/null | grep ':{port} ' | awk '{{print $7}}' | cut -d'/' -f1 | xargs kill -9 2>/dev/null || true",
        ]
        
        for cmd in kill_commands:
            try:
                sandbox.commands.run(cmd, timeout=5, user="user")
            except Exception:
                pass
        
        # STEP 2: Wait for port to be released
        time.sleep(3)
        
        # STEP 3: Verify port is actually free
        check_port = sandbox.commands.run(
            f"netstat -tlnp 2>/dev/null | grep ':{port} ' || echo 'port-free'",
            timeout=5, user="user"
        )
        if 'port-free' not in (check_port.stdout or ""):
            logger.warning(f"Port {port} still in use after kill attempts")
            # Force kill with SIGKILL on specific PID
            get_pid = sandbox.commands.run(
                f"netstat -tlnp 2>/dev/null | grep ':{port} ' | awk '{{print $7}}' | cut -d'/' -f1",
                timeout=5, user="user"
            )
            if get_pid.stdout and get_pid.stdout.strip():
                pid = get_pid.stdout.strip()
                sandbox.commands.run(f"kill -9 {pid} 2>/dev/null || true", timeout=5, user="user")
                time.sleep(1)
        
        # STEP 4: Clear log file
        sandbox.commands.run(
            "echo '' > /tmp/app.log", timeout=5, user="user"
        )
        
    except Exception as exc:
        logger.warning(f"Error during port cleanup: {exc}")
    
    # STEP 5: Start fresh dev server
    safe_cmd = start_cmd.replace("'", '"').replace("cd /home/user && ", "")
    dev_only = (
        safe_cmd
        .replace("npm install && ", "")
        .replace("pip install -r requirements.txt && ", "")
    )
    
    try:
        sandbox.commands.run(
            f'bash -c "cd /home/user && {dev_only} > /tmp/app.log 2>&1 &"',
            background=True, user="user",
        )
        logger.info(f"Dev server restarted on port {port}")
    except RuntimeError as exc:
        if "streaming response" not in str(exc).lower():
            raise

# ── BUG3 FIX — poll-based server readiness (replaces fixed START_WAIT sleep) ─

async def _wait_for_server(
    sandbox:   "Sandbox",
    port:      int,
    has_node:  bool = True,
    max_wait:  int  = SERVER_POLL_MAX,
    interval:  int  = SERVER_POLL_INTERVAL,
) -> bool:
    """
    Poll localhost:port until the server responds with any HTTP status,
    or until max_wait seconds elapse.  Returns True if server is up.
    Replaces time.sleep(START_WAIT) throughout the heal pipeline.
    """
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        await asyncio.sleep(interval)
        probe = _probe_localhost(sandbox, port, has_node=has_node)
        status = probe.get("status", 0)
        # Any non-zero status means the server is accepting connections
        if status not in (0,):
            logger.info("Server ready", port=port, status=status)
            return True
        logger.debug("Waiting for server", port=port, elapsed=round(
            max_wait - (deadline - time.monotonic()), 1))
    logger.warning("Server did not become ready in time", port=port,
                   max_wait=max_wait)
    return False


# ── Probe helpers  (unchanged from original) ──────────────────────────────────

def _probe_url_in_sandbox(sandbox: "Sandbox", url: str) -> dict:
    """Node .cjs probe — works in ESM projects."""
    probe_script = """
const http = require('http');
const https = require('https');
const url = '__PROBE_URL__';
const isHttps = url.startsWith('https');
const lib = isHttps ? https : http;
const req = lib.get(url, { timeout: 8000 }, (res) => {
  let body = '';
  res.on('data', chunk => body += chunk);
  res.on('end', () => {
    const result = {
      status: res.statusCode,
      errors: [],
      network_errors: [],
      body_excerpt: body.slice(0, 2000),
    };
    const stackMatch = body.match(/<pre[^>]*>([\\s\\S]*?)<\\/pre>/i);
    const msgMatch = body.match(/(?:Error|TypeError|ReferenceError)[:\\s]+([^\\n<]{10,200})/i);
    if (stackMatch) {
      const cleaned = stackMatch[1].replace(/<[^>]+>/g, '').trim().slice(0, 300);
      if (cleaned) result.errors.push('Stack: ' + cleaned);
    } else if (msgMatch) {
      result.errors.push(msgMatch[0].trim().slice(0, 200));
    }
    if (res.statusCode >= 500) {
      result.errors.push('HTTP ' + res.statusCode + ' server error');
    }
    process.stdout.write(JSON.stringify(result) + '\\n');
    process.exit(0);
  });
});
req.on('timeout', () => {
  process.stdout.write(JSON.stringify({
    status: 0, errors: [],
    network_errors: ['TIMEOUT'], body_excerpt: ''
  }) + '\\n');
  req.destroy(); process.exit(0);
});
req.on('error', (err) => {
  process.stdout.write(JSON.stringify({
    status: 0, errors: [],
    network_errors: [err.code + ': ' + err.message], body_excerpt: ''
  }) + '\\n');
  process.exit(0);
});
""".replace("__PROBE_URL__", url)

    try:
        sandbox.files.write("/home/user/probe.cjs", probe_script)
        result = sandbox.commands.run(
            "node /home/user/probe.cjs", timeout=15, user="user",
        )
        if result.stderr and result.stderr.strip():
            logger.warning("Probe stderr",
                           stderr=result.stderr.strip()[:200])
        output = (result.stdout or "").strip()
        if not output:
            return {"status": 200, "errors": [], "network_errors": [],
                    "body_excerpt": ""}
        return json.loads(output)
    except Exception as exc:
        logger.warning("Probe exception — skipping", error=str(exc)[:200])
        return {"status": 200, "errors": [], "network_errors": [],
                "body_excerpt": ""}


def _probe_with_python(sandbox: "Sandbox", url: str) -> dict:
    """Python urllib probe — for templates without Node."""
    probe_script = f"""
import urllib.request, json, sys
url = '{url}'
try:
    req = urllib.request.urlopen(url, timeout=8)
    body = req.read(2000).decode('utf-8', errors='replace')
    result = {{'status': req.status, 'errors': [], 'network_errors': [], 'body_excerpt': body}}
    if req.status >= 500:
        result['errors'].append('HTTP ' + str(req.status) + ' server error')
except urllib.error.HTTPError as e:
    body = e.read(1000).decode('utf-8', errors='replace')
    result = {{'status': e.code, 'errors': ['HTTP ' + str(e.code)],
               'network_errors': [], 'body_excerpt': body}}
except Exception as e:
    result = {{'status': 0, 'errors': [],
               'network_errors': [str(e)], 'body_excerpt': ''}}
print(json.dumps(result))
"""
    try:
        sandbox.files.write("/home/user/probe.py", probe_script)
        result = sandbox.commands.run(
            "python3 /home/user/probe.py", timeout=15, user="user",
        )
        output = (result.stdout or "").strip()
        if not output:
            return {"status": 200, "errors": [], "network_errors": [],
                    "body_excerpt": ""}
        return json.loads(output)
    except Exception as exc:
        logger.warning("Python probe exception", error=str(exc)[:200])
        return {"status": 200, "errors": [], "network_errors": [],
                "body_excerpt": ""}


def _probe_localhost(
    sandbox: "Sandbox", port: int, has_node: bool = True
) -> dict:
    url = f"http://localhost:{port}"
    return (_probe_url_in_sandbox(sandbox, url)
            if has_node else _probe_with_python(sandbox, url))


def _probe_external_url(
    sandbox: "Sandbox", url: str, has_node: bool = True
) -> dict:
    return (_probe_url_in_sandbox(sandbox, url)
            if has_node else _probe_with_python(sandbox, url))


def _detect_actual_port(sandbox: "Sandbox", expected_port: int) -> int:
    try:
        result = sandbox.commands.run(
            "grep -oP '(?<=localhost:)\\d+' /tmp/app.log | tail -1",
            timeout=5, user="user",
        )
        detected = (result.stdout or "").strip()
        if detected and detected.isdigit():
            return int(detected)
    except Exception:
        pass
    return expected_port


def _classify_probe_errors(probe: dict) -> list[str]:
    errors: list[str] = []
    for e in probe.get("errors", []):
        if "<" in e and ">" in e and len(e) > 100:
            clean = re.sub(r"<[^>]+>", "", e).strip()
            if clean and len(clean) > 5:
                errors.append(clean[:200])
        else:
            errors.append(e)
    for e in probe.get("network_errors", []):
        if e.strip() in ("ECONNREFUSED: ", "ECONNREFUSED", "ECONNREFUSED:"):
            continue
        errors.append(e)
    body = probe.get("body_excerpt", "")
    if body:
        for line in body.splitlines():
            clean = re.sub(r"<[^>]+>", "", line).strip()
            if clean and BROWSER_ERROR_RE.search(clean) and len(clean) < 200:
                if clean not in errors:
                    errors.append(clean)
                if len(errors) >= 5:
                    break
    return [e for e in errors if e and len(e) > 3]


def _upload_fixed_files(
    sandbox: "Sandbox", fixed: dict[str, str]
) -> None:
    ast_grep = get_ast_grep()
    for filepath, content in fixed.items():
        try:
            # ── NEW: Apply ast-grep fixes before upload ──────────────────
            if filepath.endswith(".py") and content:
                content = ast_grep.fix_all(content, filepath)
                logger.debug(f"ast-grep applied to {filepath} before upload")
            
            # ── NEW: Validate with Tree-sitter before upload ──────────────
            parser = get_parser()
            is_valid, errors = parser.validate_syntax(filepath, content)
            if not is_valid:
               logger.warning(f"File {filepath} has syntax errors before upload: {errors[:3]}")
               # Still upload it, but log the warning
           
            sandbox.files.write(f"/home/user/{filepath}", content)
            logger.info("Fixed file uploaded", path=filepath)
        except Exception as exc:
            logger.warning("Fixed file upload failed",
                           path=filepath, error=str(exc))


# ═══════════════════════════════════════════════════════════════════════════════
# ADK Phase A — LoopAgent sub-agents
# ═══════════════════════════════════════════════════════════════════════════════

class InstallRunnerAgent(BaseAgent):
    """
    Iteration step 1 — runs npm install or pip install inside the sandbox.
    Writes:
      heal:error_lines  — install errors (filtered)
      heal:stack_logs   — raw log tail for LLM context
    Handles the bad-package auto-strip logic for pip (unchanged from original).
    """

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        sandbox_id  = ctx.session.state["heal:sandbox_id"]
        project_id  = ctx.session.state["heal:project_id"]
        is_node     = ctx.session.state["heal:is_node"]
        is_python   = ctx.session.state["heal:is_python"]
        attempt     = ctx.session.state.get("heal:attempt", 1)

        sandbox     = _get_sandbox(sandbox_id)
        install_log = ""
        error_lines: list[str] = []

        if is_node:
            logger.info("npm install", sandbox_id=sandbox_id, attempt=attempt)
            with sandbox_span("npm_install", project_id=project_id,
                              sandbox_id=sandbox_id):
                try:
                    result = sandbox.commands.run(
                        "cd /home/user && npm install 2>&1",
                        timeout=INSTALL_TIMEOUT, user="user",
                    )
                    install_log = result.stdout or ""
                    logger.info("npm install complete",
                                exit_code=result.exit_code,
                                tail=install_log[-400:])
                    if result.exit_code != 0:
                        logger.warning("npm install failed",
                                       log=install_log[-400:])
                except Exception as exc:
                    err_str = str(exc)
                    if ("context deadline exceeded" in err_str
                            or "timeoutMs" in err_str):
                        logger.warning("npm install timed out")
                    else:
                        install_log = err_str

        elif is_python:
            logger.info("pip install", sandbox_id=sandbox_id, attempt=attempt)
            with sandbox_span("pip_install", project_id=project_id,
                              sandbox_id=sandbox_id):
                try:
                    req_r = sandbox.commands.run(
                        "cat /home/user/requirements.txt 2>/dev/null | head -30",
                        timeout=5, user="user",
                    )
                    logger.info("requirements.txt",
                                content=(req_r.stdout or "")[:500])

                    result = sandbox.commands.run(
                        "cd /home/user && pip install -r requirements.txt 2>&1",
                        timeout=INSTALL_TIMEOUT, user="user",
                    )
                    install_log = (result.stdout or "") + "\n" + (result.stderr or "")
                    logger.info("pip install complete",
                                exit_code=result.exit_code,
                                tail=install_log[-500:])

                    if result.exit_code != 0:
                        # BUG1-adjacent: auto-strip packages that don't exist
                        no_dist = re.findall(
                            r"No matching distribution found for ([^\s(]+)",
                            install_log,
                        )
                        if no_dist:
                            logger.warning("Removing bad packages",
                                           packages=no_dist)
                            req_r2 = sandbox.commands.run(
                                "cat /home/user/requirements.txt",
                                timeout=5, user="user",
                            )
                            req_content = req_r2.stdout or ""
                            cleaned = [
                                ln for ln in req_content.splitlines()
                                if not any(
                                    bad.split("==")[0].lower() in ln.lower()
                                    for bad in no_dist
                                )
                            ]
                            sandbox.files.write(
                                "/home/user/requirements.txt",
                                "\n".join(cleaned),
                            )
                            retry = sandbox.commands.run(
                                "cd /home/user && pip install "
                                "-r requirements.txt 2>&1",
                                timeout=INSTALL_TIMEOUT, user="user",
                            )
                            install_log = retry.stdout or ""
                            logger.info("pip retry",
                                        exit_code=retry.exit_code,
                                        tail=install_log[-400:])

                except Exception as exc:
                    install_log = str(exc)
                    logger.warning("pip install error", error=str(exc)[:200])

        # Filter noise — same logic as original
        raw_errors = _extract_errors(install_log, SERVER_ERROR_RE)
        error_lines = [
            e for e in raw_errors
            if (len(e.strip()) > 10
                and "Requires-Python" not in e
                and "notice" not in e.lower())
        ]

        if error_lines:
            logger.warning("Install errors", attempt=attempt,
                           errors=error_lines[:5])

        yield Event(
            author=self.name,
            actions=EventActions(state_delta={
                "heal:error_lines": error_lines,
                "heal:stack_logs":  install_log[-600:],
                "heal:attempt":     attempt,
            }),
            content=genai_types.Content(parts=[genai_types.Part(
                text=f"install: {len(error_lines)} error(s)"
            )]),
        )


class DevServerAgent(BaseAgent):
    """
    Iteration step 2 — starts the dev server when install had no errors.
    Uses _wait_for_server() poll loop instead of fixed sleep (BUG3 fix).
    Writes additional server-log errors into heal:error_lines.

    V4: Captures full app.log (not truncated), extracts ValidationError
    field-level details, and merges stderr for silent crashes.
   
   MODIFIED: Uses Tree-sitter to validate package.json/requirements.txt
    """

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        sandbox_id  = ctx.session.state["heal:sandbox_id"]
        project_id  = ctx.session.state["heal:project_id"]
        start_cmd   = ctx.session.state["heal:start_cmd"]
        port        = ctx.session.state["heal:port"]
        has_node    = ctx.session.state["heal:has_node"]
        is_node     = ctx.session.state.get("heal:is_node", False)  # ← ADD THIS
        error_lines = ctx.session.state.get("heal:error_lines", [])
        attempt     = ctx.session.state.get("heal:attempt", 1)

        sandbox     = _get_sandbox(sandbox_id)
       
        # ── NEW: Validate package.json with Tree-sitter ────────────────
        try:
            parser = get_parser()
            if is_node:
                pkg_content = sandbox.commands.run(
                    "cat /home/user/package.json 2>/dev/null || echo '{}'",
                    timeout=5, user="user"
                ).stdout or "{}"
                is_valid, errors = parser.validate_syntax("package.json", pkg_content)
                if not is_valid:
                    logger.warning(f"package.json has syntax errors: {errors[:3]}")
                    # Try to fix with Instructor
                    try:
                        from app.llm.structured import generate_structured
                        class PackageJsonFix(BaseModel):
                            content: str
                            errors_fixed: list[str]
                        
                        fixed = await generate_structured(
                            prompt=f"Fix this invalid package.json:\n{pkg_content[:2000]}",
                            response_model=PackageJsonFix,
                            system_prompt="Fix JSON syntax errors. Return valid JSON only.",
                        )
                        if fixed.content:
                            sandbox.files.write("/home/user/package.json", fixed.content)
                            logger.info("Fixed package.json with Instructor")
                    except Exception as exc:
                        logger.warning(f"Failed to fix package.json: {exc}")
        except Exception as exc:
            logger.debug(f"Tree-sitter validation failed: {exc}")

        # Skip server start if install already failed
        if error_lines:
            yield Event(
                author=self.name,
                actions=EventActions(state_delta={}),
                content=genai_types.Content(parts=[genai_types.Part(
                    text="skipping server start — install errors present"
                )]),
            )
            return

        dev_only_cmd = (
            start_cmd.replace("'", '"')
            .replace("cd /home/user && ", "")
            .replace("npm install && ", "")
            .replace("pip install -r requirements.txt && ", "")
        )


        logger.info("Starting dev server",
                    sandbox_id=sandbox_id, cmd=dev_only_cmd, attempt=attempt)

        with sandbox_span("dev_server_start", project_id=project_id,
                          sandbox_id=sandbox_id):
            try:
                # ── V4: Capture BOTH stdout and stderr ──────────────────
                sandbox.commands.run(
                    f'bash -c "cd /home/user && {dev_only_cmd} '
                    f'> /tmp/app.log 2> /tmp/app.err &"',
                    background=True, user="user",
                )
            except RuntimeError as exc:
                if "streaming response" not in str(exc).lower():
                    raise

        # BUG3 FIX — poll instead of fixed sleep
        await _wait_for_server(sandbox, port, has_node=has_node)

        # ── V4: Read FULL log (not truncated) + stderr ──────────────
        app_log = _read_full_log(sandbox)
        app_err = _read_stderr(sandbox)

        # Merge stderr into app_log if stdout is empty or short
        if (not app_log or len(app_log.strip()) < 50) and app_err:
            app_log = app_log + "\n--- STDERR ---\n" + app_err
            logger.info("Merged stderr into app_log",
                        stderr_lines=len(app_err.splitlines()))

        # Handle "command not found" (e.g. uvicorn missing)
        if "command not found" in app_log:
            missing_m = re.search(r"(\w+): command not found", app_log)
            if missing_m:
                cmd_map = {"uvicorn": "uvicorn[standard]",
                           "gunicorn": "gunicorn"}
                pkg = cmd_map.get(missing_m.group(1))
                if pkg:
                    sandbox.commands.run(
                        f"pip install {pkg} 2>&1",
                        timeout=60, user="user",
                    )
                    _restart_dev_server(sandbox, start_cmd, port)
                    await _wait_for_server(sandbox, port, has_node=has_node)
                    app_log = _read_full_log(sandbox)

        # ── Extract errors via standard patterns ──────────────────
        server_errors = _extract_errors(app_log, SERVER_ERROR_RE)
        server_errors = [
            e for e in server_errors
            if len(e.strip()) > 10 and "notice" not in e.lower()
        ]

        # ── V4: Catch Python exceptions not matched by SERVER_ERROR_RE ──
        if not server_errors:
            for line in app_log.splitlines():
                stripped = line.strip()
                if any(x in stripped for x in (
                    "Error:", "Exception:", "Traceback (most recent",
                    "ImportError", "ModuleNotFoundError", "ValidationError",
                    "Connection refused", "Address already in use",
                    "cannot import", "No module named",
                    "pydantic_core", "SyntaxError", "AttributeError",
                    "KeyError", "TypeError", "ValueError",
                )):
                    if len(stripped) > 10:
                        server_errors.append(stripped)

        validation_detail = _extract_validation_error(app_log)
        if validation_detail:
            logger.info("ValidationError detail", detail=validation_detail[:500])
            server_errors.insert(0, f"VALIDATION_DETAIL: {validation_detail[:800]}")

        # ── NEW: parse missing fields and inject them into .env immediately ──
        missing_fields = re.findall(
            r'^([A-Z][A-Z0-9_]{2,})\s*\n\s+Field required',
            validation_detail,
            re.MULTILINE,
        )
        if missing_fields:
            _inject_missing_env_fields(sandbox, missing_fields)
            logger.info("Auto-injected missing env fields",
                        fields=missing_fields)
        
        if server_errors:
            logger.warning("Server log errors",
                           attempt=attempt, errors=server_errors[:5])

        yield Event(
            author=self.name,
            actions=EventActions(state_delta={
                "heal:error_lines": server_errors,
                "heal:stack_logs":  app_log[-3000:],  # V4: more context
            }),
            content=genai_types.Content(parts=[genai_types.Part(
                text=f"server: {len(server_errors)} error(s)"
            )]),
        )

class ProbeAgent(BaseAgent):
    """
    Iteration step 3 — probes localhost and writes probe results to state.
    Phoenix probe_span fired here.
    """

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        sandbox_id  = ctx.session.state["heal:sandbox_id"]
        project_id  = ctx.session.state["heal:project_id"]
        port        = ctx.session.state["heal:port"]
        has_node    = ctx.session.state["heal:has_node"]
        error_lines = ctx.session.state.get("heal:error_lines", [])
        attempt     = ctx.session.state.get("heal:attempt", 1)

        sandbox     = _get_sandbox(sandbox_id)

        # Skip probe if upstream steps already found errors
        if error_lines:
            yield Event(
                author=self.name,
                actions=EventActions(state_delta={
                    "heal:probe_errors": [],
                    "heal:probe_status": 0,
                }),
                content=genai_types.Content(parts=[genai_types.Part(
                    text="skipping probe — upstream errors present"
                )]),
            )
            return

        actual_port = _detect_actual_port(sandbox, port)

        with probe_span("localhost", project_id=project_id,
                        sandbox_id=sandbox_id,
                        port=actual_port) as p_span:
            probe        = _probe_localhost(sandbox, actual_port, has_node)
            probe_errors = _classify_probe_errors(probe)
            record_probe_result(p_span, probe)

        logger.info("Localhost probe",
                    sandbox_id=sandbox_id, attempt=attempt,
                    status=probe.get("status"),
                    errors=probe_errors[:3] if probe_errors else [])

        yield Event(
            author=self.name,
            actions=EventActions(state_delta={
                "heal:probe_errors": probe_errors,
                "heal:probe_status": probe.get("status", 0),
                "heal:port":         actual_port,  # update with detected port
            }),
            content=genai_types.Content(parts=[genai_types.Part(
                text=f"probe: status={probe.get('status')} "
                     f"errors={len(probe_errors)}"
            )]),
        )


class HealExitChecker(BaseAgent):
    """
    Iteration step 4 — escalates (cleanly exits the loop) when there are
    no errors from install, server log, or probe.
    Otherwise lets the loop continue to the fixer block.
    """

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        error_lines  = ctx.session.state.get("heal:error_lines",  [])
        probe_errors = ctx.session.state.get("heal:probe_errors", [])
        sandbox_id   = ctx.session.state["heal:sandbox_id"]
        attempt      = ctx.session.state.get("heal:attempt", 1)

        all_clear = not error_lines and not probe_errors

        if all_clear:
            logger.info("All startup checks passed ✓",
                        sandbox_id=sandbox_id, attempt=attempt)

        yield Event(
            author=self.name,
            actions=EventActions(
                escalate=all_clear,
                state_delta={"heal:done": all_clear},
            ),
            content=genai_types.Content(parts=[genai_types.Part(
                text="clean ✓ — escalating" if all_clear
                else f"errors remain: {len(error_lines)} install, "
                     f"{len(probe_errors)} probe"
            )]),
        )

# =============================================================================
# Phase B — Fixer + Critic as BaseAgent subclasses (direct LLM calls)
# =============================================================================

class HealFixerAgent(BaseAgent):
    """
    Phase B — Fixer.
    Calls Codestral directly via ChatMistralAI.
    Reads stack_desc from session state (heal:stack_desc).
    Writes JSON {filepath: fixed_content} to heal:proposed_fixes.
    
    MODIFIED: Uses Instructor for structured healing.

    BUG4 FIX: this used to build a dead `context = context_manager
    .build_context(...)` value that referenced `referenced_content`
    before it was ever defined anywhere in the function, raising an
    unconditional NameError on every single call (crashing the whole
    heal LoopAgent). That block has been removed — its result was
    never consumed by either the Instructor path or the legacy LLM
    path below, so removing it changes no behavior other than fixing
    the crash.
    """

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_mistralai import ChatMistralAI
        import re as _re

        # Read everything from session state
        error_lines  = ctx.session.state.get("heal:error_lines",  [])
        probe_errors = ctx.session.state.get("heal:probe_errors", [])
        stack_logs   = ctx.session.state.get("heal:stack_logs",   "")
        file_tree    = ctx.session.state.get("heal:file_tree",    {})
        stack_desc   = ctx.session.state.get("heal:stack_desc",   "")
        
        skill_content = _load_heal_skill()

        # ── NEW: Use Instructor for structured healing ──────────────
        try:
            # First, try Instructor for cleaner structured output
            all_errors = (error_lines + probe_errors)[:15]
            error_text = "\n".join(f"- {e}" for e in all_errors)
            
            # Use Instructor to generate structured fixes
            class FixerResponse(BaseModel):
                fixes: dict[str, str]
                reasoning: str
            
            result = await generate_structured(
                prompt=(
                    f"Errors:\n{error_text}\n\n"
                    f"Server log:\n{stack_logs[-3000:]}\n\n"
                    f"Project files:\n{', '.join(list(file_tree.keys())[:40])}\n\n"
                    "Fix ALL errors. Return the complete fixed content for each affected file."
                ),
                response_model=FixerResponse,
                system_prompt=(
                    f"You are a senior {stack_desc} developer fixing sandbox errors.\n"
                    f"# OPTIMIZED HEAL SKILL (v{_HEAL_SKILL_VERSION})\n{skill_content}\n\n"
                    "CRITICAL RULES:\n"
                    "- ModuleNotFoundError → create ALL __init__.py files in the chain\n"
                    "- Pydantic ValidationError → fix .env (NOT .env.example)\n"
                    "- ImportError → add missing class/function WITHOUT removing existing ones\n"
                    "- For .env files: ALWAYS include DATABASE_URL=sqlite:///./preview.db\n"
                    "- For Python: use pydantic_settings.BaseSettings\n"
                    "- NEVER set extra='forbid' on any Settings class\n"
                    "- Return ONLY the fixes dict with filepath as key and complete content as value"
                ),
                temperature=0.1,
            )
            
            if result and result.fixes:
                # Parse the fixes dict into JSON
                proposed_fixes_json = json.dumps(result.fixes)
                logger.info(f"Instructor generated {len(result.fixes)} fixes")
                
                yield Event(
                    author=self.name,
                    actions=EventActions(state_delta={
                        "heal:proposed_fixes": proposed_fixes_json,
                        "heal:instructor_used": True,
                    }),
                    content=genai_types.Content(parts=[genai_types.Part(
                        text=f"fixer (Instructor): {len(result.fixes)} file(s)"
                    )]),
                )
                return
        except Exception as exc:
            logger.warning(f"Instructor fixer failed, falling back to LLM: {exc}")

        # ── LEGACY: Direct LLM call as fallback ──────────────────────
        # Extract files referenced in errors
        # Extract files referenced in errors
        referenced_files = []
        missing_symbols: dict[str, str] = {}  # file path -> missing symbol name
        for err in error_lines:
            # Most reliable signal: the absolute path Python prints in parens,
            # e.g. "cannot import name 'X' from 'a.b.c' (/home/user/a/b/c.py)"
            abs_match = _re.search(r"\(/home/user/([^)]+\.py)\)", err)
            dotted_match = _re.search(r"from '([\w.]+)'", err)
            missing_name_match = _re.search(r"cannot import name '(\w+)'", err)

            path = None
            if abs_match:
                path = abs_match.group(1)
            elif dotted_match:
                # Fall back: convert dotted module notation to a file path
                candidate = dotted_match.group(1).replace(".", "/") + ".py"
                if candidate in file_tree:
                    path = candidate

            if path and path in file_tree:
                if (path, file_tree[path]) not in referenced_files:
                    referenced_files.append((path, file_tree[path]))
                if missing_name_match:
                    missing_symbols[path] = missing_name_match.group(1)
        # Find files that import from error-mentioned modules
        importing_files = []
        for err in error_lines:
            m = _re.search(r"(?:cannot import name|No module named)\s+['\"]?([\w.]+)", err)
            if m:
                module_name = m.group(1)
                for fp, content in file_tree.items():
                    if fp.endswith(".py") and module_name in content and fp not in [r[0] for r in referenced_files]:
                        importing_files.append((fp, content[:1500]))
                        if len(importing_files) >= 3:
                            break
                            
        # Build referenced content
        # Build referenced content
        referenced_content = ""
        if referenced_files:
            snippets = []
            for path, content in referenced_files[:5]:
                label = f"=== {path} ==="
                if path in missing_symbols:
                    label += (
                        f"  ⚠ MISSING SYMBOL: '{missing_symbols[path]}' must be "
                        f"added to THIS EXACT FILE — do not create or patch "
                        f"__init__.py instead."
                    )
                snippets.append(f"{label}\n{content[:2000]}")
            referenced_content = "\nReferenced file contents:\n" + "\n\n".join(snippets)
        if importing_files:
            for path, content in importing_files[:3]:
                referenced_content += f"\n\n=== {path} (importing file) ===\n{content}"

        all_errors = (error_lines + probe_errors)[:15]
        error_text = "\n".join(f"- {e}" for e in all_errors)
        file_list  = "\n".join(f"  - {f}" for f in list(file_tree.keys())[:40])

        # Build a project structure summary so the LLM knows where imports live
        project_structure = []
        for fp in sorted(file_tree.keys()):
            if any(x in fp for x in [
                "config/database.py", "config/settings.py",
                "src/utils/helpers.py", "src/models/base.py",
                "src/services/auth.py",
            ]):
                exports = []
                content = file_tree[fp]
                for match in re.finditer(r'^(?:def|class|async def)\s+(\w+)', content, re.MULTILINE):
                    exports.append(match.group(1))
                project_structure.append(f"  {fp} — exports: {', '.join(exports[:10])}")
            elif fp.endswith("__init__.py") or fp.startswith(("docs/", ".")):
                continue
            else:
                project_structure.append(f"  {fp}")

        structure_summary = (
            "PROJECT STRUCTURE — use this to locate imports:\n" +
            "\n".join(project_structure[:50]) +
            ("\n  ..." if len(project_structure) > 50 else "")
        )

        llm = ChatMistralAI(
            model=settings.codestral_model,
            api_key=settings.codestral_api_key,
            base_url=settings.codestral_base_url,
            temperature=0.1,
            max_tokens=4096,
        )

        try:
            resp = await llm.ainvoke([
                SystemMessage(content=(
                    f"You are a senior {stack_desc} developer fixing sandbox errors.\n"
                    + (f"# OPTIMIZED HEAL SKILL (v{_HEAL_SKILL_VERSION})\n{skill_content}\n\n"
                       if skill_content else "")
                    + "Classify the error (build/runtime/network/config) from the raw context.\n"
                    "🚨🚨🚨 CRITICAL — PRESERVE ALL EXISTING CLASSES AND FUNCTIONS 🚨🚨🚨\n"
                    "When fixing a file, you MUST keep EVERY existing class and function.\n"
                    "Your fix MUST include the COMPLETE original file PLUS your changes.\n"
                    "NEVER remove a class like TokenData, UserBase, or TradeCreate.\n"
                    "NEVER remove a function like get_db, init_db, or verify_password.\n"
                    "Scan the CURRENT file content carefully — every class/function you see "
                    "MUST appear in your fixed version.\n"
                    "If you need to add something, ADD it — do not replace the whole file.\n\n"
                    "CRITICAL RULES:\n"
                    "- ModuleNotFoundError → create ALL __init__.py files in the chain\n"
                    "- Pydantic ValidationError → fix .env (NOT .env.example), use sensible defaults\n"
                    "- ImportError → add missing class/function WITHOUT removing existing ones\n"
                    "- For .env files: ALWAYS include DATABASE_URL=sqlite:///./preview.db\n"
                    "- For SvelteKit: guard localStorage with browser import\n"
                    "- For Python: use pydantic_settings.BaseSettings not pydantic.BaseSettings\n\n"
                    "Output ONLY valid JSON: {\"filepath\": \"complete file content\", ...}\n"
                    "If no fix is possible, return: {}"
                    "- For Python: use pydantic_settings.BaseSettings not pydantic.BaseSettings\n"
                    "- For Python models: use DeclarativeBase, NEVER declarative_base()\n"
                    "- For Python models: Base class MUST be empty (just 'pass'), no columns\n"
                    "- For Python settings: NEVER use PostgresDsn type for DATABASE_URL — use str with default 'sqlite:///./preview.db'\n"
                    "- For Python settings: NEVER add field_validator that rejects non-PostgreSQL URLs\n"
                    "- For Python database: get_db() MUST be sync Generator[Session], never async\n"
                    "- For Python routers: NEVER include /api/v1/ in route decorators — prefix comes from main.py\n"
                    "- For Python routers: NEVER use await with db.query(), db.add(), db.commit() — they are sync\n"
                    "- For ALLOWED_ORIGINS: it's List[str] — NEVER call .split(',') on it\n"
                    "- For init_db(): it's sync — NEVER write 'await init_db()'\n"
                    "- For Python IDs: use String(36), never UUID type — SQLite doesn't support it\n"
                    "- For Python enums: use String with default, never Enum type — SQLite doesn't support it\n"
                    "- NEVER import from sqlalchemy.dialects — breaks SQLite preview\n\n"
                    "- NEVER set extra='forbid', extra=\"forbid\", or extra=Extra.forbid on any "
                    "Settings/BaseSettings/model_config — this breaks the preview .env, which "
                    "may contain extra fields not declared on the model. If a Settings class "
                    "already has extra='forbid' anywhere, remove it and use extra='ignore' "
                    "or omit the extra setting entirely.\n"
                    "- MINIMAL EDITS on Settings/BaseSettings classes: when fixing a settings "
                    "file, do NOT regenerate the whole class. Only add, remove, or rename the "
                    "exact field(s) causing the reported error. Keep every other field name, "
                    "casing, type, and default exactly as they were in the current content.\n\n"
                    "CRITICAL — Python Syntax Validation:\n"
                    "- Before returning ANY Python file fix, verify it has valid syntax\n"
                    "- Check that ALL try/except blocks have proper indentation\n"
                    "- Check that ALL function bodies are indented\n"
                    "- NEVER return a fix that introduces an IndentationError\n"
                    "- If you cannot produce syntactically valid Python, return {} instead"
                    "- FIELD CASING: always use UPPER_SNAKE_CASE for Settings fields (e.g. "
                    "DATABASE_URL, SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES) to match "
                    "the .env file and how other files in the project access settings "
                    "(settings.DATABASE_URL, not settings.database_url). NEVER introduce a "
                    "lowercase or mixed-case field name — check the referenced file contents"                   
                    "CRITICAL — PRESERVE EXISTING CONTENT:\n"
                    "- When fixing a file, ADD missing classes/functions WITHOUT removing existing ones\n"
                    "- Scan the CURRENT file content for all class/function definitions\n"
                    "- Your fix MUST include ALL existing classes PLUS the new ones\n"
                    "- NEVER regenerate a file from scratch — only ADD what's missing\n"
                    "- If schemas/trading.py has StrategyCreate, TradeCreate, TradeRead\n"
                    "  and the error says DashboardStats is missing → ADD DashboardStats\n"
                    "  but KEEP StrategyCreate, TradeCreate, TradeRead exactly as they were\n"
                    "for the exact attribute name other code expects and match it precisely.\n\n"
                    # In the heal fixer prompt, add:
                    "CRITICAL — .env field names MUST match Settings class EXACTLY:"
                    "- If Settings has 'database_url' → .env must have 'database_url"
                    "- If Settings has 'DATABASE_URL' → .env must have 'DATABASE_URL"
                    "- Case matters! 'database_url' ≠ 'DATABASE_URL"
                    "CRITICAL — Preserve ALL existing .env fields:"
                    "- Read the original .env file"
                    "- Keep EVERY field that was there"
                    "- Only add/change the specific field that caused the error"
                    "- NEVER drop fields unless they're causing the error"
                    "Output ONLY valid JSON: {\"filepath\": \"complete file content\", ...}\n"
                    "If no fix is possible, return: {}"
                    "CRITICAL — IMPORT RESOLUTION:\n"
                    "- NameError: name 'get_db' → import it from config.database, NEVER define it locally\n"
                    "- NameError: name 'Depends' → import from fastapi, NEVER define it locally\n"
                    "- NameError: name 'HTTPException' → import from fastapi\n"
                    "- NameError: name 'settings' → import from config.settings\n"
                    "- NameError: name 'Base' → import from config.database\n"
                    "- NameError: name 'Session' → import from sqlalchemy.orm\n"
                    "- NEVER generate a new file at src/config/ — config/ is at project root\n"
                    "- BEFORE defining a missing function, check if it already exists in another file\n"
                    "- BEFORE creating a new file, check if it already exists under a different path\n\n"
                    "CRITICAL — DO NOT REGENERATE THE ENTIRE PROJECT:\n"
                    "- Fix ONLY the files with errors — never touch files that work\n"
                    "- 40 fixes is a RED FLAG — you're doing something wrong\n"
                    "- If you find yourself fixing >10 files, STOP and reconsider\n"
                    "- Focus on the ACTUAL error, not hypothetical improvements"
                )),
               
                HumanMessage(content=(
                    f"{structure_summary}\n\n"
                    f"Errors:\n{error_text}\n\n"
                    f"Server log:\n{stack_logs[-3000:]}\n\n"
                    f"Project files:\n{file_list}\n\n"
                    f"{referenced_content}\n\n"
                    "IMPORTANT: Fix ALL errors in ONE response. Return JSON with fixed file contents."
                    "🚨 BEFORE generating fixes, scan each file's CURRENT content. "
                    "Every class and function you see MUST be in your fixed version. "
                    
                )),
            ])
            result = resp.content.strip()
        except Exception as exc:
            logger.warning("HealFixerAgent LLM error", error=str(exc)[:200])
            result = "{}"

        yield Event(
            author=self.name,
            actions=EventActions(state_delta={"heal:proposed_fixes": result}),
            content=genai_types.Content(
                parts=[genai_types.Part(text=f"fixer: {len(result)} chars")]
            ),
        )
        
class HealCriticAgent(BaseAgent):
    """
    Phase B — Critic.
    Reflects on proposed fixes before upload using ChatMistralAI directly.
    Reads stack_desc from session state (heal:stack_desc).
    Writes JSON {approved: bool, reason: str} to heal:critique.
    
    MODIFIED: Uses Instructor for structured critique.
    """

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_mistralai import ChatMistralAI

        error_lines    = ctx.session.state.get("heal:error_lines",    [])
        proposed_fixes = ctx.session.state.get("heal:proposed_fixes", "{}")
        stack_desc     = ctx.session.state.get("heal:stack_desc",     "")
        stack_desc   = ctx.session.state.get("heal:stack_desc",   "")
        skill_content = _load_heal_skill()  # ← add this line

        error_text = "\n".join(f"- {e}" for e in error_lines[:10])

        # ── NEW: Use Instructor for structured critique ──────────────
        try:
            from pydantic import BaseModel
            
            class CritiqueResult(BaseModel):
                approved: bool
                reason: str
                concerns: list[str] = []
            
            result = await generate_structured(
                prompt=(
                    f"Original errors:\n{error_text}\n\n"
                    f"Proposed fixes (JSON):\n{proposed_fixes[:2000]}\n\n"
                    "Review and approve if fixes correctly address the errors."
                ),
                response_model=CritiqueResult,
                system_prompt=(
                    f"You are a senior {stack_desc} code reviewer.\n"
                    f"# OPTIMIZED HEAL SKILL (v{_HEAL_SKILL_VERSION})\n{skill_content}\n\n"
                    "APPROVAL RULES:\n"
                    "1. 'Module not found: ./globals.css' → APPROVE if fix creates app/globals.css\n"
                    "2. 'Module not found: package-name' → APPROVE if fix adds it to package.json\n"
                    "3. 'EADDRINUSE' → APPROVE if fix changes the port or kills existing process\n"
                    "4. Any missing file error → APPROVE if the fix creates that file\n"
                    "5. REJECT only if the fix does NOT resolve the original error\n"
                    "If the fix resolves the error, APPROVE it even if not perfect."
                ),
                temperature=0.1,
            )
            
            if result:
                critique_json = json.dumps({
                    "approved": result.approved,
                    "reason": result.reason,
                    "concerns": result.concerns,
                })
                logger.info(f"Critique: approved={result.approved}, reason={result.reason[:50]}")
                
                yield Event(
                    author=self.name,
                    actions=EventActions(state_delta={"heal:critique": critique_json}),
                    content=genai_types.Content(parts=[genai_types.Part(
                        text=f"critic (Instructor): {'✅ approved' if result.approved else '❌ rejected'}"
                    )]),
                )
                return
        except Exception as exc:
            logger.warning(f"Instructor critic failed, falling back to LLM: {exc}")

        # ── LEGACY: Direct LLM call as fallback ──────────────────────
        llm = ChatMistralAI(
            model=settings.codestral_model,
            api_key=settings.codestral_api_key,
            base_url=settings.codestral_base_url,
            temperature=0.1,
            max_tokens=512,
        )

        try:
            resp = await llm.ainvoke([
                SystemMessage(content=(
                    f"You are a senior {stack_desc} code reviewer.\n"
                    + (f"# OPTIMIZED HEAL SKILL (v{_HEAL_SKILL_VERSION})\n{skill_content}\n\n"
                       if skill_content else "")
                    + "Classify the error (build/runtime/network/config) from the raw context.\n"
                    "Review the proposed fixes and approve if they correctly address the errors.\n\n"
                    "APPROVAL RULES:\n"
                    "1. 'Module not found: ./globals.css' → APPROVE if fix creates app/globals.css\n"
                    "2. 'Module not found: package-name' → APPROVE if fix adds it to package.json\n"
                    "3. 'EADDRINUSE' → APPROVE if fix changes the port or kills existing process\n"
                    "4. Any missing file error → APPROVE if the fix creates that file\n"
                    "5. REJECT only if the fix does NOT resolve the original error (missing imports, syntax errors, wrong types).\n"
                    "NEVER reject for missing __repr__, __str__, docstrings, type hints, or other cosmetic improvements.\n"
                    "If the fix resolves the error, APPROVE it even if it's not perfect.\n\n"
                    "Return ONLY valid JSON: {\"approved\": true or false, \"reason\": \"one sentence\"}"
                )),
                HumanMessage(content=(
                    f"Original errors:\n{error_text}\n\n"
                    f"Proposed fixes (JSON):\n{proposed_fixes[:2000]}\n\n"
                    "Return approval JSON."
                )),
            ])
            result = resp.content.strip()
        except Exception as exc:
            logger.warning("HealCriticAgent LLM error", error=str(exc)[:200])
            result = json.dumps({"approved": True,
                                  "reason": "critic unavailable — auto-approving"})

        yield Event(
            author=self.name,
            actions=EventActions(state_delta={"heal:critique": result}),
            content=genai_types.Content(
                parts=[genai_types.Part(text=f"critic: {result[:80]}")]
            ),
        )

def _build_fixer_agent(stack_desc: str) -> "HealFixerAgent":
    """Factory kept for _build_heal_pipeline compatibility."""
    return HealFixerAgent(name="HealFixerAgent")


def _build_critic_agent(stack_desc: str) -> "HealCriticAgent":
    """Factory kept for _build_heal_pipeline compatibility."""
    return HealCriticAgent(name="HealCriticAgent")
    
class ApplyOrRetryAgent(BaseAgent):
    """
    Phase B — applies critic-approved fixes to the sandbox and updates
    heal:file_tree in session state.  Restarts the dev server after apply.
    If critic rejected, logs the reason and lets the loop retry.

    BUG1 FIX: validates proposed fix keys against the actual file_tree
    before uploading — prevents Codestral from fixing phantom files.
    EXCEPTION: allows __init__.py files for import errors, and new files
    when the error is ModuleNotFoundError.
    BUG3 FIX: uses _wait_for_server() after restart instead of fixed sleep.
    
    MODIFIED: Uses Tree-sitter for validation before upload.
    """

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        start_time  = time.time()
        sandbox_id  = ctx.session.state["heal:sandbox_id"]
        project_id  = ctx.session.state["heal:project_id"]
        start_cmd   = ctx.session.state["heal:start_cmd"]
        port        = ctx.session.state["heal:port"]
        has_node    = ctx.session.state["heal:has_node"]
        attempt     = ctx.session.state.get("heal:attempt", 1)
        file_tree   = dict(ctx.session.state.get("heal:file_tree", {}))

        sandbox     = _get_sandbox(sandbox_id)

        # ── Parse critique ────────────────────────────────────────────────────
        raw_critique = ctx.session.state.get("heal:critique", "{}")
        try:
            if isinstance(raw_critique, dict):
                critique = raw_critique
            elif isinstance(raw_critique, str):
                raw_critique = re.sub(r"^```(?:json)?\s*", "", raw_critique.strip())
                raw_critique = re.sub(r"\s*```$", "", raw_critique)
                critique = json.loads(raw_critique)
            else:
                critique = {"approved": False, "reason": "invalid type"}
        except Exception:
            critique = {"approved": False, "reason": "parse error"}
    
        approved = critique.get("approved", False)
        reason   = critique.get("reason", "")

        if not approved:
            logger.warning("Critic rejected fix — retrying",
                           attempt=attempt, reason=reason)
            yield Event(
                author=self.name,
                actions=EventActions(state_delta={
                    "heal:stack_logs": (
                        ctx.session.state.get("heal:stack_logs", "")
                        + f"\nCritic rejected: {reason}"
                    ),
                    "heal:attempt": attempt + 1,
                }),
                content=genai_types.Content(parts=[genai_types.Part(
                    text=f"fix rejected by critic: {reason}"
                )]),
            )
            return

        # ── Parse proposed fixes ──────────────────────────────────────────────
        raw_fixes = ctx.session.state.get("heal:proposed_fixes", "{}")
        proposed = {}

        try:
            if isinstance(raw_fixes, dict):
                # Already a dict from parallel collector
                proposed = raw_fixes
            elif isinstance(raw_fixes, str):
                # String from HealFixerAgent - clean and parse
                cleaned = re.sub(r"^```(?:json)?\s*", "", raw_fixes.strip())
                cleaned = re.sub(r"\s*```$", "", cleaned)
                proposed = json.loads(cleaned)
            else:
                logger.warning(f"Unexpected type for heal:proposed_fixes: {type(raw_fixes)}")
                proposed = {}
        except (json.JSONDecodeError, AttributeError, TypeError) as exc:
            logger.warning(f"Failed to parse proposed_fixes: {exc}")
            proposed = {}
            
        # ── BUG1 filter — validate proposed files ────────────────────
        error_lines = ctx.session.state.get("heal:error_lines", [])
        is_module_not_found = any(
            "ModuleNotFoundError" in e or "No module named" in e
            for e in error_lines
        )
        # After the existing is_module_not_found check, add:
        is_config_missing = any(
            "src.config.settings" in e or "src.config.database" in e
            for e in error_lines
        )

        is_validation_error = any(
            "ValidationError" in e
            for e in error_lines
        )
        
        # ── Files that are always safe to create new ─────────────

        ALWAYS_ALLOW_NEW = {
            ".env", ".env.local", ".env.example",
            "src/__init__.py",
            "src/core/__init__.py",
            "src/core/config.py",
            "src/core/auth.py",
            "src/core/security.py",
            "src/config/__init__.py",      # ← ADD
            "src/config/settings.py",      # ← ADD
            "src/config/database.py",      # ← ADD
            "src/schemas/__init__.py",
            "src/models/__init__.py",
            "src/models/base.py",      # ← ADD — frequently regenerated
            "src/models/user.py", 
            "src/api/__init__.py",
            "src/services/__init__.py",
            "src/utils/__init__.py",
            "src/routers/__init__.py",
            "app/__init__.py",
            "app/core/__init__.py",
            "app/schemas/__init__.py",
            "app/models/__init__.py",
            "app/api/__init__.py",
            "app/services/__init__.py",
            "config/__init__.py",        # ← ADD
            "config/settings.py",
            "src/services/task_queue.py",     # ← ADD: PG Durable
            "src/services/outbox.py",         # ← ADD: PG Durable
            "src/services/healer.py",         # ← ADD: PG Durable
            "src/workers/__init__.py",        # ← ADD: PG Durable
            "src/workers/task_processor.py",  # ← ADD: PG Durable
        }

        validated = {}
        phantom   = []
        # ── NEW: Apply ast-grep fixes before uploading ─────────────────
        ast_grep = get_ast_grep()
        parser = get_parser()
        
        for fp, content in proposed.items():
            # Apply ast-grep fixes first
            if fp.endswith(".py") and content:
                content = ast_grep.fix_all(content, fp)
                logger.debug(f"ast-grep applied to {fp}")          
        
        # ── NEW: Tree-sitter validation for each fix ──────────────────
        parser = get_parser()
        
        for fp, content in proposed.items():
            # Validate syntax before accepting
            if content and len(content.strip()) > 20:
                is_valid, errors = parser.validate_syntax(fp, content)
                if not is_valid:
                    logger.warning(f"Proposed fix for {fp} has syntax errors: {errors[:3]}")
                    # Try to heal the syntax with a targeted fix
                    try:
                        class SyntaxFix(BaseModel):
                            content: str
                            errors_fixed: list[str]
                        
                        fixed = await generate_structured(
                            prompt=f"Fix these syntax errors in {fp}:\n{chr(10).join(errors[:5])}\n\nContent:\n{content[:3000]}",
                            response_model=SyntaxFix,
                            system_prompt="Fix syntax errors only. Return complete valid content.",
                        )
                        if fixed.content:
                            # Apply ast-grep again after Instructor fix
                            content = ast_grep.fix_all(fixed.content, fp)
                            
                            logger.info(f"Fixed syntax errors in {fp} with Instructor")
                    except Exception as exc:
                        logger.warning(f"Failed to fix syntax in {fp}: {exc}")
            
            if fp in file_tree:
                # Existing file — always allow
                validated[fp] = content
            
            elif fp.endswith("__init__.py") and len(content.strip()) < 100:
                # Only auto-create empty __init__.py files
                validated[fp] = content
                logger.info("Created missing __init__.py", path=fp)
            
            elif is_config_missing and any(
                x in fp for x in ("config/settings", "config/database", "config/__init__")
            ) and content and len(content.strip()) > 20:
                validated[fp] = content
                logger.info("BUG1 exception: allowing config file for missing module", path=fp)
            elif is_module_not_found and content and len(content.strip()) > 20:
                # A ModuleNotFoundError means this new file is genuinely required
                # to resolve the import chain — allow it even though it's not
                # yet in file_tree.
                validated[fp] = content
                logger.info("Allowing new file for ModuleNotFoundError", path=fp)
            elif fp in ALWAYS_ALLOW_NEW and content and len(content.strip()) > 10:
                # Known-safe scaffolding files (settings, config, core __init__
                # modules etc.) are always allowed even outside a ModuleNotFoundError.
                validated[fp] = content
                logger.info("Allowing new file (in ALWAYS_ALLOW_NEW)", path=fp)
            
            else:
                # All other new files require full AI generation
                logger.warning("Skipping new file (requires full AI generation)", path=fp)
                phantom.append(fp)
    
        if phantom:
            logger.warning("Discarding phantom file fixes", files=phantom)

        if not validated:
            logger.warning("No valid fixes after BUG1 filter — retrying",
                            attempt=attempt)
            yield Event(
                author=self.name,
                actions=EventActions(state_delta={"heal:attempt": attempt + 1}),
                content=genai_types.Content(parts=[genai_types.Part(
                    text="no valid fixes — retrying"
                )]),
            )
            return

        # ── Strip extra='forbid' from ANY settings file before upload ──
        # ── STRIP extra='forbid' from ALL settings files before upload ──
        import re as _re_strip
        for fp in list(validated.keys()):
            if fp.endswith(("settings.py", "config.py")) and "config" in fp.lower():
                old = validated[fp]
                new = _re_strip.sub(
                    r"""extra\s*[=:]\s*['"]forbid['"]\s*,?\s*""",
                    '', old
                )
                new = _re_strip.sub(
                    r""""extra"\s*:\s*"forbid"\s*,?\s*""",
                    '', new
                )
                if new != old:
                    validated[fp] = new
                    file_tree[fp] = new
                    logger.info("Heal loop: stripped extra='forbid' from uploaded file",
                                path=fp)

        # Also check file_tree for any settings files NOT in validated
        for fp, content in file_tree.items():
            if fp.endswith(("settings.py", "config.py")) and "config" in fp.lower():
                if "extra" in content and "forbid" in content:
                    file_tree[fp] = _re_strip.sub(
                        r"""extra\s*[=:]\s*['"]forbid['"]\s*,?\s*""",
                        '', content
                    )
                    logger.info("Heal loop: stripped extra='forbid' from existing file",
                                path=fp)

        # ── Fix base.py: replace declarative_base() with DeclarativeBase ──
        
        for fp in list(validated.keys()):
            if fp.endswith("base.py") and "models" in fp:
                current = validated[fp]
                if "declarative_base()" in current and "DeclarativeBase" not in current:
                    current = current.replace(
                        "from sqlalchemy.orm import declarative_base",
                        "from sqlalchemy.orm import DeclarativeBase"
                    ).replace(
                        "Base = declarative_base()",
                        "class Base(DeclarativeBase):\n    pass"
                    )
                    validated[fp] = current
                    logger.info("Heal loop: replaced declarative_base() with DeclarativeBase",
                                path=fp)
                # Strip columns from Base class — operate on the CURRENT
                # (possibly just-updated) content, not the stale `old` value,
                # so the import fix above is never silently discarded.
                if "class Base" in current:
                    validated[fp] = re.sub(
                        r'(class Base\([^)]+\):).*?(?=\n\S|\Z)',
                        r'\1\n    pass',
                        current,
                        flags=re.DOTALL,
                    )
                    logger.info("Heal loop: stripped columns from Base class", path=fp)

        # ── Guard: never let a fix silently drop existing .env fields ──────
        # In ApplyOrRetryAgent, find _merge_env_preserving_fields and add:

        LIST_FIELDS = {"ALLOWED_ORIGINS", "CORS_ORIGINS"}

        def _merge_env_preserving_fields(original: str, new: str) -> str:
            def _parse(text: str) -> dict[str, str]:
                out = {}
                for line in text.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    out[k.strip()] = v.strip()
                return out

            original_fields = _parse(original)
            new_fields = _parse(new)
    
            # Fix List[str] fields — ensure JSON array format
            for field in LIST_FIELDS:
                if field in new_fields and not new_fields[field].startswith("["):
                    new_fields[field] = f'["{new_fields[field]}"]'
                if field in original_fields and not original_fields[field].startswith("["):
                    original_fields[field] = f'["{original_fields[field]}"]'
    
            missing = {k: v for k, v in original_fields.items() if k not in new_fields}
            if not missing:
                return "\n".join(f"{k}={v}" for k, v in new_fields.items()) + "\n"

            logger.warning("ApplyOrRetryAgent: fix dropped .env fields, restoring",
                            dropped=list(missing.keys()))
    
            merged = {**new_fields, **missing}
            return "\n".join(f"{k}={v}" for k, v in merged.items()) + "\n"
    
        if ".env" in validated and ".env" in file_tree:
            validated[".env"] = _merge_env_preserving_fields(
                file_tree[".env"], validated[".env"]
            )

        # ── NEW: Final Tree-sitter validation pass ──────────────────────
        invalid_files = []
        for fp, content in validated.items():
            if content and len(content.strip()) > 20:
                is_valid, errors = parser.validate_syntax(fp, content)
                if not is_valid:
                    invalid_files.append(f"{fp}: {errors[:2]}")
        
        if invalid_files:
            logger.warning(f"Some fixes still have syntax errors: {invalid_files}")
            # Remove the _push call or replace with a proper SSE mechanism if available
        # Upload fixes + restart
        file_tree.update(validated)
        _upload_fixed_files(sandbox, validated)
        
        with sandbox_span("dev_server_restart", project_id=project_id,
                          sandbox_id=sandbox_id):
            _restart_dev_server(sandbox, start_cmd, port)

        # BUG3 FIX — poll instead of sleep
        await _wait_for_server(sandbox, port, has_node=has_node)

        # ── Verify after restart (Phoenix span) ───────────────────────────────
        actual_port = _detect_actual_port(sandbox, port)
        with probe_span("verify", project_id=project_id,
                        sandbox_id=sandbox_id) as v_span:
            recheck        = _probe_localhost(sandbox, actual_port, has_node)
            recheck_errors = _classify_probe_errors(recheck)
            record_probe_result(v_span, recheck)
            record_heal_result(v_span, validated,
                               verified=not recheck_errors,
                               error_type="post_fix")

        logger.info("Fix applied",
                    attempt=attempt,
                    files=list(validated.keys()),
                    verified=not recheck_errors,
                    remaining_errors=recheck_errors[:3])

        # ── SkillOpt-style trajectory logging (fire-and-forget) ──────────
        # ── AFTER verifying and applying fixes ──────────────────────────────
        # Enhanced trajectory logging with full fields
        try:
            from app.services.skill_trajectory_service import log_heal_trajectory, log_failure_signature
            
            project_id = ctx.session.state.get("heal:project_id", "")
            stack_desc = ctx.session.state.get("heal:stack_desc", "")
            stack = ctx.session.state.get("heal:stack", {})
            error_lines = ctx.session.state.get("heal:error_lines", [])
            
            # Determine which errors were resolved
            all_errors = error_lines + ctx.session.state.get("heal:probe_errors", [])
            resolved_errors = [e for e in all_errors if e not in recheck_errors]
            remaining_errors = recheck_errors
            
            # Log full trajectory
            await log_heal_trajectory(
                project_id=project_id,
                stack_desc=stack_desc,
                stack=stack,
                prompt=f"Fix attempt {attempt}",
                skill_version=_HEAL_SKILL_VERSION,
                skill_content=_load_heal_skill(),
                skill_type="heal",
                attempt=attempt,
                errors_seen=all_errors,
                errors_resolved=resolved_errors,
                errors_remaining=remaining_errors,
                files_modified=list(validated.keys()),
                files_created=[f for f in validated.keys() if f not in ctx.session.state.get("heal:file_tree", {})],
                verified=not recheck_errors,
                exhausted=attempt >= MAX_ATTEMPTS,
                time_to_heal_ms=int((time.time() - start_time) * 1000) if 'start_time' in locals() else 0,
                iterations_to_heal=attempt,
                final_status="clean" if not recheck_errors else "healed" if validated else "exhausted",
            )
            
            # Log failure signatures for remaining errors
            for err in recheck_errors[:5]:
                await log_failure_signature(
                    project_id=project_id,
                    stack_desc=stack_desc,
                    error_pattern=err,
                    error_count=1,
                    failed_files=list(validated.keys()),
                    skill_version=_HEAL_SKILL_VERSION,
                    error_category=_categorize_error(err),
                )
                
        except Exception as exc:
            logger.warning("Trajectory logging failed", error=str(exc))
        
        yield Event(
            author=self.name,
            actions=EventActions(state_delta={
                "heal:file_tree":   file_tree,
                "heal:probe_errors": recheck_errors,
                "heal:port":         actual_port,
                "heal:attempt":      attempt + 1,
            }),
            content=genai_types.Content(parts=[genai_types.Part(
                text=f"fix applied: {len(validated)} file(s), "
                     f"verify errors={len(recheck_errors)}"
            )]),
        )
# ═══════════════════════════════════════════════════════════════════════════════
# ADK Phase C — Parallel file fixing
# ═══════════════════════════════════════════════════════════════════════════════

class SingleFileFixWorker(BaseAgent):
    """
    Phase C — fixes exactly one file concurrently with other workers.
    Reads error context from session state, writes to fix:{filepath}.
    BUG1 FIX: validates filepath exists in file_tree before writing.
    
    MODIFIED: Uses Tree-sitter for validation after fix.
    """
    filepath:   str
    stack_desc: str

    model_config = {"arbitrary_types_allowed": True}

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        filepath    = self.filepath
        stack_desc  = self.stack_desc
        file_tree   = ctx.session.state.get("heal:file_tree",   {})
        error_lines = ctx.session.state.get("heal:error_lines", [])
        probe_errors = ctx.session.state.get("heal:probe_errors", [])
        stack_logs  = ctx.session.state.get("heal:stack_logs",  "")

        original = file_tree.get(filepath, "")
        if not original:
            yield Event(
                author=self.name,
                actions=EventActions(state_delta={}),
                content=genai_types.Content(parts=[genai_types.Part(
                    text=f"skip {filepath} — not in file_tree (BUG1)"
                )]),
            )
            return

        all_errors = (error_lines + probe_errors)[:10]
        error_text = "\n".join(f"- {e}" for e in all_errors)
        skill_content = _load_heal_skill()  # ← add this line
        
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_mistralai import ChatMistralAI

        # ── NEW: Use Instructor for single-file fix ──────────────────────
        try:
            from pydantic import BaseModel
            
            class SingleFileFix(BaseModel):
                content: str
                changes_made: list[str]
            
            result = await generate_structured(
                prompt=(
                    f"File: {filepath}\n\n"
                    f"Errors:\n{error_text}\n\n"
                    f"Server log:\n{stack_logs[-500:]}\n\n"
                    f"Current content:\n{original[:4000]}\n\n"
                    "Return the complete fixed file."
                ),
                response_model=SingleFileFix,
                system_prompt=(
                    f"You are an expert {stack_desc} developer. "
                    + (f"# OPTIMIZED HEAL SKILL (v{_HEAL_SKILL_VERSION})\n{skill_content}\n\n"
                       if skill_content else "")
                    + "Fix the errors in this file. "
                    "If this file defines a Settings/BaseSettings class or model_config, "
                    "NEVER set extra='forbid' — remove it if present. "
                    "Make the SMALLEST possible edit to fix the reported error. "
                    "Always use UPPER_SNAKE_CASE for any Settings field you add or rename. "
                    "Output ONLY the complete fixed file content."
                ),
                temperature=0.1,
            )
            
            if result and result.content and len(result.content.strip()) > 20:
                # Validate with Tree-sitter
                parser = get_parser()
                is_valid, errors = parser.validate_syntax(filepath, result.content)
                if not is_valid:
                    logger.warning(f"SingleFileFixWorker result for {filepath} has syntax errors: {errors[:3]}")
                    # Still store it, but log the warning
                
                state_key = f"fix:{filepath}"
                yield Event(
                    author=self.name,
                    actions=EventActions(state_delta={state_key: result.content}),
                    content=genai_types.Content(parts=[genai_types.Part(
                        text=f"fixed (Instructor): {filepath}"
                    )]),
                )
                return
        except Exception as exc:
            logger.warning(f"Instructor single-file fix failed, falling back to LLM: {exc}")

        # ── LEGACY: Direct LLM call as fallback ──────────────────────────
        llm = ChatMistralAI(
            model=settings.codestral_model,
            api_key=settings.codestral_api_key,
            base_url=settings.codestral_base_url,
            temperature=0.1,
            max_tokens=4096,
        )

        try:
            resp = await llm.ainvoke([
                
                SystemMessage(content=(
                    f"You are an expert {stack_desc} developer. "
                    + (f"# OPTIMIZED HEAL SKILL (v{_HEAL_SKILL_VERSION})\n{skill_content}\n\n"
                       if skill_content else "")
                    + "Classify the error (build/runtime/network/config) from the raw context.\n"
                    "🚨🚨🚨 CRITICAL — PRESERVE ALL EXISTING CLASSES AND FUNCTIONS 🚨🚨🚨\n"
                    "When fixing a file, you MUST keep EVERY existing class and function.\n"
                    "Scan the CURRENT file content — every class/function you see "
                    "MUST appear in your fixed version.\n"
                    "NEVER remove TokenData, UserBase, or any existing class.\n"
                    "If you need to add something, ADD it — do not replace the whole file.\n\n"
                    "Fix the errors in this file. "
                    "Make the SMALLEST possible edit to fix the reported error — do not "
                    "regenerate the class; keep every unrelated field name, casing, type, "
                    "and default exactly as in the current content. "
                    "Always use UPPER_SNAKE_CASE for any Settings field you add or rename "
                    "(e.g. DATABASE_URL, not database_url) to match how other files access "
                    "settings attributes. "
                    "no markdown fences, no explanation."
                )),
                HumanMessage(content=(
                    f"File: {filepath}\n\n"
                    f"Errors:\n{error_text}\n\n"
                    f"Server log:\n{stack_logs[-500:]}\n\n"
                    f"Current content:\n{original[:4000]}\n\n"
                    "Return the complete fixed file."
                )),
            ])
            content = resp.content.strip()
            content = re.sub(r"^```[\w]*\n?", "", content)
            content = re.sub(r"\n?```$", "", content)
        except Exception as exc:
            logger.warning("SingleFileFixWorker LLM error",
                           filepath=filepath, error=str(exc)[:100])
            content = ""

        # ── Validate with Tree-sitter before storing ──────────────────────
        if content and len(content.strip()) > 20:
            parser = get_parser()
            is_valid, errors = parser.validate_syntax(filepath, content)
            if not is_valid:
                logger.warning(f"SingleFileFixWorker result for {filepath} has syntax errors: {errors[:3]}")
        
        state_key = f"fix:{filepath}"
        yield Event(
            author=self.name,
            actions=EventActions(state_delta={
                state_key: content if len(content.strip()) > 20 else "",
            }),
            content=genai_types.Content(parts=[genai_types.Part(
                text=f"{'fixed' if content else 'failed'}: {filepath}"
            )]),
        )


class ParallelFileFixCollector(BaseAgent):
    """
    Phase C — runs after ParallelAgent completes.
    Gathers all fix:{filepath} keys from session state, validates them
    (BUG1), and writes the merged result to heal:proposed_fixes.
    Also generates a critic-compatible JSON for HealCriticAgent.

    BUG1: Only applies fixes for files that exist in file_tree.
    EXCEPTION: Allows creating new files when the error is a
    ModuleNotFoundError (import failure) — those require new files.
    
    MODIFIED: Uses Tree-sitter to validate collected fixes.
    """
    filepaths:  list[str]
    stack_desc: str

    model_config = {"arbitrary_types_allowed": True}

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        file_tree   = ctx.session.state.get("heal:file_tree",  {})
        error_lines = ctx.session.state.get("heal:error_lines", [])
        collected: dict[str, str] = {}

        # ── Detect import errors (ModuleNotFoundError) ────────────────────
        is_import_error = any(
            "ModuleNotFoundError" in e or "No module named" in e
            for e in error_lines
        )

        # ── NEW: Tree-sitter validation ──────────────────────────────────
        parser = get_parser()
        
        for fp in self.filepaths:
            content = ctx.session.state.get(f"fix:{fp}", "")
            if not content or len(content.strip()) <= 20:
                continue

            # Validate syntax before collecting
            is_valid, errors = parser.validate_syntax(fp, content)
            if not is_valid:
                logger.warning(f"Parallel fix for {fp} has syntax errors: {errors[:3]}")
                # Try to fix syntax with Instructor
                try:
                    from pydantic import BaseModel
                    class SyntaxFix(BaseModel):
                        content: str
                    
                    fixed = await generate_structured(
                        prompt=f"Fix syntax errors in this file:\n{chr(10).join(errors[:3])}\n\nContent:\n{content[:3000]}",
                        response_model=SyntaxFix,
                        system_prompt="Fix syntax errors. Return complete valid content.",
                    )
                    if fixed.content:
                        content = fixed.content
                        logger.info(f"Fixed syntax in {fp} with Instructor")
                except Exception as exc:
                    logger.warning(f"Failed to fix syntax in {fp}: {exc}")

            # BUG1 FIX — validate file exists in actual tree
            # EXCEPTION: allow new files for import errors
            if fp in file_tree:
                # File exists — apply fix
                collected[fp] = content
            elif is_import_error:
                # New file needed for import resolution — allow it
                collected[fp] = content
                logger.info(
                    "ParallelFileFixCollector: allowing new file for import error",
                    filepath=fp,
                )
            else:
                # Phantom file — discard
                logger.warning(
                    "ParallelFileFixCollector: discarding phantom file",
                    filepath=fp,
                )

        phantom_count = len(self.filepaths) - len(collected)
        if phantom_count > 0:
            logger.warning("ParallelFileFixCollector: phantom files discarded",
                           count=phantom_count)

        yield Event(
            author=self.name,
            actions=EventActions(state_delta={
                "heal:proposed_fixes": json.dumps(collected),
                # Pre-approve parallel fixes — Critic still runs next
                "heal:critique": json.dumps({
                    "approved": len(collected) > 0,
                    "reason":   f"{len(collected)} file(s) fixed in parallel"
                                + (" (including new files for import errors)"
                                   if is_import_error and collected else ""),
                }),
            }),
            content=genai_types.Content(parts=[genai_types.Part(
                text=f"collected {len(collected)} parallel fixes"
                     + (f" (+ new files for import errors)"
                        if is_import_error and collected else "")
            )]),
        )

def _build_parallel_fixer(
    error_files: list[str],
    stack_desc:  str,
) -> ParallelAgent:
    """
    Phase C — build a ParallelAgent that fixes multiple files concurrently,
    followed by a collector that merges results into heal:proposed_fixes.
    Called from _build_heal_pipeline() when > 1 file needs fixing.
    """
    workers = [
        SingleFileFixWorker(
            name=f"FixWorker_{re.sub(r'[^a-zA-Z0-9]', '_', fp)[:25]}",
            filepath=fp,
            stack_desc=stack_desc,
        )
        for fp in error_files
    ]
    return ParallelAgent(
        name="ParallelFileFixer",
        sub_agents=workers,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline factory
# ═══════════════════════════════════════════════════════════════════════════════

def _build_heal_pipeline(stack_desc: str) -> LoopAgent:
    """
    Build the full heal LoopAgent pipeline.

    Each iteration:
      InstallRunnerAgent → DevServerAgent → ProbeAgent → HealExitChecker
      → HealFixerAgent → HealCriticAgent → ApplyOrRetryAgent

    max_iterations is 5 for Python/FastAPI stacks (cascading import errors
    need more attempts) and 3 for JS/TS stacks.
    """
    _is_python = any(
        x in stack_desc.lower()
        for x in ("python", "fastapi", "django", "flask")
    )

    return LoopAgent(
        name="SandboxHealLoop",
        max_iterations=3 if _is_python else 2,
        sub_agents=[
            InstallRunnerAgent(name="InstallRunnerAgent"),
            DevServerAgent(name="DevServerAgent"),
            ProbeAgent(name="ProbeAgent"),
            HealExitChecker(name="HealExitChecker"),
            _build_fixer_agent(stack_desc),
            _build_critic_agent(stack_desc),
            ApplyOrRetryAgent(name="ApplyOrRetryAgent"),
        ],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# InMemoryRunner wrapper
# ═══════════════════════════════════════════════════════════════════════════════

async def _run_heal_session(
    sandbox:    "Sandbox",
    file_tree:  dict[str, str],
    stack:      dict,
    start_cmd:  str,
    port:       int,
    project_id: str,
    has_node:   bool,
    is_node:    bool,
    is_python:  bool,
    stack_desc: str,
) -> dict[str, str]:
    """
    Spins an ADK InMemoryRunner for the SandboxHealLoop.
    Seeds session state with all heal:* keys.
    Returns the final file_tree from session state.
    """
    pipeline    = _build_heal_pipeline(stack_desc)
    runner      = InMemoryRunner(
        agent=pipeline,
        app_name="agents",
    )

    initial_state = {
        "heal:file_tree":   dict(file_tree),
        "heal:error_lines": [],
        "heal:stack_logs":  "",
        "heal:probe_errors": [],
        "heal:probe_status": 0,
        "heal:proposed_fixes": "{}",
        "heal:critique":    json.dumps({"approved": False, "reason": ""}),
        "heal:done":        False,
        "heal:stack":       stack,
        "heal:stack_desc":  stack_desc,
        "heal:start_cmd":   start_cmd,
        "heal:port":        port,
        "heal:sandbox_id":  sandbox.sandbox_id,
        "heal:project_id":  project_id,
        "heal:is_node":     is_node,
        "heal:is_python":   is_python,
        "heal:has_node":    has_node,
        "heal:attempt":     1,
    }

    session = await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id=project_id or "anon",
        state=initial_state,
    )

    user_msg = genai_types.Content(
        role="user",
        parts=[genai_types.Part(
            text=(
                f"Heal sandbox for project: {project_id}\n"
                f"Stack: {stack_desc}\n"
                f"Port: {port}\n"
                f"Files: {len(file_tree)}"
            )
        )],
    )

    async for _event in runner.run_async(
        user_id=project_id or "anon",
        session_id=session.id,
        new_message=user_msg,
    ):
        pass

    final = await runner.session_service.get_session(
        app_name=runner.app_name,
        user_id=project_id or "anon",
        session_id=session.id,
    )

    final_tree  = final.state.get("heal:file_tree", file_tree)
    final_done  = final.state.get("heal:done", False)
    final_probe = final.state.get("heal:probe_errors", [])

    # ── Log generation outcome for heal skill ──────────────────────────────
    try:
        from app.services.skill_trajectory_service import log_generation_outcome, log_skill_snapshot
        
        if final_done and not final_probe:
            final_status = "clean"
        elif final_tree != file_tree:
            final_status = "healed"
        else:
            final_status = "exhausted"
        
        # Log skill snapshot
        await log_skill_snapshot(
            project_id=project_id,
            skill_version=_HEAL_SKILL_VERSION,
            skill_content=_load_heal_skill(),
            skill_type="heal",
            stack_desc=stack_desc,
            stack=stack,
        )
        
        # Log generation outcome
        await log_generation_outcome(
            project_id=project_id,
            stack_desc=stack_desc,
            stack=stack,
            prompt=f"Heal project with {len(file_tree)} files",
            skill_version=_HEAL_SKILL_VERSION,
            skill_content=_load_heal_skill(),
            total_heal_attempts=final.state.get("heal:attempt", 1),
            final_status=final_status,
        )
        logger.info("Heal outcome logged", project_id=project_id, status=final_status)
    except Exception as exc:
        logger.warning("Outcome logging failed", error=str(exc))
    

    if not final_done and final_probe:
        try:
            from app.services.skill_trajectory_service import log_generation_outcome
            asyncio.create_task(log_generation_outcome(
                project_id=project_id,
                stack_desc=stack_desc,
                total_heal_attempts=MAX_ATTEMPTS,
                final_status="exhausted",
            ))
        except Exception:
            pass
        raise PreviewHealError(
            f"Heal loop exhausted after {MAX_ATTEMPTS} attempts. "
            f"Last probe errors: {'; '.join(final_probe[:3])}",
            partial_tree=final_tree,
        )

    return final_tree


# ═══════════════════════════════════════════════════════════════════════════════
# Public API — install_and_heal  (same signature as original)
# ═══════════════════════════════════════════════════════════════════════════════

async def install_and_heal(
    sandbox:    "Sandbox",
    file_tree:  dict[str, str],
    stack:      dict,
    start_cmd:  str,
    port:       int,
    project_id: str = "",
) -> dict[str, str]:
    """
    ADK LoopAgent-driven heal pipeline.
    Signature and return type are identical to the original so
    e2b_service.py requires no changes.

    Replaces:
      - bare for attempt in range(1, MAX_ATTEMPTS + 1) loop
      - _ask_codestral_to_fix two-shot plan+fix
      - time.sleep(START_WAIT) fixed delay
    """
    sandbox_id = sandbox.sandbox_id

    parts = [v for k, v in stack.items()
             if k in ("frontend", "backend", "database")
             and v and v.lower() not in ("none", "")]
    stack_desc = " + ".join(parts) if parts else "JavaScript"

    is_node   = "package.json" in file_tree
    is_python = "requirements.txt" in file_tree

    # ── Register sandbox for agent access ─────────────────────────────────────
    _register_sandbox(sandbox)

    try:
        # ── Runtime check (unchanged from original) ───────────────────────────
        with sandbox_span("runtime_check", project_id=project_id,
                          sandbox_id=sandbox_id) as span:
            try:
                check = sandbox.commands.run(
                    "node --version && npm --version 2>&1 || "
                    "python3 --version 2>&1 || echo 'no-runtime'",
                    timeout=10,
                )
                output = (check.stdout or "").strip()[:100]
                logger.info("Runtime check",
                            sandbox_id=sandbox_id, output=output)
                if span.is_recording():
                    span.set_attribute("chiscode.runtime", output)
            except Exception as exc:
                logger.warning("Runtime check failed", error=str(exc))

        # ── Check Node availability for probes ────────────────────────────────
        has_node = False
        try:
            chk = sandbox.commands.run(
                "which node 2>/dev/null || echo no", timeout=5)
            has_node = "no" not in (chk.stdout or "")
        except Exception:
            pass

        # ── Restore node_modules cache (unchanged from original) ──────────────
        if is_node:
            try:
                cache = sandbox.commands.run(
                    "if [ -d /home/user/node_modules_cache ] && "
                    "[ ! -d /home/user/node_modules ]; then "
                    "cp -r /home/user/node_modules_cache "
                    "/home/user/node_modules && "
                    "echo 'cache-restored'; else echo 'no-cache'; fi",
                    timeout=30, user="user",
                )
                logger.info("Node modules cache",
                            status=(cache.stdout or "").strip())
            except Exception:
                pass

        # ── Run ADK heal pipeline ─────────────────────────────────────────────
        with heal_span(attempt=0, project_id=project_id,
                       sandbox_id=sandbox_id) as root_span:
            final_tree = await _run_heal_session(
                sandbox=sandbox,
                file_tree=file_tree,
                stack=stack,
                start_cmd=start_cmd,
                port=port,
                project_id=project_id,
                has_node=has_node,
                is_node=is_node,
                is_python=is_python,
                stack_desc=stack_desc,
            )
            if root_span.is_recording():
                root_span.set_attribute(OTEL_ATTRS["verified"], True)
                root_span.set_attribute("chiscode.heal.files_in",
                                        len(file_tree))
                root_span.set_attribute("chiscode.heal.files_out",
                                        len(final_tree))

        return final_tree

    finally:
        _deregister_sandbox(sandbox_id)


# ═══════════════════════════════════════════════════════════════════════════════
# Public API — probe_live_url  (BUG2 fix: now retries up to 2 times)
# ═══════════════════════════════════════════════════════════════════════════════

async def probe_live_url(
    sandbox:    "Sandbox",
    url:        str,
    file_tree:  dict[str, str],
    stack:      dict,
    start_cmd:  str,
    port:       int,
    project_id: str = "",
) -> dict[str, str]:
    """
    Post-launch probe of the public E2B URL.
    BUG2 FIX: now runs a mini 2-iteration fix loop instead of single attempt.
    Fully traced in Phoenix. Non-blocking (called from asyncio.create_task).
    """
    loop        = asyncio.get_running_loop()
    sandbox_id  = sandbox.sandbox_id
    current_tree = dict(file_tree)

    parts = [v for k, v in stack.items()
             if k in ("frontend", "backend", "database")
             and v and v.lower() not in ("none", "")]
    stack_desc = " + ".join(parts) if parts else "JavaScript"

    has_node = False
    try:
        chk = sandbox.commands.run(
            "which node 2>/dev/null || echo no", timeout=5)
        has_node = "no" not in (chk.stdout or "")
    except Exception:
        pass

    logger.info("Post-launch URL probe", url=url, sandbox_id=sandbox_id)
    await asyncio.sleep(8)  # let E2B proxy warm up

    # BUG2 FIX — retry loop (max 2 attempts instead of 1)
    for attempt in range(1, 3):
        with probe_span(f"post_launch_external_{attempt}",
                        project_id=project_id,
                        sandbox_id=sandbox_id, url=url) as p_span:
            probe = await loop.run_in_executor(
                None, _probe_external_url, sandbox, url, has_node)
            probe_errors = _classify_probe_errors(probe)
            record_probe_result(p_span, probe)

        logger.info("Post-launch probe",
                    sandbox_id=sandbox_id, attempt=attempt,
                    status=probe.get("status"),
                    errors=probe_errors[:3] if probe_errors else [])

        if not probe_errors:
            logger.info("Post-launch probe passed ✓",
                        sandbox_id=sandbox_id, attempt=attempt)
            return current_tree

        if attempt == 2:
            # Exhausted retries — return best tree we have
            logger.warning("Post-launch probe: errors remain after 2 attempts",
                           errors=probe_errors[:3])
            return current_tree

        # ── Attempt fix ───────────────────────────────────────────────────────
        error_type = "network" if probe.get("network_errors") else "runtime"
        logger.warning("Post-launch errors", error_type=error_type,
                       errors=probe_errors[:5])

        try:
            app_log = await loop.run_in_executor(None, _read_log, sandbox)
            stack_lines = [
                ln.strip() for ln in app_log.splitlines()
                if "at " in ln or "Error" in ln
            ][:10]

            ctx = {
                "component":     "post_launch_browser",
                "error_summary": " | ".join(probe_errors[:3]),
                "stack_trace":   "\n".join(stack_lines),
            }

            # Use original two-shot _ask_codestral_to_fix for the lightweight
            # post-launch probe — full ADK runner overhead not warranted here
            fixed = await _ask_codestral_to_fix(
                current_tree, probe_errors, stack,
                error_type=error_type, trace_context=ctx,
            )

            if fixed:
                current_tree.update(fixed)
                await loop.run_in_executor(
                    None, _upload_fixed_files, sandbox, fixed)
                await loop.run_in_executor(
                    None, _restart_dev_server, sandbox, start_cmd, port)
                # BUG3 FIX in post-launch too — poll instead of fixed sleep
                await _wait_for_server(sandbox, port, has_node=has_node)

        except Exception as exc:
            logger.warning("Post-launch fix error (non-fatal)",
                           error=str(exc)[:200])
            return current_tree

    return current_tree


# ═══════════════════════════════════════════════════════════════════════════════
# _ask_codestral_to_fix  — kept as legacy fallback for probe_live_url
# and any caller outside the ADK pipeline.  MODIFIED with Instructor.
# ═══════════════════════════════════════════════════════════════════════════════

async def _ask_codestral_to_fix(
    file_tree:     dict[str, str],
    error_lines:   list[str],
    stack:         dict,
    error_type:    str = "build",
    trace_context: dict | None = None,
) -> dict[str, str]:
    """
    Legacy two-shot LLM fix.  Used only by probe_live_url.
    The main heal loop now uses HealFixerAgent + HealCriticAgent instead.
    BUG1 FIX applied here too: files_to_fix validated against file_tree.
    
    MODIFIED: Uses Instructor for structured fixes.
    """
    compactor = get_token_compactor()
    compacted_tree = compactor.compact_file_tree(file_tree, max_files=20)
   
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_mistralai import ChatMistralAI

    parts = [v for k, v in stack.items()
             if k in ("frontend", "backend", "database")
             and v and v.lower() not in ("none", "")]
    stack_desc = " + ".join(parts) if parts else "JavaScript"

    error_text = "\n".join(f"- {e}" for e in error_lines[:15])
    file_list  = "\n".join(f"  - {f}" for f in file_tree.keys())

    trace_context_text = ""
    if trace_context:
        trace_context_text = (
            f"\nTrace context:\n"
            f"  Component: {trace_context.get('component', 'unknown')}\n"
            f"  Error summary: {trace_context.get('error_summary', '')}\n"
            f"  Stack trace: {trace_context.get('stack_trace', '')[:500]}\n"
        )

    context_map = {
        "runtime": (
            "BROWSER RUNTIME errors after app loads. "
            "Common: browser APIs at module level (localStorage, window, document), "
            "undefined variables, missing null checks."
        ),
        "network": (
            "NETWORK errors — API calls failing. "
            "Common: wrong URLs, CORS, missing env vars, backend not running."
        ),
        "build": "BUILD/COMPILE errors from dev server or pip install.",
    }
    context = context_map.get(error_type, "Build errors.")

    # ── NEW: Try Instructor first for structured fixes ──────────────────
    try:
        from pydantic import BaseModel
        
        class LegacyFixResult(BaseModel):
            files: list[str]
            fixes: dict[str, str]
            explanation: str
        
        result = await generate_structured(
            prompt=(
                f"Stack: {stack_desc}\nError type: {error_type}\n"
                f"Errors:\n{error_text}\n"
                f"{trace_context_text}\n"
                f"Project files:\n{file_list}\n\n"
                "Which files need fixing? Return the complete fixed content."
            ),
            response_model=LegacyFixResult,
            system_prompt=(
                f"You are a senior {stack_desc} developer debugging errors. "
                f"{context} "
                "Identify which files are causing the errors. "
                "Return the complete fixed content for each file."
            ),
            temperature=0.1,
        )
        
        if result and result.fixes:
            # BUG1 FIX — validate against actual file_tree keys
            validated_fixes = {}
            for fp, content in result.fixes.items():
                if fp in file_tree and content and len(content.strip()) > 20:
                    # Validate with Tree-sitter
                    parser = get_parser()
                    is_valid, errors = parser.validate_syntax(fp, content)
                    if is_valid:
                        validated_fixes[fp] = content
                    else:
                        logger.warning(f"Fix for {fp} has syntax errors: {errors[:3]}")
            
            if validated_fixes:
                logger.info(f"Legacy fix (Instructor): {len(validated_fixes)} files")
                return validated_fixes
    except Exception as exc:
        logger.warning(f"Instructor legacy fix failed, falling back to LLM: {exc}")

    # ── LEGACY: Direct LLM call as fallback ──────────────────────────────
    llm = ChatMistralAI(
        model=settings.codestral_model,
        api_key=settings.codestral_api_key,
        base_url=settings.codestral_base_url,
        temperature=0.1,
        max_tokens=2048,
    )

    plan_resp = await llm.ainvoke([
        SystemMessage(content=(
            f"You are a senior {stack_desc} developer debugging errors. "
            f"{context} "
            f"\n\nProject files (compacted to show {len([f for f in file_tree if f in compacted_tree])} of {len(file_tree)}):\n{compacted_tree}\n\n"
            "🚨 CRITICAL — PRESERVE ALL EXISTING CONTENT:\n"
            "- Keep EVERY class and function that already exists in the file\n"
            "- Only ADD missing ones — never remove or replace existing ones\n"
            "- Scan the CURRENT content before generating a fix\n\n"
            "Identify which files are causing the errors. "
            'Return ONLY valid JSON: {"files": ["path1"], "explanation": "reason"}'
            
        )),

        HumanMessage(content=(
            f"Stack: {stack_desc}\nError type: {error_type}\n"
            f"Errors:\n{error_text}\n"
            f"{trace_context_text}\n"
            f"Project files:\n{file_list}\n\nWhich files need fixing?"
        )),
    ])

    try:
        raw  = re.sub(r"^```(?:json)?\s*", "", plan_resp.content.strip())
        raw  = re.sub(r"\s*```$", "", raw)
        plan = json.loads(raw)
        # BUG1 FIX — validate against actual file_tree keys
        files_to_fix = [
            f for f in plan.get("files", [])
            if f in file_tree
        ]
    except Exception:
        files_to_fix = [
            f for f in file_tree
            if f.endswith((".ts", ".tsx", ".js", ".jsx", ".svelte",
                           ".vue", ".py", ".json"))
        ][:5]

    if not files_to_fix:
        return {}

    fixed: dict[str, str] = {}
    for filepath in files_to_fix:
        original = file_tree.get(filepath, "")
        if not original:
            continue

        fix_resp = await llm.ainvoke([
            SystemMessage(content=(
                f"You are an expert {stack_desc} developer fixing {error_type} errors. "
                f"{context} "
                "For SvelteKit: NEVER use localStorage/window/document at module level. "
                "For Python: NEVER use declarative_base(), PostgresDsn, async get_db(), await init_db(), "
                "await db.query(), .split(',') on ALLOWED_ORIGINS, sqlalchemy.dialects imports, "
                "or columns in Base class. "
                "Use DeclarativeBase, str for DATABASE_URL with sqlite default, sync get_db(), "
                "sync init_db(), String(36) for IDs, and empty Base class.\n"
                "If this file defines a Settings/BaseSettings class or model_config, "
                "Make the SMALLEST possible edit to fix the reported error — do not "
                "regenerate the class; keep every unrelated field name, casing, type, "
                "and default exactly as in the current content. "
                "Always use UPPER_SNAKE_CASE for any Settings field you add or rename "
                "(e.g. DATABASE_URL, not database_url) to match how other files access "
                "settings attributes. "
                "🚨 CRITICAL — PRESERVE ALL EXISTING CONTENT:\n"
                "- Keep EVERY class and function that already exists in the file\n"
                "- Only ADD missing ones — never remove or replace existing ones\n"
                "- Scan the CURRENT content before generating a fix\n\n"
                "Output ONLY the complete fixed file — no markdown, no explanation."
            )),
            
            HumanMessage(content=(
                f"File: {filepath}\nError type: {error_type}\n"
                f"Errors:\n{error_text}\n"
                f"{trace_context_text}\n"
                f"Current content:\n{original[:4000]}\n\nReturn the complete fixed file."
            )),
        ])
        content = fix_resp.content.strip()
        content = re.sub(r"^```(?:\w+)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        
        # ── Validate with Tree-sitter ───────────────────────────────────────
        if content and len(content.strip()) > 20:
            parser = get_parser()
            is_valid, errors = parser.validate_syntax(filepath, content)
            if not is_valid:
                logger.warning(f"Legacy fix for {filepath} has syntax errors: {errors[:3]}")
                # Still store it, but log the warning
        
        if len(content.strip()) > 20:
            fixed[filepath] = content
            logger.info("File fixed", path=filepath, error_type=error_type)

    return fixed