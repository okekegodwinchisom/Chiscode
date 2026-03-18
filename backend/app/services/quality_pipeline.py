"""
ChisCode — Code Quality Pipeline (Phase 5)
===========================================
Runs linting, syntax validation, and security checks against a file_tree dict.
All checks are in-process (no subprocess/gVisor needed at this stage) so the
pipeline works on HF Spaces without Docker-in-Docker.

Checks per language:
  Python  — ast.parse (syntax) + ruff rules (inline)
  JS/TS   — basic pattern checks + common pitfall detection
  JSON    — json.loads
  HTML    — structure validation
  CSS     — unclosed-brace detection
  General — empty files, missing README, hardcoded secrets scan

Returns a QualityReport with issues grouped by severity.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Literal

from app.core.logging import get_logger

logger = get_logger(__name__)

Severity = Literal["error", "warning", "info"]


@dataclass
class QualityIssue:
    file:     str
    line:     int | None
    severity: Severity
    code:     str          # e.g. "PY001", "JS003"
    message:  str


@dataclass
class QualityReport:
    issues:    list[QualityIssue] = field(default_factory=list)
    passed:    bool               = True
    score:     int                = 100   # 0–100
    file_count: int               = 0

    def add(self, file: str, severity: Severity, code: str, message: str, line: int | None = None):
        self.issues.append(QualityIssue(file=file, line=line, severity=severity, code=code, message=message))
        self.passed = not any(i.severity == "error" for i in self.issues)

    def to_dict(self) -> dict:
        return {
            "passed":     self.passed,
            "score":      self._compute_score(),
            "file_count": self.file_count,
            "issue_count": len(self.issues),
            "errors":   [i.__dict__ for i in self.issues if i.severity == "error"],
            "warnings": [i.__dict__ for i in self.issues if i.severity == "warning"],
            "info":     [i.__dict__ for i in self.issues if i.severity == "info"],
        }

    def _compute_score(self) -> int:
        if not self.issues:
            return 100
        deductions = sum(10 if i.severity == "error" else 3 if i.severity == "warning" else 1
                         for i in self.issues)
        return max(0, 100 - deductions)


# ── Secret patterns (simple heuristics) ───────────────────────
_SECRET_PATTERNS = [
    (re.compile(r'(?i)(api[_-]?key|secret[_-]?key|private[_-]?key)\s*=\s*["\'][a-zA-Z0-9_\-]{16,}["\']'), "Possible hardcoded secret"),
    (re.compile(r'(?i)password\s*=\s*["\'][^"\']{6,}["\']'),      "Possible hardcoded password"),
    (re.compile(r'(?i)token\s*=\s*["\'][a-zA-Z0-9_.\-]{20,}["\']'), "Possible hardcoded token"),
    (re.compile(r'sk-[a-zA-Z0-9]{32,}'),                           "Possible OpenAI key"),
    (re.compile(r'ghp_[a-zA-Z0-9]{36}'),                           "Possible GitHub PAT"),
    (re.compile(r'AIza[0-9A-Za-z\-_]{35}'),                        "Possible Google API key"),
]

# ── JS/TS pitfall patterns ─────────────────────────────────────
_JS_PITFALLS = [
    (re.compile(r'\beval\s*\('),                               "JS001", "warning", "eval() is a security risk"),
    (re.compile(r'\bdocument\.write\s*\('),                    "JS002", "warning", "document.write() is unsafe"),
    (re.compile(r'\bvar\s+\w'),                                "JS003", "info",    "Use let/const instead of var"),
    (re.compile(r'innerHTML\s*=\s*(?![\'"]\s*[\'"])'),         "JS004", "warning", "innerHTML assignment may cause XSS"),
    (re.compile(r'\bconsole\.(log|warn|error)\s*\('),          "JS005", "info",    "console statement left in code"),
    (re.compile(r'==\s*null|null\s*=='),                       "JS006", "info",    "Use === null for strict equality"),
]

# ── Python pitfall patterns ────────────────────────────────────
_PY_PITFALLS = [
    (re.compile(r'\bprint\s*\('),                              "PY001", "info",    "print() left in code — use logging"),
    (re.compile(r'except\s*:'),                                "PY002", "warning", "Bare except: clause — catch specific exceptions"),
    (re.compile(r'\bexec\s*\('),                               "PY003", "warning", "exec() is a security risk"),
    (re.compile(r'import\s+\*\s*$', re.MULTILINE),            "PY004", "warning", "Wildcard import pollutes namespace"),
    (re.compile(r'# type: ignore'),                            "PY005", "info",    "type: ignore suppressor found"),
    (re.compile(r'TODO|FIXME|HACK|XXX'),                       "PY006", "info",    "TODO/FIXME marker left in code"),
]


# ═══════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════

async def run_quality_pipeline(
    file_tree: dict[str, str],
    file_plan: list[str] | None = None,
) -> QualityReport:
    """
    Run all quality checks on a file_tree dict.
    file_plan: expected files — used to flag missing files.
    """
    report = QualityReport(file_count=len(file_tree))

    # 1. Missing files
    if file_plan:
        for expected in file_plan:
            if expected not in file_tree:
                report.add(expected, "warning", "GEN001", "Expected file was not generated")

    # 2. Per-file checks
    for path, content in file_tree.items():
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""

        # Empty file
        if not content or len(content.strip()) < 10:
            report.add(path, "error", "GEN002", "File is empty or nearly empty")
            continue

        # Secrets scan (all files)
        _scan_secrets(path, content, report)

        # Language-specific
        if ext == "py":
            _check_python(path, content, report)
        elif ext in ("js", "ts", "jsx", "tsx", "mjs", "cjs"):
            _check_javascript(path, content, report)
        elif ext == "json":
            _check_json(path, content, report)
        elif ext == "html":
            _check_html(path, content, report)
        elif ext == "css":
            _check_css(path, content, report)
        elif ext in ("yaml", "yml"):
            _check_yaml(path, content, report)

    # 3. Project-level checks
    _check_project_structure(file_tree, report)

    logger.info("Quality pipeline complete",
                files=len(file_tree), issues=len(report.issues),
                passed=report.passed, score=report._compute_score())
    return report


# ── Checkers ──────────────────────────────────────────────────

def _scan_secrets(path: str, content: str, report: QualityReport):
    # Skip example/template files
    if any(x in path.lower() for x in (".example", ".sample", "template", "test_", "spec.")):
        return
    for pattern, message in _SECRET_PATTERNS:
        if pattern.search(content):
            report.add(path, "warning", "SEC001", message)
            break  # one warning per file


def _check_python(path: str, content: str, report: QualityReport):
    # Syntax check
    try:
        ast.parse(content)
    except SyntaxError as e:
        report.add(path, "error", "PY000",
                   f"SyntaxError line {e.lineno}: {e.msg}", line=e.lineno)
        return   # No point running further checks on unparseable code

    # Pitfall patterns
    lines = content.splitlines()
    for i, line in enumerate(lines, 1):
        for pattern, code, severity, message in _PY_PITFALLS:
            if pattern.search(line):
                report.add(path, severity, code, message, line=i)  # type: ignore[arg-type]

    # Check: no __main__ guard in scripts
    if path.endswith(".py") and "if __name__" not in content and "def " in content:
        # Only flag top-level scripts (not modules)
        if not any(x in path for x in ("app/", "src/", "__init__")):
            report.add(path, "info", "PY007", "Script missing if __name__ == '__main__' guard")


def _check_javascript(path: str, content: str, report: QualityReport):
    lines = content.splitlines()
    for i, line in enumerate(lines, 1):
        for pattern, code, severity, message in _JS_PITFALLS:
            if pattern.search(line):
                report.add(path, severity, code, message, line=i)  # type: ignore[arg-type]

    # Check: missing 'use strict' in non-module JS
    if path.endswith(".js") and "use strict" not in content and "import " not in content and "export " not in content:
        report.add(path, "info", "JS007", "Non-module JS missing 'use strict'")

    # Check: React components missing PropTypes or TypeScript types
    if path.endswith(".jsx") and "PropTypes" not in content and "interface " not in content:
        report.add(path, "info", "JS008", "React component has no prop type definitions")


def _check_json(path: str, content: str, report: QualityReport):
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        report.add(path, "error", "JSON001", f"Invalid JSON: {e.msg} at line {e.lineno}", line=e.lineno)
        return

    # package.json specific
    if path.endswith("package.json"):
        for field in ("name", "version"):
            if field not in parsed:
                report.add(path, "warning", "PKG001", f"package.json missing '{field}' field")
        if "dependencies" not in parsed and "devDependencies" not in parsed:
            report.add(path, "info", "PKG002", "package.json has no dependencies")

    # tsconfig.json
    if "tsconfig" in path and "compilerOptions" not in parsed:
        report.add(path, "warning", "TS001", "tsconfig.json missing compilerOptions")


def _check_html(path: str, content: str, report: QualityReport):
    low = content.lower()

    if "<!doctype" not in low:
        report.add(path, "warning", "HTML001", "Missing <!DOCTYPE html> declaration")
    if "<html" not in low:
        report.add(path, "error", "HTML002", "Missing <html> root element")
    if "<head" not in low:
        report.add(path, "warning", "HTML003", "Missing <head> element")
    if "<title" not in low:
        report.add(path, "info", "HTML004", "Missing <title> element")
    if "<body" not in low:
        report.add(path, "warning", "HTML005", "Missing <body> element")
    if "charset" not in low:
        report.add(path, "info", "HTML006", "Missing charset meta tag")
    if "viewport" not in low:
        report.add(path, "info", "HTML007", "Missing viewport meta (mobile responsive)")

    # XSS patterns
    if re.search(r'\.innerHTML\s*=', content):
        report.add(path, "warning", "HTML008", "innerHTML assignment may allow XSS")


def _check_css(path: str, content: str, report: QualityReport):
    # Count braces
    opens  = content.count("{")
    closes = content.count("}")
    if opens != closes:
        report.add(path, "error", "CSS001",
                   f"Mismatched braces: {opens} open, {closes} close")

    # Detect !important overuse
    important_count = content.count("!important")
    if important_count > 5:
        report.add(path, "info", "CSS002",
                   f"!important used {important_count} times — consider refactoring specificity")


def _check_yaml(path: str, content: str, report: QualityReport):
    # Basic tab check (YAML doesn't allow tabs)
    if "\t" in content:
        report.add(path, "error", "YAML001", "YAML file contains tabs — use spaces only")

    # Docker-compose specific
    if "docker-compose" in path or "compose" in path:
        if "version:" not in content:
            report.add(path, "info", "DC001", "docker-compose.yml missing version field")
        if "services:" not in content:
            report.add(path, "error", "DC002", "docker-compose.yml missing services block")


def _check_project_structure(file_tree: dict, report: QualityReport):
    paths_lower = [p.lower() for p in file_tree]

    if not any("readme" in p for p in paths_lower):
        report.add("README.md", "warning", "PROJ001", "Project is missing a README.md")

    if not any(".env.example" in p or "env.example" in p for p in paths_lower):
        report.add(".env.example", "info", "PROJ002",
                   "No .env.example — document required environment variables")

    if not any(p.endswith(".gitignore") for p in paths_lower):
        report.add(".gitignore", "info", "PROJ003", "No .gitignore file")

    # Flag if main entry point seems to be missing
    has_py_main   = any(p in ("main.py", "app.py", "run.py", "server.py") for p in file_tree)
    has_js_main   = any(p in ("index.js", "server.js", "app.js", "index.html") for p in file_tree)
    has_src_main  = any("src/" in p for p in file_tree)
    if not (has_py_main or has_js_main or has_src_main):
        report.add("?", "warning", "PROJ004", "No recognisable entry point found")
        
