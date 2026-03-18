"""
ChisCode — Smart Stack Advisor
================================
LangGraph node that analyzes the user's prompt and suggests
the best tech stack with rationale. Returns multiple options
ranked by fit — the user picks one (HITL) before generation starts.

Stack categories covered:
  Frontend:  HTML/CSS/JS, React, Next.js, Vue, Svelte, TanStack
  Backend:   FastAPI, Express/Node.js, Rust (Axum), Go (Gin), Django, Rails
  Database:  SQLite, PostgreSQL, MongoDB, Redis, Supabase, PlanetScale
  Extras:    Tailwind, shadcn/ui, Prisma, tRPC, GraphQL, WebSockets, etc.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI
import json, re

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ── Stack catalogue ───────────────────────────────────────────

STACK_KNOWLEDGE = """
APP TYPE → RECOMMENDED STACKS

landing_page / marketing:
  - HTML + CSS + Vanilla JS  (zero dependencies, fast, SEO-perfect)
  - Next.js + Tailwind       (if animations/blog needed)

dashboard / admin_panel:
  - React + Vite + Tailwind + shadcn/ui + FastAPI backend
  - Next.js App Router + Prisma + PostgreSQL  (full-stack)

api_only / microservice:
  - FastAPI + Python         (AI/ML workloads, rapid prototyping)
  - Express + Node.js        (JS ecosystem, large npm library access)
  - Axum + Rust              (high performance, low latency)
  - Go + Gin                 (concurrency-heavy, cloud-native)

cli_tool:
  - Python + Typer/Click     (scripting, data processing)
  - Rust                     (performance-critical binaries)
  - Node.js + Commander      (JS developer tools)

real_time_app / chat / collaborative:
  - Next.js + Supabase Realtime + PostgreSQL
  - React + FastAPI + WebSockets + Redis pub/sub

e_commerce:
  - Next.js + Stripe + Prisma + PostgreSQL
  - Nuxt.js + Medusa.js

data_app / analytics:
  - Python + Streamlit       (quickest data viz)
  - React + FastAPI + Pandas + PostgreSQL  (production grade)

mobile_web / pwa:
  - React + Vite + PWA plugin + FastAPI
  - SvelteKit                (tiny bundle, excellent PWA support)

game / interactive:
  - Vanilla JS + Canvas/WebGL
  - React + Three.js / Phaser

ai_app / chatbot:
  - Next.js + Vercel AI SDK + FastAPI  (streaming)
  - React + FastAPI + LangChain

COMPLEXITY MODIFIERS:
  simple   → prefer minimal deps, single-file where possible
  moderate → standard framework, split frontend/backend
  complex  → monorepo, typed APIs (tRPC/OpenAPI), proper auth layer
"""


# ── Main function ─────────────────────────────────────────────

async def suggest_stacks(
    prompt:      str,
    app_type:    str,
    complexity:  str,
    features:    list[str],
) -> list[dict[str, Any]]:
    """
    Returns a list of 3 stack options, each:
    {
      "id": "option_a",
      "label": "FastAPI + React + PostgreSQL",
      "frontend": "React + Vite + Tailwind",
      "backend": "FastAPI (Python)",
      "database": "PostgreSQL",
      "extras": ["shadcn/ui", "Alembic"],
      "rationale": "Best for...",
      "complexity_fit": "moderate",
      "pros": ["..."],
      "cons": ["..."],
    }
    """
    llm = ChatMistralAI(
        model=settings.codestral_model,
        api_key=settings.codestral_api_key,
        base_url=settings.codestral_base_url,
        temperature=0.2,
        max_tokens=2048,
    )

    system = SystemMessage(content=f"""You are a senior software architect advising on tech stack selection.

{STACK_KNOWLEDGE}

Return ONLY a valid JSON array of exactly 3 stack options. No markdown, no explanation.

Each option:
{{
  "id": "option_a" | "option_b" | "option_c",
  "label": "Short name e.g. FastAPI + React",
  "frontend": "e.g. React + Vite + Tailwind",
  "backend": "e.g. FastAPI (Python 3.11)",
  "database": "e.g. PostgreSQL" or "None",
  "extras": ["lib1", "lib2"],
  "rationale": "2 sentences on why this fits this specific app",
  "complexity_fit": "simple" | "moderate" | "complex",
  "pros": ["pro1", "pro2", "pro3"],
  "cons": ["con1", "con2"]
}}

Order: best fit first. Offer variety (e.g. don't suggest 3 React options).
""")

    human = HumanMessage(content=f"""App description: {prompt}
App type: {app_type}
Complexity: {complexity}
Key features: {', '.join(features) if features else 'none specified'}

Suggest 3 tech stacks.""")

    try:
        response = await llm.ainvoke([system, human])
        raw = response.content.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        options = json.loads(raw)
        if not isinstance(options, list):
            raise ValueError("Expected list")
        return options[:3]
    except Exception as exc:
        logger.warning("Stack advisor failed, using defaults", error=str(exc))
        return _default_stacks(app_type, complexity)


def _default_stacks(app_type: str, complexity: str) -> list[dict]:
    """Fallback stacks when LLM call fails."""
    return [
        {
            "id": "option_a",
            "label": "HTML + CSS + Vanilla JS",
            "frontend": "HTML5 + CSS3 + Vanilla JS",
            "backend": "None",
            "database": "None",
            "extras": [],
            "rationale": "Zero dependencies, works everywhere, fastest to ship.",
            "complexity_fit": "simple",
            "pros": ["No build step", "Universal browser support", "Easy to deploy"],
            "cons": ["No component reuse", "Manual DOM manipulation"],
        },
        {
            "id": "option_b",
            "label": "FastAPI + React + SQLite",
            "frontend": "React + Vite + Tailwind CSS",
            "backend": "FastAPI (Python 3.11)",
            "database": "SQLite",
            "extras": ["shadcn/ui"],
            "rationale": "Production-grade full-stack with minimal infrastructure.",
            "complexity_fit": "moderate",
            "pros": ["Type-safe API", "Rich UI components", "Easy local dev"],
            "cons": ["Two runtimes to manage", "SQLite not for multi-server"],
        },
        {
            "id": "option_c",
            "label": "Next.js + PostgreSQL",
            "frontend": "Next.js 14 App Router + Tailwind",
            "backend": "Next.js API Routes",
            "database": "PostgreSQL + Prisma ORM",
            "extras": ["shadcn/ui", "NextAuth.js"],
            "rationale": "Full-stack React framework with built-in SSR and auth.",
            "complexity_fit": "complex",
            "pros": ["SSR/SSG built-in", "Single codebase", "Vercel-ready"],
            "cons": ["JS-only", "Complex caching model"],
        },
    ]
    
