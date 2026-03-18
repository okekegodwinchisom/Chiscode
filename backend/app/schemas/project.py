"""
ChisCode — Project Schemas
Pydantic v2 models for project and version data.
"""
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from app.schemas.user import PyObjectId

ProjectStatus = Literal[
    "pending",
    "analyzing",
    "awaiting_stack_selection",   # ← Phase 3: HITL pause after analyze
    "stack_selected",             # ← Phase 3: user picked a stack
    "generating",
    "quality_check",
    "self_healing",
    "awaiting_confirmation",
    "committing",
    "complete",
    "failed",
    "cancelled",
]


class TechStack(BaseModel):
    frontend: str = ""
    backend:  str = ""
    database: str = ""
    extras:   list[str] = Field(default_factory=list)


class StackOption(BaseModel):
    id:             str
    label:          str
    frontend:       str = ""
    backend:        str = ""
    database:       str = ""
    extras:         list[str] = Field(default_factory=list)
    rationale:      str = ""
    complexity_fit: str = ""
    pros:           list[str] = Field(default_factory=list)
    cons:           list[str] = Field(default_factory=list)


class ProjectSpec(BaseModel):
    """Structured requirement spec derived from the user's natural language prompt."""
    app_type:         str = ""
    app_name:         str = ""
    description:      str = ""
    features:         list[str] = Field(default_factory=list)
    auth_required:    bool = False
    database_needed:  bool = False
    api_needed:       bool = False
    mobile_responsive: bool = True
    preferred_stack:  Optional[TechStack] = None
    complexity:       Literal["simple", "moderate", "complex"] = "simple"


# ── Core Project Model ─────────────────────────────────────────

class ProjectInDB(BaseModel):
    model_config = {"populate_by_name": True, "arbitrary_types_allowed": True}

    id:               Optional[PyObjectId] = Field(default=None, alias="_id")
    user_id:          PyObjectId
    name:             str
    description:      str
    original_prompt:  str

    spec:             Optional[ProjectSpec] = None
    stack:            Optional[TechStack] = None
    stack_options:    list[dict]           = Field(default_factory=list)
    status:           ProjectStatus        = "pending"

    # File tree: {relative_path: file_content}
    file_tree:        dict[str, str]       = Field(default_factory=dict)
    file_plan_hint:   list[str]            = Field(default_factory=list)
    generation_log:   list[str]            = Field(default_factory=list)
    error_message:    Optional[str]        = None
    self_heal_attempts: int                = 0

    # GitHub
    github_repo_url:  Optional[str]        = None
    github_repo_name: Optional[str]        = None
    github_owner:     Optional[str]        = None

    # Version tracking
    current_version:  int                  = 0
    pinecone_id:      Optional[str]        = None

    created_at:       datetime             = Field(default_factory=datetime.utcnow)
    updated_at:       datetime             = Field(default_factory=datetime.utcnow)


class ProjectVersion(BaseModel):
    model_config = {"populate_by_name": True, "arbitrary_types_allowed": True}

    id:              Optional[PyObjectId] = Field(default=None, alias="_id")
    project_id:      PyObjectId
    user_id:         PyObjectId
    version:         int
    commit_sha:      Optional[str] = None
    commit_message:  str
    pr_url:          Optional[str] = None
    changes_summary: str           = ""
    file_snapshot:   dict[str, str] = Field(default_factory=dict)
    created_at:      datetime      = Field(default_factory=datetime.utcnow)


# ── Request Schemas ────────────────────────────────────────────

class GenerateProjectRequest(BaseModel):
    prompt:          str            = Field(min_length=3, max_length=5000)
    preferred_stack: Optional[TechStack] = None
    project_name:    Optional[str]  = None


class SelectStackRequest(BaseModel):
    option_id:    str
    custom_stack: Optional[TechStack] = None


class IterateProjectRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=3000)


class ConfirmProjectRequest(BaseModel):
    commit_message: Optional[str] = None
    push_to_github: bool          = True


# ── Response Schemas ───────────────────────────────────────────

class ProjectPublic(BaseModel):
    model_config = {"populate_by_name": True}

    id:              Optional[PyObjectId] = Field(default=None, alias="_id")
    name:            str
    description:     str
    status:          ProjectStatus
    stack:           Optional[TechStack]  = None
    github_repo_url: Optional[str]        = None
    current_version: int
    file_count:      int                  = 0
    created_at:      datetime
    updated_at:      datetime


class ProjectDetail(ProjectPublic):
    """Full project detail including file tree and generation log."""
    original_prompt: str
    spec:            Optional[ProjectSpec]  = None
    file_tree:       dict[str, str]         = Field(default_factory=dict)
    generation_log:  list[str]              = Field(default_factory=list)
    error_message:   Optional[str]          = None


class GenerationStarted(BaseModel):
    project_id: str
    message:    str = "Generation started."


class ProjectVersionPublic(BaseModel):
    model_config = {"populate_by_name": True}

    id:              Optional[PyObjectId] = Field(default=None, alias="_id")
    version:         int
    commit_sha:      Optional[str] = None
    commit_message:  str
    pr_url:          Optional[str] = None
    changes_summary: str
    created_at:      datetime
    
