"""
ChisCode — Core Configuration
Loads and validates all environment variables at startup.
"""
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        
    )

    # ── App ──────────────────────────────────────────────────
    app_name: str = "ChisCode"
    app_env: Literal["development", "staging", "production"] = "development"
    app_version: str = "0.1.0"
    secret_key: str = Field(..., min_length=32)
    debug: bool = False
    allowed_hosts: str = "localhost,127.0.0.1"
    port: int = 8000

    # ── AI / LLM ─────────────────────────────────────────────
    codestral_api_key: str = Field(default="")
    codestral_base_url: str = "https://codestral.mistral.ai/v1"
    codestral_model: str = "codestral-latest"

    # LangSmith
    langchain_api_key: str = Field(default="")
    langchain_tracing_v2: bool = True
    langchain_project: str = "chiscode"
    langchain_endpoint: str = "https://api.smith.langchain.com"

    # ── Databases ────────────────────────────────────────────
    mongodb_url: str = Field(default="mongodb://localhost:27017")
    mongodb_db: str = "chiscode"

    pinecone_api_key: str = Field(default="")
    pinecone_index: str = "chiscode-embeddings"
    pinecone_environment: str = "us-east-1"

        # ── Upstash Redis (HTTP REST SDK) ────────────────────────────
    # Set both in HF Spaces → Settings → Variables and Secrets:
    #   UPSTASH_REDIS_REST_URL   = https://growing-ladybug-64231.upstash.io
    #   UPSTASH_REDIS_REST_TOKEN = <your-token>
    upstash_redis_rest_url:   str = Field(default="")
    upstash_redis_rest_token: str = Field(default="")

    # ── Auth ─────────────────────────────────────────────────
    github_client_id: str = Field(default="")
    github_client_secret: str = Field(default="")
    github_redirect_uri: str = "http://localhost:8000/api/v1/auth/github/callback"

    jwt_secret_key: str = Field(default="")
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 1440   # 24h
    jwt_refresh_token_expire_days: int = 30

    # ── Payments ─────────────────────────────────────────────
    polar_access_token: str = Field(default="")          # Polar API key
    polar_webhook_secret: str = Field(default="")        # Webhook secret from Polar dashboard
    polar_organization_id: str = Field(default="")       # Your Polar org ID

    # Product IDs from Polar dashboard (Settings → Products)
    polar_product_basic:  str = Field(default="")        # Basic plan product ID
    polar_product_pro:    str = Field(default="")        # Pro plan product ID
    polar_product_yearly: str = Field(default="")        # Yearly plan product ID
    
    # ── Search ───────────────────────────────────────────────
    duckduckgo_max_results: int = 10

    # ── Rate Limits (requests/day) ───────────────────────────
    rate_limit_free: int = 5
    rate_limit_basic: int = 100
    rate_limit_pro: int = 1000
    rate_limit_yearly: int = 1000

    # ── Frontend ─────────────────────────────────────────────
    frontend_base_url: str = "http://localhost:8000"

    @field_validator("allowed_hosts", mode="before")
    @classmethod
    def parse_allowed_hosts(cls, v: object) -> str:
        # Always normalise to a comma-separated string.
        # pydantic-settings v2 tries to JSON-parse list fields from env vars
        # before validators run — storing as str bypasses that entirely.
        if isinstance(v, list):
            return ",".join(str(h) for h in v)
        return str(v) if v else "localhost,127.0.0.1"

    @property
    def allowed_hosts_list(self) -> list[str]:
        """Split the comma-separated string into a list for middleware use."""
        return [h.strip() for h in self.allowed_hosts.split(",") if h.strip()]

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    def get_rate_limit(self, plan: str) -> int:
        """Return daily request limit for the given subscription plan."""
        return {
            "free": self.rate_limit_free,
            "basic": self.rate_limit_basic,
            "pro": self.rate_limit_pro,
            "yearly": self.rate_limit_yearly,
        }.get(plan, self.rate_limit_free)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings instance — called once at startup."""
    return Settings()


settings = get_settings()
  
