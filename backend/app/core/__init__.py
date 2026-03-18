"""
ChisCode — Core Package
Core configuration, security, logging, and utility modules.
"""

from app.core.config import settings
from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    encrypt_value,
    decrypt_value,
    generate_api_key,
    verify_api_key,
)
from app.core.logging import setup_logging, get_logger
from app.api.deps import (
    get_current_user,
    get_optional_user,
    check_rate_limit,
    require_plan,
)

# Convenience exports
__all__ = [
    # Config
    "settings",
    
    # Security
    "hash_password",
    "verify_password",
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "encrypt_value",
    "decrypt_value",
    "generate_api_key",
    "verify_api_key",
    
    # Logging
    "setup_logging",
    "get_logger",
    
    # Dependencies
    "get_current_user",
    "get_optional_user",
    "check_rate_limit",
    "require_plan",
]
