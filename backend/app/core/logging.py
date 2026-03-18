"""
ChisCode — Structured Logging
Configures structlog with JSON output in production, pretty output in development.
"""
import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from app.core.config import settings


def add_app_context(
    logger: Any, method_name: str, event_dict: EventDict
) -> EventDict:
    """Inject app-level context into every log entry."""
    event_dict["app"] = settings.app_name
    event_dict["version"] = settings.app_version
    event_dict["env"] = settings.app_env
    return event_dict


def setup_logging() -> None:
    """Configure structlog and standard logging."""
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        add_app_context,
    ]

    if settings.is_production:
        # Machine-readable JSON for log aggregators (Datadog, CloudWatch, etc.)
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Human-readable coloured output for local development
        processors = shared_processors + [
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Route standard library logging through structlog
    log_level = logging.DEBUG if settings.debug else logging.INFO
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Silence noisy libraries in production
    if settings.is_production:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("motor").setLevel(logging.WARNING)


def get_logger(name: str = "chiscode") -> structlog.BoundLogger:
    """Return a bound logger with the given name."""
    return structlog.get_logger(name)
