"""Logging initialisation.

Pattern adapted from metascreener (小组项目/SRC/) structlog setup.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
) -> None:
    """Configure structlog + stdlib logging for the application.

    Args:
        level: Logging level name (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to a log file. Parent dirs are created
                  automatically.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_file), encoding="utf-8"))

    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        handlers=handlers,
        force=True,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
