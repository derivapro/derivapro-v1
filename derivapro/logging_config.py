import logging
import os


DEFAULT_LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def _resolve_log_level(level_name: str) -> int:
    normalized = (level_name or DEFAULT_LOG_LEVEL).upper()
    return getattr(logging, normalized, logging.INFO)


def configure_logging() -> None:
    root_logger = logging.getLogger()

    if root_logger.handlers:
        return

    log_level_name = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL)
    log_level = _resolve_log_level(log_level_name)

    logging.basicConfig(level=log_level, format=LOG_FORMAT)
    root_logger.setLevel(log_level)
