"""Consistent process logging without interfering with library importers."""

import logging
from pathlib import Path


def configure_logging(level: str = "INFO", logfile: Path | None = None) -> None:
    """Configure timestamped console and optional UTF-8 file logging."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile, encoding="utf-8"))
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=handlers, force=True)
