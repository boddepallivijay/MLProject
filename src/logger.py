# src/logger.py
from __future__ import annotations
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime

def get_logger(name: str = "mlproject", base_dir: str | Path | None = None) -> logging.Logger:
    """
    Create (or return) a configured logger.
    - Logs go to <project_root>/logs/YYYY_MM_DD_HH_MM_SS.log and to console.
    - Safe on re-imports: it won't add duplicate handlers.
    """
    logger = logging.getLogger(name)
    if logger.handlers:      # already configured for this name
        return logger

    base_dir = Path.cwd() if base_dir is None else Path(base_dir)
    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{datetime.now():%Y_%m_%d_%H_%M_%S}.log"

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_h = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    file_h.setFormatter(fmt)
    file_h.setLevel(logging.INFO)

    console_h = logging.StreamHandler()
    console_h.setFormatter(fmt)
    console_h.setLevel(logging.INFO)

    logger.setLevel(logging.INFO)
    logger.addHandler(file_h)
    logger.addHandler(console_h)
    logger.propagate = False

    logger.info("Logger initialized. Writing to %s", log_file)
    return logger
