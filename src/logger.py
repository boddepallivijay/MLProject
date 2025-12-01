# logger.py
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / (datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".log")

def get_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:  # avoid duplicate handlers on re-runs
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        fh = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=5)
        fh.setFormatter(fmt)
        fh.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
        ch.setLevel(logging.INFO)

        logger.addHandler(fh)
        logger.addHandler(ch)
        logger.propagate = False

    return logger
