from __future__ import annotations

import os
import sys
from pathlib import Path
import pickle

try:
    import dill  # better for complex Python objects
    _HAVE_DILL = True
except Exception:
    _HAVE_DILL = False

from src.exception import CustomException
from src.logger import get_logger

logger = get_logger("utils")


def _ensure_parent_dir(path: os.PathLike | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_object(file_path: os.PathLike | str, obj) -> None:
    """
    Serialize any Python object to `file_path`.
    Uses dill if available; otherwise falls back to pickle.
    """
    try:
        p = _ensure_parent_dir(file_path)
        serializer = dill if _HAVE_DILL else pickle
        with open(p, "wb") as f:
            serializer.dump(obj, f)
        logger.info("Saved object to %s via %s", p, "dill" if _HAVE_DILL else "pickle")
    except Exception as e:
        logger.exception("Failed saving object to %s", file_path)
        raise CustomException(e, sys) from e


def load_object(file_path: os.PathLike | str):
    """
    Load a Python object previously saved by `save_object`.
    Tries dill first (if installed), otherwise pickle.
    """
    try:
        p = Path(file_path)
        with open(p, "rb") as f:
            if _HAVE_DILL:
                return dill.load(f)
            return pickle.load(f)
    except Exception as e:
        logger.exception("Failed loading object from %s", file_path)
        raise CustomException(e, sys) from e
