# src/utils.py
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

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

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


def evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    models: Dict[str, Any],
    params: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Fit/tune each model with GridSearchCV using params[model_name] if provided.
    Returns:
      - report: {model_name: R^2 on X_test}
      - trained_models: {model_name: best_estimator_fitted}
    Never raises inside the loop—logs and continues on model failure.
    """
    try:
        logger.info("evaluate_models: %d models", len(models))
        scores: Dict[str, float] = {}
        trained: Dict[str, Any] = {}

        for name, base_model in models.items():
            try:
                logger.info("Tuning model: %s", name)
                pgrid = params.get(name, {})

                if pgrid:
                    gs = GridSearchCV(
                        estimator=base_model,
                        param_grid=pgrid,
                        scoring="r2",
                        cv=3,
                        n_jobs=-1,
                        verbose=0,
                    )
                    gs.fit(X_train, y_train)
                    best_model = gs.best_estimator_
                    logger.info("Best params for %s: %s", name, gs.best_params_)
                else:
                    base_model.fit(X_train, y_train)
                    best_model = base_model
                    logger.info("No params for %s; fitted base estimator.", name)

                y_pred = best_model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                scores[name] = float(r2)
                trained[name] = best_model
                logger.info("%s R^2=%.4f", name, r2)

            except Exception as inner_e:
                logger.exception("Model %s failed; skipping", name)
                scores[name] = float("-inf")

        return scores, trained

    except Exception as e:
        logger.exception("evaluate_models failed")
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    # quick smoke test for save/load + evaluate_models (uses a tiny synthetic dataset)
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression

    try:
        X, y = make_regression(n_samples=200, n_features=5, noise=5.0, random_state=42)
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]

        models = {"Linear Regression": LinearRegression()}
        params = {"Linear Regression": {}}  # no tuning

        rep, trained = evaluate_models(X_train, y_train, X_test, y_test, models, params)
        best_name = max(rep, key=rep.get)
        logger.info("Self-test best model: %s (R^2=%.4f)", best_name, rep[best_name])

        # save & load the trained model
        test_model_path = Path("artifacts") / "tmp" / "utils_selftest_model.pkl"
        save_object(test_model_path, trained[best_name])
        loaded = load_object(test_model_path)

        # sanity prediction
        _ = loaded.predict(X_test[:2])
        print("SELF-TEST OK: best =", best_name, "R^2 =", rep[best_name], "saved to", test_model_path)

    except Exception as e:
        raise CustomException(e, sys) from e
