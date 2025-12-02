from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from typing import Dict, Any, Tuple, Optional

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import get_logger
from src.utils import save_object, evaluate_models

logger = get_logger("model_trainer")


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array) -> float:
        """
        Train/tune multiple regressors, pick the best by validation score,
        persist it to artifacts/model.pkl, and return its R² on the test set.
        """
        try:
            logger.info("Splitting train/test arrays into X/y")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Candidate models
            models: Dict[str, Any] = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(verbosity=0),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                # You listed KNN import; add if you want:
                # "KNN Regressor": KNeighborsRegressor(),
            }

            # Hyperparameters (keep modest to avoid long runs)
            params: Dict[str, Dict[str, Any]] = {
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                },
                "Random Forest": {
                    "n_estimators": [32, 64, 128, 256],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.05, 0.01],
                    "subsample": [0.7, 0.85, 1.0],
                    "n_estimators": [64, 128, 256],
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.05, 0.01],
                    "n_estimators": [64, 128, 256],
                },
                "CatBoosting Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.1, 0.05, 0.01],
                    "iterations": [50, 100, 200],
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.5, 0.1, 0.01],
                    "n_estimators": [64, 128, 256],
                },
                # "KNN Regressor": {
                #     "n_neighbors": [3, 5, 7, 9],
                #     "weights": ["uniform", "distance"],
                # }
            }

            logger.info("Evaluating models with hyperparameters...")
            # Be compatible with evaluate_models that returns either:
            #   (a) score_dict
            #   (b) (score_dict, trained_models_dict)
            result = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params,            # <-- use 'params'
            )

            if isinstance(result, tuple) and len(result) == 2:
                model_report, trained_models = result  # type: ignore
            else:
                model_report = result
                trained_models = None  # type: ignore

            if not isinstance(model_report, dict) or not model_report:
                raise CustomException("Model evaluation returned an empty report")

            # Pick best model name & score
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            logger.info("Best model: %s (score=%.4f)", best_model_name, best_model_score)

            # Guardrail: require a minimum validation quality
            if best_model_score < 0.60:
                raise CustomException(
                    f"No sufficiently good model found (best score={best_model_score:.3f} < 0.60)"
                )

            # Get the trained/tuned best model if provided
            if trained_models and best_model_name in trained_models:
                best_model = trained_models[best_model_name]
                logger.info("Using tuned/trained best model from evaluate_models.")
            else:
                # Fallback: fit the chosen base estimator on training data
                logger.info("No trained model returned; fitting %s on the training data.", best_model_name)
                best_model = models[best_model_name]
                best_model.fit(X_train, y_train)

            # Persist the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )
            logger.info("Saved best model to %s", self.model_trainer_config.trained_model_file_path)

            # Final hold-out evaluation
            y_pred = best_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            logger.info("Final R² on test: %.4f", r2)
            return float(r2)

        except Exception as e:
            logger.exception("Model training failed")
            raise CustomException(e, sys) from e
