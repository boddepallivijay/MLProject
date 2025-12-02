# src/pipeline/predict_pipeline.py
from __future__ import annotations
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

from src.exception import CustomException
from src.logger import get_logger
from src.utils import load_object

logger = get_logger("predict_pipeline")

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
PREPROC_PATH  = ARTIFACTS_DIR / "preprocessor.pkl"
MODEL_PATH    = ARTIFACTS_DIR / "model.pkl"

# NOTE: We trained the preprocessor on SNAKE_CASE columns
SNAKE_CASE_COLS = [
    "gender",
    "race_ethnicity",
    "parental_level_of_education",
    "lunch",
    "test_preparation_course",
    "reading_score",
    "writing_score",
]

class PredictPipeline:
    def __init__(self) -> None:
        if not PREPROC_PATH.exists() or not MODEL_PATH.exists():
            raise CustomException(
                f"Artifacts missing. Expected {PREPROC_PATH} and {MODEL_PATH}"
            )
        self.preprocessor = load_object(PREPROC_PATH)
        self.model = load_object(MODEL_PATH)
        logger.info("Loaded preprocessor & model.")

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        try:
            # ensure columns in correct order/names
            df = features_df.copy()
            df.columns = [c.strip().lower().replace("/", "_").replace(" ", "_") for c in df.columns]
            missing = [c for c in SNAKE_CASE_COLS if c not in df.columns]
            if missing:
                raise CustomException(f"Missing columns for inference: {missing}")

            X = df[SNAKE_CASE_COLS]
            X_t = self.preprocessor.transform(X)
            preds = self.model.predict(X_t)
            return preds
        except Exception as e:
            raise CustomException(e, sys) from e


class CustomData:
    """
    Accepts raw form inputs and produces a one-row DataFrame with SNAKE_CASE columns.
    """
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: float,
        writing_score: float,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self) -> pd.DataFrame:
        data = {
            "gender": [self.gender],
            "race_ethnicity": [self.race_ethnicity],
            "parental_level_of_education": [self.parental_level_of_education],
            "lunch": [self.lunch],
            "test_preparation_course": [self.test_preparation_course],
            "reading_score": [float(self.reading_score)],
            "writing_score": [float(self.writing_score)],
        }
        return pd.DataFrame(data)
