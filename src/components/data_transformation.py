# src/components/data_transformation.py
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import get_logger
from src.utils import save_object

logger = get_logger("data_transformation")

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

def _to_snake(name: str) -> str:
    return (
        name.strip()
            .lower()
            .replace("/", "_")
            .replace(" ", "_")
    )

def _standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: _to_snake(c) for c in df.columns})

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Pipelines expect snake_case column names.
        """
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ("scaler", StandardScaler()),
                ]
            )

            logger.info("Categorical columns: %s", categorical_columns)
            logger.info("Numerical columns: %s", numerical_columns)

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ],
                remainder="drop",
            )
            return preprocessor

        except Exception as e:
            logger.exception("Failed building preprocessing object")
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logger.info("Read train (%s) and test (%s)", train_path, test_path)

            # Standardize column names to snake_case
            train_df = _standardize_cols(train_df)
            test_df  = _standardize_cols(test_df)
            logger.info("Standardized columns: %s", train_df.columns.tolist())

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"   # snake_case after standardization

            X_train = train_df.drop(columns=[target_column_name], axis=1)
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=[target_column_name], axis=1)
            y_test = test_df[target_column_name]

            logger.info("Fitting preprocessor on training features and transforming both train/test.")
            X_train_t = preprocessing_obj.fit_transform(X_train)
            X_test_t = preprocessing_obj.transform(X_test)

            if not isinstance(X_train_t, np.ndarray):
                X_train_t = np.asarray(X_train_t)
            if not isinstance(X_test_t, np.ndarray):
                X_test_t = np.asarray(X_test_t)

            train_arr = np.c_[X_train_t, y_train.to_numpy()]
            test_arr  = np.c_[X_test_t,  y_test.to_numpy()]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )
            logger.info("Saved preprocessing object to %s", self.data_transformation_config.preprocessor_obj_file_path)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logger.exception("Data transformation failed")
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    try:
        project_root = Path(__file__).resolve().parents[2]
        train_csv = project_root / "artifacts" / "train.csv"
        test_csv  = project_root / "artifacts" / "test.csv"

        if not train_csv.exists() or not test_csv.exists():
            raise FileNotFoundError(
                f"Expected train/test at {train_csv} and {test_csv}. "
                "Run ingestion first: python -m src.components.data_ingestion"
            )

        dt = DataTransformation()
        train_arr, test_arr, pkl_path = dt.initiate_data_transformation(str(train_csv), str(test_csv))
        logger.info("Transformation OK. train_arr=%s, test_arr=%s, pkl=%s",
                    train_arr.shape, test_arr.shape, pkl_path)
        print("OK:", train_arr.shape, test_arr.shape, pkl_path)
    except Exception as e:
        raise CustomException(e, sys) from e
