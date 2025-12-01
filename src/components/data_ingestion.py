# src/components/data_ingestion.py
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import get_logger  # <-- use our configured logger

# paths (assumes you run from project root: D:\Data science\MLProject)
PROJECT_ROOT = Path.cwd()
DATA_SRC     = PROJECT_ROOT / r"notebook\data\stud.csv"  # <-- relative inside repo
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

logger = get_logger("data_ingestion")  # logs to <project_root>/logs/....log

@dataclass
class DataIngestionConfig:
    train_data_path: Path = ARTIFACT_DIR / "train.csv"
    test_data_path:  Path = ARTIFACT_DIR / "test.csv"
    raw_data_path:   Path = ARTIFACT_DIR / "raw.csv"


class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self) -> tuple[str, str]:
        logger.info("Entered the data ingestion component")

        try:
            # ensure artifacts directory exists
            self.ingestion_config.train_data_path.parent.mkdir(parents=True, exist_ok=True)

            # read source CSV
            if not DATA_SRC.exists():
                raise FileNotFoundError(f"Input CSV not found at {DATA_SRC}")
            logger.info("Reading dataset from: %s", DATA_SRC)
            df = pd.read_csv(DATA_SRC)
            logger.info("Read %d rows x %d columns", df.shape[0], df.shape[1])

            # save raw
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logger.info("Wrote raw data to: %s", self.ingestion_config.raw_data_path)

            # split
            logger.info("Train/test split started (test_size=0.2, random_state=42)")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # write train & test
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            logger.info(
                "Saved train to %s and test to %s",
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

            logger.info("Ingestion completed successfully")
            return (str(self.ingestion_config.train_data_path), str(self.ingestion_config.test_data_path))

        except Exception as e:
            logger.exception("Ingestion failed")
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    DataIngestion().initiate_data_ingestion()
