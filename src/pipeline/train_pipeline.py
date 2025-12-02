# src/pipeline/train_pipeline.py
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def main():
    train_path, test_path = DataIngestion().initiate_data_ingestion()
    train_arr, test_arr, pkl_path = DataTransformation().initiate_data_transformation(train_path, test_path)
    print("Preprocessor saved at:", pkl_path)

    r2 = ModelTrainer().initiate_model_trainer(train_arr, test_arr)
    print("R²:", r2)

if __name__ == "__main__":
    main()
