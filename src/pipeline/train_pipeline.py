"""
Training pipeline — orchestrates the full end-to-end flow:
    Data Ingestion → Data Transformation → Model Training
"""
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def run_training_pipeline():
    """Run all pipeline stages sequentially and return the best model metrics."""
    try:
        # ── Stage 1: Data Ingestion ─────────────────────────────────────────── #
        logging.info("=== Stage 1: Data Ingestion ===")
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"Train data: {train_data_path} | Test data: {test_data_path}")

        # ── Stage 2: Data Transformation ────────────────────────────────────── #
        logging.info("=== Stage 2: Data Transformation ===")
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = (
            data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )
        )
        logging.info(
            f"Train shape: {train_arr.shape} | Test shape: {test_arr.shape} | "
            f"Preprocessor saved: {preprocessor_path}"
        )

        # ── Stage 3: Model Training ──────────────────────────────────────────── #
        logging.info("=== Stage 3: Model Training ===")
        model_trainer = ModelTrainer()
        best_model_name, best_roc_auc = model_trainer.initiate_model_trainer(
            train_arr, test_arr
        )

        logging.info(
            f"Training pipeline complete. "
            f"Best model: {best_model_name} | ROC-AUC: {best_roc_auc:.4f}"
        )
        return best_model_name, best_roc_auc

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    best_model_name, roc_auc = run_training_pipeline()
    print(f"\nBest model : {best_model_name}")
    print(f"ROC-AUC    : {roc_auc:.4f}")
