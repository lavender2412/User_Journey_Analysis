import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.joblib")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray):
        """
        Train Logistic Regression, Random Forest, and XGBoost classifiers.
        Select the best model by ROC-AUC and persist it to disk.

        Parameters
        ----------
        train_array : np.ndarray  shape (n_train, n_features + 1)
            Last column is the binary conversion label.
        test_array  : np.ndarray  shape (n_test,  n_features + 1)

        Returns
        -------
        best_model_name : str
        best_roc_auc    : float
        """
        try:
            logging.info("Splitting train/test arrays into features and labels")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # ------------------------------------------------------------------ #
            # Model catalogue with hyperparameter grids                           #
            # ------------------------------------------------------------------ #
            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000, random_state=42
                ),
                "Random Forest": RandomForestClassifier(
                    n_estimators=200, random_state=42, n_jobs=-1
                ),
                "XGBoost": XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4,
                    eval_metric="logloss",
                    random_state=42,
                    verbosity=0,
                ),
            }

            param_grids = {
                "Logistic Regression": {
                    "C": [0.01, 0.1, 1, 10],
                    "solver": ["lbfgs", "liblinear"],
                },
                "Random Forest": {
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5],
                },
                "XGBoost": {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5],
                    "learning_rate": [0.05, 0.1],
                },
            }

            logging.info("Starting model training and hyper-parameter search")
            report = {}

            for name, model in models.items():
                logging.info(f"Grid-searching {name} ...")
                grid = GridSearchCV(
                    model,
                    param_grids[name],
                    cv=5,
                    scoring="roc_auc",
                    n_jobs=-1,
                    refit=True,
                )
                grid.fit(X_train, y_train)
                best_estimator = grid.best_estimator_

                if hasattr(best_estimator, "predict_proba"):
                    y_prob = best_estimator.predict_proba(X_test)[:, 1]
                else:
                    y_prob = best_estimator.decision_function(X_test)

                auc = roc_auc_score(y_test, y_prob)
                report[name] = {"model": best_estimator, "roc_auc": auc}
                logging.info(
                    f"{name} | best params: {grid.best_params_} | test ROC-AUC: {auc:.4f}"
                )

            # ------------------------------------------------------------------ #
            # Select best model                                                    #
            # ------------------------------------------------------------------ #
            best_name = max(report, key=lambda k: report[k]["roc_auc"])
            best_auc = report[best_name]["roc_auc"]
            best_model = report[best_name]["model"]

            logging.info(f"Best model: {best_name} with ROC-AUC = {best_auc:.4f}")

            if best_auc < 0.60:
                raise CustomException(
                    "No model achieved ROC-AUC above 0.60 — check data quality.",
                    sys,
                )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )
            logging.info(
                f"Best model saved to {self.model_trainer_config.trained_model_file_path}"
            )

            return best_name, best_auc

        except Exception as e:
            raise CustomException(e, sys)
