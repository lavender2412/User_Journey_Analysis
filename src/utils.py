import os
import sys
import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    """Serialize and save a Python object to disk using joblib."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(obj, file_path)
        logging.info(f"Object saved to {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """Load a serialized Python object from disk using joblib."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file found at {file_path}")
        obj = joblib.load(file_path)
        logging.info(f"Object loaded from {file_path}")
        return obj
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Train each model in the dict and evaluate ROC-AUC on the test set.

    Parameters
    ----------
    X_train, y_train : training features and labels
    X_test, y_test   : test features and labels
    models           : dict of {model_name: model_instance}

    Returns
    -------
    report : dict of {model_name: roc_auc_score}
    """
    try:
        report = {}
        for name, model in models.items():
            logging.info(f"Training {name} ...")
            model.fit(X_train, y_train)

            # Use predict_proba for probability-based AUC
            if hasattr(model, "predict_proba"):
                y_test_pred = model.predict_proba(X_test)[:, 1]
            else:
                y_test_pred = model.decision_function(X_test)

            auc = roc_auc_score(y_test, y_test_pred)
            report[name] = auc
            logging.info(f"{name} ROC-AUC: {auc:.4f}")

        return report
    except Exception as e:
        raise CustomException(e, sys)
