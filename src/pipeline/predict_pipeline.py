"""
Prediction pipeline — loads the saved preprocessor and model, and serves
conversion probability predictions for new user journeys.
"""
import sys
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.components.data_transformation import (
    parse_journey,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

PREPROCESSOR_PATH = "artifacts/preprocessor.joblib"
MODEL_PATH = "artifacts/model.joblib"


class PredictPipeline:
    """Load artefacts once and reuse for multiple predictions (singleton-friendly)."""

    def __init__(self):
        try:
            self.preprocessor = load_object(PREPROCESSOR_PATH)
            self.model = load_object(MODEL_PATH)
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Transform raw feature DataFrame and return conversion probabilities.

        Parameters
        ----------
        features : pd.DataFrame
            One row per user with columns matching NUMERIC_FEATURES + CATEGORICAL_FEATURES.

        Returns
        -------
        np.ndarray : probability of conversion for each row
        """
        try:
            X = self.preprocessor.transform(features)
            proba = self.model.predict_proba(X)[:, 1]
            return proba
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Maps raw user input (list of session journeys + subscription type) into
    the feature DataFrame expected by PredictPipeline.
    """

    def __init__(self, subscription_type: str, sessions: list[str]):
        """
        Parameters
        ----------
        subscription_type : str
            e.g. "Annual", "Monthly"
        sessions : list[str]
            Each element is a raw journey string for one session,
            e.g. ["Homepage-Log in-Pricing", "Pricing-Sign up-Other"]
        """
        self.subscription_type = subscription_type
        self.sessions = sessions

    def get_data_as_dataframe(self) -> pd.DataFrame:
        """Aggregate session journeys and return a single-row feature DataFrame."""
        try:
            # Aggregate action counts across all sessions
            totals = {
                "checkout": 0, "homepage": 0, "login": 0, "signup": 0,
                "pricing": 0, "coupon": 0, "other": 0, "total": 0,
            }
            for journey in self.sessions:
                counts = parse_journey(journey)
                for key in totals:
                    totals[key] += counts[key]

            num_sessions = len(self.sessions)
            non_checkout_actions = totals["total"] - totals["checkout"]

            row = {
                "num_sessions": num_sessions,
                "homepage": totals["homepage"],
                "login": totals["login"],
                "signup": totals["signup"],
                "pricing": totals["pricing"],
                "coupon": totals["coupon"],
                "other": totals["other"],
                "non_checkout_actions": non_checkout_actions,
                "has_pricing": int(totals["pricing"] > 0),
                "has_signup": int(totals["signup"] > 0),
                "subscription_type": self.subscription_type,
            }
            return pd.DataFrame([row], columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES)

        except Exception as e:
            raise CustomException(e, sys)
