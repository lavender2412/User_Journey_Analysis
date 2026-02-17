import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.joblib")


# Actions tracked in the dataset
JOURNEY_ACTIONS = ["Checkout", "Homepage", "Log in", "Sign up", "Pricing", "Coupon", "Other"]

# Numeric features fed to the model (checkout excluded — used only for labelling)
NUMERIC_FEATURES = [
    "num_sessions",
    "homepage",
    "login",
    "signup",
    "pricing",
    "coupon",
    "other",
    "non_checkout_actions",
    "has_pricing",
    "has_signup",
]
CATEGORICAL_FEATURES = ["subscription_type"]
TARGET_COLUMN = "converted"


def parse_journey(journey_str):
    """
    Count each action type in a single session journey string.

    Example
    -------
    "Homepage-Log in-Pricing-Checkout" ->
        {'checkout': 1, 'homepage': 1, 'login': 1, 'signup': 0, 'pricing': 1,
         'coupon': 0, 'other': 0, 'total': 4}
    """
    actions = journey_str.split("-")
    return {
        "checkout": actions.count("Checkout"),
        "homepage": actions.count("Homepage"),
        "login": actions.count("Log in"),
        "signup": actions.count("Sign up"),
        "pricing": actions.count("Pricing"),
        "coupon": actions.count("Coupon"),
        "other": actions.count("Other"),
        "total": len(actions),
    }


def aggregate_user_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate multi-session, per-row data to one row per user.

    Steps
    -----
    1. Parse journey strings into action counts.
    2. Group by user_id and sum action counts across all sessions.
    3. Derive the conversion label: converted = 1 if user ever visited Checkout.
    4. Build engineered features (binary flags, non-checkout total).
    5. Drop columns not used as model features.
    """
    logging.info("Parsing user journey strings into action counts")
    journey_counts = df["user_journey"].apply(parse_journey).apply(pd.Series)
    df = pd.concat([df.reset_index(drop=True), journey_counts], axis=1)

    logging.info("Aggregating sessions to user level")
    agg_dict = {
        "session_id": "count",        # → num_sessions
        "subscription_type": "first",
        "checkout": "sum",
        "homepage": "sum",
        "login": "sum",
        "signup": "sum",
        "pricing": "sum",
        "coupon": "sum",
        "other": "sum",
        "total": "sum",
    }
    user_df = df.groupby("user_id").agg(agg_dict).reset_index()
    user_df.rename(columns={"session_id": "num_sessions", "total": "total_actions"}, inplace=True)

    # --- Label ---
    user_df[TARGET_COLUMN] = (user_df["checkout"] > 0).astype(int)
    logging.info(
        f"Conversion rate: {user_df[TARGET_COLUMN].mean():.2%} "
        f"({user_df[TARGET_COLUMN].sum()} / {len(user_df)} users)"
    )

    # --- Engineered features ---
    user_df["has_pricing"] = (user_df["pricing"] > 0).astype(int)
    user_df["has_signup"] = (user_df["signup"] > 0).astype(int)
    # Total non-checkout actions capture engagement depth without leaking the label
    user_df["non_checkout_actions"] = user_df["total_actions"] - user_df["checkout"]

    return user_df


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Build a ColumnTransformer that:
        - Imputes + scales numeric features
        - Imputes + one-hot-encodes categorical features
        """
        try:
            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            )
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_pipeline, NUMERIC_FEATURES),
                    ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
                ]
            )
            logging.info(
                f"Numeric features: {NUMERIC_FEATURES}\n"
                f"Categorical features: {CATEGORICAL_FEATURES}"
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Full transformation workflow:
        1. Load raw train / test CSVs (session-level rows).
        2. Aggregate to user level & create labels.
        3. Fit preprocessor on training data; transform both splits.
        4. Save the fitted preprocessor to disk.

        Returns
        -------
        train_arr : np.ndarray  (features + label in last column)
        test_arr  : np.ndarray
        preprocessor_path : str
        """
        try:
            logging.info("Loading train and test dataframes")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Aggregating train data to user level")
            train_user_df = aggregate_user_data(train_df)
            logging.info("Aggregating test data to user level")
            test_user_df = aggregate_user_data(test_df)

            # Separate features and target
            X_train = train_user_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
            y_train = train_user_df[TARGET_COLUMN]
            X_test = test_user_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
            y_test = test_user_df[TARGET_COLUMN]

            logging.info("Fitting preprocessing pipeline on training data")
            preprocessor = self.get_data_transformer_object()
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # Stack features and labels into a single array (label = last column)
            train_arr = np.c_[X_train_transformed, y_train.values]
            test_arr = np.c_[X_test_transformed, y_test.values]

            logging.info("Saving preprocessor object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
