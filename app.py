"""
Flask REST API — serves real-time conversion probability predictions
for new user journeys.

Endpoints
---------
GET  /              health check
POST /predict       predict conversion probability for a user journey
POST /predict/batch predict for multiple users in one call

Request body for /predict
-------------------------
{
    "subscription_type": "Annual",
    "sessions": [
        "Homepage-Log in-Pricing",
        "Pricing-Sign up-Other"
    ]
}

Response
--------
{
    "conversion_probability": 0.72,
    "predicted_label": 1
}
"""
import os
import traceback

from flask import Flask, request, jsonify

from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.logger import logging

# AWS Elastic Beanstalk expects the Flask app to be named 'application'
application = Flask(__name__)
app = application  # alias for local dev convenience

# Load artefacts once at startup
_pipeline = None


def get_pipeline() -> PredictPipeline:
    """Lazy-load the prediction pipeline (artefacts must exist in artifacts/)."""
    global _pipeline
    if _pipeline is None:
        _pipeline = PredictPipeline()
    return _pipeline


# ── Routes ─────────────────────────────────────────────────────────────────── #

@app.route("/", methods=["GET"])
def health_check():
    """Simple liveness probe."""
    return jsonify({"status": "ok", "service": "user-journey-conversion-api"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict conversion probability for a single user.

    Expected JSON body:
        subscription_type : str   — e.g. "Annual"
        sessions          : list  — list of raw journey strings
    """
    try:
        body = request.get_json(force=True)
        if not body:
            return jsonify({"error": "Request body must be JSON"}), 400

        subscription_type = body.get("subscription_type")
        sessions = body.get("sessions")

        if not subscription_type:
            return jsonify({"error": "Missing field: subscription_type"}), 400
        if not sessions or not isinstance(sessions, list) or len(sessions) == 0:
            return jsonify({"error": "Missing or empty field: sessions (must be a non-empty list)"}), 400

        custom_data = CustomData(
            subscription_type=subscription_type,
            sessions=sessions,
        )
        features_df = custom_data.get_data_as_dataframe()
        pipeline = get_pipeline()
        proba = pipeline.predict(features_df)[0]

        logging.info(
            f"Prediction: subscription={subscription_type} "
            f"sessions={len(sessions)} → prob={proba:.4f}"
        )

        return jsonify(
            {
                "conversion_probability": round(float(proba), 4),
                "predicted_label": int(proba >= 0.5),
            }
        ), 200

    except FileNotFoundError as e:
        return jsonify(
            {"error": "Model artefacts not found. Run the training pipeline first.", "detail": str(e)}
        ), 503
    except Exception:
        logging.error(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Predict conversion probability for multiple users in one request.

    Expected JSON body:
        users : list of objects, each with subscription_type and sessions fields
    """
    try:
        body = request.get_json(force=True)
        if not body or "users" not in body:
            return jsonify({"error": "Expected JSON with a 'users' list"}), 400

        users = body["users"]
        if not isinstance(users, list) or len(users) == 0:
            return jsonify({"error": "'users' must be a non-empty list"}), 400

        pipeline = get_pipeline()
        results = []

        for i, user in enumerate(users):
            subscription_type = user.get("subscription_type")
            sessions = user.get("sessions", [])
            if not subscription_type or not sessions:
                results.append({"index": i, "error": "Missing subscription_type or sessions"})
                continue

            custom_data = CustomData(subscription_type=subscription_type, sessions=sessions)
            features_df = custom_data.get_data_as_dataframe()
            proba = pipeline.predict(features_df)[0]
            results.append(
                {
                    "index": i,
                    "conversion_probability": round(float(proba), 4),
                    "predicted_label": int(proba >= 0.5),
                }
            )

        return jsonify({"predictions": results}), 200

    except FileNotFoundError as e:
        return jsonify(
            {"error": "Model artefacts not found. Run the training pipeline first.", "detail": str(e)}
        ), 503
    except Exception:
        logging.error(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500


# ── Entry point ─────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
