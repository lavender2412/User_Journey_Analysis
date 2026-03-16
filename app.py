"""
app.py
──────
Flask REST API for the User Journey Funnel Stage Predictor.

Endpoints:
    POST /predict   — predict funnel stage from a user journey string
    GET  /health    — AWS Elastic Beanstalk health check

Usage (local):
    python app.py

Usage (production via gunicorn):
    gunicorn app:app --workers 2 --bind 0.0.0.0:8080

Example request:
    curl -X POST http://localhost:5000/predict \
         -H "Content-Type: application/json" \
         -d '{"user_journey": "Homepage-Pricing-Sign up-Log in-Coupon",
              "subscription_type": "Annual"}'
"""

import os
import joblib
import numpy as np
from flask import Flask, request, jsonify

from feature_utils import engineer_single, FUNNEL_STAGE_NAMES, STAGE_RECOMMENDATIONS

# ── App setup ─────────────────────────────────────────────────────────────────

app = Flask(__name__)

# ── Load model artifact once at startup (not per request) ─────────────────────
# Loading is expensive (~100ms); keeping it in memory makes inference fast (~1ms).

ARTIFACT_PATH = os.environ.get(
    'MODEL_ARTIFACT_PATH',
    'model_artifacts/funnel_model.joblib'
)

try:
    artifact      = joblib.load(ARTIFACT_PATH)
    MODEL         = artifact['model']
    SCALER        = artifact.get('scaler')
    NEEDS_SCALING = artifact.get('needs_scaling', False)
    MODEL_NAME    = artifact.get('model_name', 'Unknown')
    FEATURE_NAMES = artifact.get('feature_names', [])
    print(f'[startup] Model loaded: {MODEL_NAME}')
    print(f'[startup] Features    : {len(FEATURE_NAMES)}')
    print(f'[startup] ROC-AUC     : {artifact.get("test_roc_auc", "n/a")}')
except FileNotFoundError:
    print(f'[startup] WARNING: artifact not found at {ARTIFACT_PATH}')
    print('[startup] Run model_training.ipynb Cell 10 first to generate it.')
    artifact = MODEL = SCALER = None
    NEEDS_SCALING = False
    MODEL_NAME    = 'not loaded'
    FEATURE_NAMES = []


# ── Helpers ───────────────────────────────────────────────────────────────────

VALID_SUBSCRIPTION_TYPES = {'annual', 'monthly', 'quarterly'}


def validate_input(data: dict) -> tuple[bool, str]:
    """
    Validate the incoming JSON payload.

    Returns:
        (is_valid: bool, error_message: str)
    """
    if 'user_journey' not in data:
        return False, "'user_journey' field is required"

    journey = data['user_journey']

    if not isinstance(journey, str) or not journey.strip():
        return False, "'user_journey' must be a non-empty string"

    if len(journey) > 10_000:
        return False, "'user_journey' exceeds maximum allowed length (10,000 chars)"

    sub_type = data.get('subscription_type', 'Quarterly')
    if sub_type.lower() not in VALID_SUBSCRIPTION_TYPES:
        return False, (
            f"'subscription_type' must be one of: "
            f"{sorted(VALID_SUBSCRIPTION_TYPES)}"
        )

    return True, ''


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    """
    AWS Elastic Beanstalk health check endpoint.
    Returns 200 if the model is loaded and ready, 503 otherwise.
    """
    if MODEL is None:
        return jsonify({
            'status':  'unhealthy',
            'reason':  'model artifact not loaded',
            'model':   MODEL_NAME,
        }), 503

    return jsonify({
        'status': 'healthy',
        'model':  MODEL_NAME,
        'roc_auc': artifact.get('test_roc_auc'),
        'weighted_f1': artifact.get('weighted_f1'),
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the funnel stage for a single user journey.

    Request body (JSON):
        user_journey      : str   — page sequence, dash-separated
                                    e.g. "Homepage-Pricing-Sign up-Log in"
        subscription_type : str   — "Annual" | "Monthly" | "Quarterly"  (optional, default Quarterly)

    Response body (JSON):
        funnel_stage      : int   — predicted stage (0–3)
        stage_label       : str   — human-readable stage name
        recommendation    : str   — suggested business action
        probabilities     : dict  — P(stage) for each of the 4 stages
        model             : str   — model name (for transparency)
    """
    # ── 1. Parse & validate ───────────────────────────────────────────────────
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 400

    data = request.get_json()

    is_valid, error_msg = validate_input(data)
    if not is_valid:
        return jsonify({'error': error_msg}), 422

    if MODEL is None:
        return jsonify({'error': 'model not loaded — run training notebook first'}), 503

    journey       = data['user_journey'].strip()
    subscription  = data.get('subscription_type', 'Quarterly')

    # ── 2. Feature engineering ────────────────────────────────────────────────
    # engineer_single handles the full preprocessing pipeline for one user:
    # deduplication is NOT applied here — the API accepts already-concatenated
    # journeys. For session-level input, deduplicate before calling the API.
    try:
        X_input = engineer_single(journey, subscription_type=subscription)
    except Exception as e:
        return jsonify({'error': f'feature engineering failed: {str(e)}'}), 500

    # Reindex to guarantee column order matches training
    # (safety net in case feature_utils is updated without retraining)
    X_input = X_input.reindex(columns=FEATURE_NAMES, fill_value=0)

    # ── 3. Scale if required (Logistic Regression needs StandardScaler) ───────
    X_inference = SCALER.transform(X_input) if NEEDS_SCALING else X_input

    # ── 4. Predict ────────────────────────────────────────────────────────────
    try:
        stage_idx  = int(MODEL.predict(X_inference)[0])
        raw_probs  = MODEL.predict_proba(X_inference)[0]
    except Exception as e:
        return jsonify({'error': f'prediction failed: {str(e)}'}), 500

    # ── 5. Build response ─────────────────────────────────────────────────────
    probabilities = {
        FUNNEL_STAGE_NAMES[i]: round(float(p), 4)
        for i, p in enumerate(raw_probs)
    }

    response = {
        'funnel_stage':   stage_idx,
        'stage_label':    FUNNEL_STAGE_NAMES[stage_idx],
        'recommendation': STAGE_RECOMMENDATIONS[stage_idx],
        'probabilities':  probabilities,
        'model':          MODEL_NAME,
    }

    return jsonify(response), 200


@app.route('/model-info', methods=['GET'])
def model_info():
    """
    Return metadata about the currently loaded model.
    Useful for debugging and monitoring in production.
    """
    if MODEL is None:
        return jsonify({'error': 'model not loaded'}), 503

    return jsonify({
        'model_name':     MODEL_NAME,
        'feature_count':  len(FEATURE_NAMES),
        'num_classes':    4,
        'stage_labels':   FUNNEL_STAGE_NAMES,
        'test_roc_auc':   artifact.get('test_roc_auc'),
        'weighted_f1':    artifact.get('weighted_f1'),
        'needs_scaling':  NEEDS_SCALING,
    }), 200


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port  = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
