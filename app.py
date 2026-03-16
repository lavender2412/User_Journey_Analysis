"""
app.py
──────
FastAPI REST API for the User Journey Funnel Stage Predictor.

Endpoints:
    POST /predict    — predict funnel stage from a user journey string
    GET  /health     — AWS Elastic Beanstalk health check
    GET  /model-info — model metadata

Interactive docs (Swagger UI):
    http://localhost:8000/docs          (local)
    http://<your-eb-url>/docs           (production)

Usage (local):
    uvicorn app:app --reload --port 8000

Usage (production via uvicorn):
    uvicorn app:app --workers 2 --host 0.0.0.0 --port 8000

Example request:
    curl -X POST http://localhost:8000/predict \\
         -H "Content-Type: application/json" \\
         -d '{"user_journey": "Homepage-Pricing-Sign up-Log in-Coupon",
              "subscription_type": "Annual"}'
"""

import os
from contextlib import asynccontextmanager
from typing import Dict, Optional

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from feature_utils import engineer_single, FUNNEL_STAGE_NAMES, STAGE_RECOMMENDATIONS


# ── Globals (populated at startup) ────────────────────────────────────────────

ARTIFACT_PATH = os.environ.get(
    'MODEL_ARTIFACT_PATH',
    'model_artifacts/funnel_model.joblib'
)

artifact      = None
MODEL         = None
SCALER        = None
NEEDS_SCALING = False
MODEL_NAME    = 'not loaded'
FEATURE_NAMES = []


# ── Lifespan: load model once at startup ──────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model artifact when the server starts, clean up when it stops."""
    global artifact, MODEL, SCALER, NEEDS_SCALING, MODEL_NAME, FEATURE_NAMES
    try:
        artifact      = joblib.load(ARTIFACT_PATH)
        MODEL         = artifact['model']
        SCALER        = artifact.get('scaler')
        NEEDS_SCALING = artifact.get('needs_scaling', False)
        MODEL_NAME    = artifact.get('model_name', 'Unknown')
        FEATURE_NAMES = artifact.get('feature_names', [])
        print(f'[startup] Model loaded : {MODEL_NAME}')
        print(f'[startup] Features     : {len(FEATURE_NAMES)}')
        print(f'[startup] ROC-AUC      : {artifact.get("test_roc_auc", "n/a")}')
    except FileNotFoundError:
        print(f'[startup] WARNING: artifact not found at {ARTIFACT_PATH}')
        print('[startup] Run model_training.ipynb Cell 10 to generate it.')
    yield
    # Nothing to clean up for a pure in-memory model


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = 'User Journey Funnel Predictor',
    description = (
        'Predicts which **funnel stage** a user is in based on their '
        'clickstream journey.\n\n'
        '**Funnel stages:**\n'
        '- `0` Browsing — just exploring, no purchase signals\n'
        '- `1` Abandoned — showed intent but left\n'
        '- `2` Interested — strong engagement, not yet converted\n'
        '- `3` Converted — purchased or earned a certificate\n\n'
        'Try it out using the **POST /predict** endpoint below — '
        'paste any dash-separated page journey and hit **Execute**.'
    ),
    version     = '1.0.0',
    lifespan    = lifespan,
)


# ── Pydantic models ───────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    user_journey: str = Field(
        ...,
        description='Dash-separated sequence of pages visited by the user.',
        examples=['Homepage-Pricing-Sign up-Log in-Coupon-Checkout'],
        min_length=1,
        max_length=10_000,
    )
    subscription_type: Optional[str] = Field(
        default='Quarterly',
        description='User subscription plan. One of: Annual, Monthly, Quarterly.',
        examples=['Annual'],
    )

    @field_validator('subscription_type')
    @classmethod
    def validate_subscription(cls, v: str) -> str:
        valid = {'annual', 'monthly', 'quarterly'}
        if v.lower() not in valid:
            raise ValueError(f"subscription_type must be one of: {sorted(valid)}")
        return v


class PredictResponse(BaseModel):
    funnel_stage:   int            = Field(..., description='Predicted stage index (0–3)')
    stage_label:    str            = Field(..., description='Human-readable stage name')
    recommendation: str            = Field(..., description='Suggested business action')
    probabilities:  Dict[str, float] = Field(..., description='Probability per stage')
    model:          str            = Field(..., description='Model used for prediction')


class HealthResponse(BaseModel):
    status:      str
    model:       str
    roc_auc:     Optional[float] = None
    weighted_f1: Optional[float] = None


class ModelInfoResponse(BaseModel):
    model_name:    str
    feature_count: int
    num_classes:   int
    stage_labels:  Dict[int, str]
    test_roc_auc:  Optional[float]
    weighted_f1:   Optional[float]
    needs_scaling: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get(
    '/health',
    response_model=HealthResponse,
    summary='Health check',
    description='Returns 200 if the model is loaded and ready. Used by AWS EB health checks.',
    tags=['Monitoring'],
)
def health():
    if MODEL is None:
        raise HTTPException(status_code=503, detail='model artifact not loaded')
    return HealthResponse(
        status      = 'healthy',
        model       = MODEL_NAME,
        roc_auc     = artifact.get('test_roc_auc'),
        weighted_f1 = artifact.get('weighted_f1'),
    )


@app.post(
    '/predict',
    response_model=PredictResponse,
    summary='Predict funnel stage',
    description=(
        'Predicts the funnel stage for a single user based on their page journey.\n\n'
        '**Try it:** paste a dash-separated journey like '
        '`Homepage-Pricing-Coupon-Checkout` and hit Execute.'
    ),
    tags=['Prediction'],
)
def predict(request: PredictRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail='model not loaded — run training notebook first')

    # ── Feature engineering ───────────────────────────────────────────────────
    try:
        X_input = engineer_single(
            journey           = request.user_journey.strip(),
            subscription_type = request.subscription_type,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'feature engineering failed: {e}')

    # Reindex to guarantee column order matches training
    X_input = X_input.reindex(columns=FEATURE_NAMES, fill_value=0)

    # ── Scale if required (Logistic Regression uses StandardScaler) ───────────
    X_inference = SCALER.transform(X_input) if NEEDS_SCALING else X_input

    # ── Predict ───────────────────────────────────────────────────────────────
    try:
        stage_idx = int(MODEL.predict(X_inference)[0])
        raw_probs = MODEL.predict_proba(X_inference)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'prediction failed: {e}')

    # ── Build response ────────────────────────────────────────────────────────
    return PredictResponse(
        funnel_stage   = stage_idx,
        stage_label    = FUNNEL_STAGE_NAMES[stage_idx],
        recommendation = STAGE_RECOMMENDATIONS[stage_idx],
        probabilities  = {
            FUNNEL_STAGE_NAMES[i]: round(float(p), 4)
            for i, p in enumerate(raw_probs)
        },
        model = MODEL_NAME,
    )


@app.get(
    '/model-info',
    response_model=ModelInfoResponse,
    summary='Model metadata',
    description='Returns information about the currently loaded model — useful for debugging and monitoring.',
    tags=['Monitoring'],
)
def model_info():
    # Always return 200 — useful for diagnosing startup failures.
    # When no model is loaded, fields default to safe zero/None values.
    return ModelInfoResponse(
        model_name    = MODEL_NAME,
        feature_count = len(FEATURE_NAMES),
        num_classes   = 4,
        stage_labels  = FUNNEL_STAGE_NAMES,
        test_roc_auc  = artifact.get('test_roc_auc') if artifact else None,
        weighted_f1   = artifact.get('weighted_f1')  if artifact else None,
        needs_scaling = NEEDS_SCALING,
    )


# ── Entry point (local dev) ────────────────────────────────────────────────────

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run('app:app', host='0.0.0.0', port=port, reload=True)
