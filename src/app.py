from typing import Any, Dict

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from src.predictor import MissingFeatures, model_is_available
from src.predictor import predict as run_prediction
from src.schema import EXAMPLE_CUSTOMER

app = FastAPI(
    title="Telco Churn Prediction API",
    description="Scores a single telecom customer for churn risk.",
    version="1.0.0",
)


@app.get("/", include_in_schema=False)
def root():
    """Send visitors straight to the interactive docs."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Simple health endpoint for ECS / load balancer checks."""
    if not model_is_available():
        return {"status": "degraded", "detail": "Model not loaded"}
    return {"status": "ok"}


@app.post("/predict")
def predict(features: Dict[str, Any] = Body(..., example=EXAMPLE_CUSTOMER)):
    """
    Predict churn for a single customer.

    Request body: JSON with the same feature names used during training
    (all columns except 'customerID' and 'Churn').
    """
    if not model_is_available():
        raise HTTPException(status_code=500, detail="Model is not loaded")

    try:
        return run_prediction(features)
    except MissingFeatures as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid feature value: {exc}")
