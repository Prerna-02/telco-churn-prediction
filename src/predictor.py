"""Model loading and prediction.

Kept separate from the frontends so that the FastAPI service (src/app.py) and
the Streamlit UI (src/streamlit_app.py) run the exact same code path.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from src.schema import NUMERIC_NAMES

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "churn_model.pkl"))
METADATA_PATH = MODEL_PATH.parent / "model_metadata.json"

# Fallbacks only for a model saved before metadata existed; train.py always
# writes these, and the values there are chosen on the validation set.
DEFAULT_THRESHOLD = 0.5
DEFAULT_HIGH_FROM = 0.75

_model = None
_metadata = None


class MissingFeatures(Exception):
    """Raised when the caller omitted features the pipeline requires."""

    def __init__(self, missing):
        self.missing = missing
        super().__init__(f"Missing required feature(s): {missing}")


def load_model():
    """Load the pipeline once and cache it for the process lifetime."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def load_metadata() -> Dict[str, Any]:
    """Threshold and risk bands chosen during training."""
    global _metadata
    if _metadata is None:
        try:
            _metadata = json.loads(METADATA_PATH.read_text())
        except (OSError, ValueError):
            _metadata = {}
    return _metadata


def get_threshold() -> float:
    return float(load_metadata().get("threshold", DEFAULT_THRESHOLD))


def model_is_available() -> bool:
    try:
        load_model()
        return True
    except Exception:
        return False


def get_risk_bucket(p: float) -> str:
    """Map churn probability to a risk level.

    Bands derive from the same threshold that decides Yes/No, so the two can
    never disagree — a "Yes" is always Medium or High, never Low.
    """
    bands = load_metadata().get("risk_bands", {})
    medium_from = float(bands.get("medium_from", get_threshold()))
    high_from = float(bands.get("high_from", DEFAULT_HIGH_FROM))

    if p >= high_from:
        return "High"
    elif p >= medium_from:
        return "Medium"
    return "Low"


def predict(features: Dict[str, Any]) -> Dict[str, Any]:
    """Score a single customer.

    `features` must contain every column the pipeline was fitted on. Numeric
    columns are coerced to float, since JSON clients often send them as strings
    and the passthrough transformer would otherwise hand text to the forest.
    """
    model = load_model()
    expected_cols = list(model.feature_names_in_)

    data = {col: features.get(col) for col in expected_cols}

    missing = [col for col, val in data.items() if val is None or val == ""]
    if missing:
        raise MissingFeatures(missing)

    for col in expected_cols:
        if col in NUMERIC_NAMES:
            data[col] = float(data[col])

    df = pd.DataFrame([data])

    proba = float(model.predict_proba(df)[0, 1])
    threshold = get_threshold()

    # Deliberately NOT model.predict(), which hardcodes a 0.5 cutoff. The
    # threshold below was tuned on the validation set to favour recall.
    return {
        "churn_probability": proba,
        "prediction": "Yes" if proba >= threshold else "No",
        "risk_level": get_risk_bucket(proba),
        "threshold": threshold,
    }
