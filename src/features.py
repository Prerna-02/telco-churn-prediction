"""Feature engineering.

These functions run *inside* the sklearn Pipeline as a FunctionTransformer, so
the transformation is saved into churn_model.pkl and travels with the model.
That means callers (the API, the Streamlit form) keep sending the same raw
columns and can never drift out of sync with training.

Must stay importable at `src.features` — joblib pickles the transformer by
reference, so renaming this module or these functions breaks existing models.
"""

import numpy as np
import pandas as pd

ADDON_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Engineered column names, exposed so the ColumnTransformer can select them.
ENGINEERED_NUMERIC = ["num_addons", "avg_monthly_spend", "charges_delta"]
ENGINEERED_CATEGORICAL = ["tenure_bucket"]

TENURE_BINS = [-0.1, 6, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-6m", "6-12m", "1-2y", "2-4y", "4y+"]


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """Derive extra columns from the raw customer record.

    Returns a copy — the transformer must never mutate the caller's frame.
    """
    X = X.copy()

    # How many optional services the customer pays for. "No internet service"
    # is not a subscription, so only an explicit "Yes" counts.
    X["num_addons"] = (X[ADDON_COLUMNS] == "Yes").sum(axis=1)

    # Average spend per month of tenure. New customers have tenure 0, so clip
    # to 1 to avoid dividing by zero.
    tenure_safe = X["tenure"].clip(lower=1)
    X["avg_monthly_spend"] = X["TotalCharges"] / tenure_safe

    # Positive when the current monthly bill is above the customer's historical
    # average — a price rise, which is a plausible churn trigger.
    X["charges_delta"] = X["MonthlyCharges"] - X["avg_monthly_spend"]

    # Tenure bucket: churn risk is heavily front-loaded in the first year, so
    # give the model an explicit "how far into the relationship" signal.
    X["tenure_bucket"] = pd.cut(
        X["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS
    ).astype(str)

    return X
