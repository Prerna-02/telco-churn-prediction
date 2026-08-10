"""Single source of truth for the model's input features.

Both the FastAPI endpoint and the Streamlit UI build their inputs from here,
so the two frontends can never drift apart. The values mirror the domains
found in data/raw/telco_churn.csv, which is what the pipeline was fitted on.
"""

# Numeric features: (name, label, min, max, default)
NUMERIC_FEATURES = [
    ("SeniorCitizen", "Senior citizen", 0, 1, 0),
    ("tenure", "Tenure (months)", 0, 72, 12),
    ("MonthlyCharges", "Monthly charges ($)", 18.25, 118.75, 64.76),
    ("TotalCharges", "Total charges ($)", 18.80, 8684.80, 2283.30),
]

# Categorical features: (name, label, allowed values)
# NOTE: `gender` is deliberately absent. It is a protected attribute that
# measurably carried no signal (recall 0.492 vs 0.508 between groups), so the
# model is trained without it. It is still audited on in train.py.
CATEGORICAL_FEATURES = [
    ("Partner", "Has partner", ["No", "Yes"]),
    ("Dependents", "Has dependents", ["No", "Yes"]),
    ("PhoneService", "Phone service", ["No", "Yes"]),
    ("MultipleLines", "Multiple lines", ["No", "Yes", "No phone service"]),
    ("InternetService", "Internet service", ["DSL", "Fiber optic", "No"]),
    ("OnlineSecurity", "Online security", ["No", "Yes", "No internet service"]),
    ("OnlineBackup", "Online backup", ["No", "Yes", "No internet service"]),
    ("DeviceProtection", "Device protection", ["No", "Yes", "No internet service"]),
    ("TechSupport", "Tech support", ["No", "Yes", "No internet service"]),
    ("StreamingTV", "Streaming TV", ["No", "Yes", "No internet service"]),
    ("StreamingMovies", "Streaming movies", ["No", "Yes", "No internet service"]),
    ("Contract", "Contract", ["Month-to-month", "One year", "Two year"]),
    ("PaperlessBilling", "Paperless billing", ["No", "Yes"]),
    (
        "PaymentMethod",
        "Payment method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    ),
]

NUMERIC_NAMES = [name for name, *_ in NUMERIC_FEATURES]
CATEGORICAL_NAMES = [name for name, *_ in CATEGORICAL_FEATURES]
ALL_FEATURE_NAMES = NUMERIC_NAMES + CATEGORICAL_NAMES

# A complete, valid request — used as the Streamlit defaults, the OpenAPI
# example, and the payload in the README's curl snippet.
EXAMPLE_CUSTOMER = {
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}
