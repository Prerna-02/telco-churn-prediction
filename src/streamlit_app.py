"""Streamlit UI for the telco churn model.

Calls src.predictor directly rather than going over HTTP, so the UI works
whether or not the FastAPI service is running alongside it.
"""

import streamlit as st

from src.predictor import MissingFeatures, model_is_available
from src.predictor import predict as run_prediction
from src.schema import CATEGORICAL_FEATURES, EXAMPLE_CUSTOMER

st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📉",
    layout="wide",
)

CATEGORICAL_OPTIONS = {name: options for name, _, options in CATEGORICAL_FEATURES}


def select(name: str, label: str, help_text: str = "") -> str:
    """Render a selectbox for a categorical feature, defaulting to the example."""
    options = CATEGORICAL_OPTIONS[name]
    default = EXAMPLE_CUSTOMER[name]
    index = options.index(default) if default in options else 0
    return st.selectbox(label, options, index=index, help=help_text)


st.title("📉 Telco Customer Churn Prediction")
st.caption(
    "Random Forest pipeline trained on the IBM Telco Customer Churn dataset. "
    "Fill in a customer profile to estimate their probability of churning."
)

if not model_is_available():
    st.error(
        "The trained model could not be loaded. Check that "
        "`models/churn_model.pkl` exists in the image."
    )
    st.stop()

st.subheader("Customer profile")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Demographics**")
    senior = st.radio("Senior citizen", ["No", "Yes"], horizontal=True)
    partner = select("Partner", "Has partner")
    dependents = select("Dependents", "Has dependents")

with col2:
    st.markdown("**Services**")
    phone_service = select("PhoneService", "Phone service")
    # The model was trained on a dataset where these combinations are
    # structurally impossible, so lock them rather than let the user
    # produce a profile the pipeline never saw.
    if phone_service == "No":
        multiple_lines = "No phone service"
        st.selectbox("Multiple lines", [multiple_lines], disabled=True)
    else:
        multiple_lines = st.selectbox("Multiple lines", ["No", "Yes"])

    internet_service = select("InternetService", "Internet service")
    has_internet = internet_service != "No"

    addons = {}
    addon_fields = [
        ("OnlineSecurity", "Online security"),
        ("OnlineBackup", "Online backup"),
        ("DeviceProtection", "Device protection"),
        ("TechSupport", "Tech support"),
        ("StreamingTV", "Streaming TV"),
        ("StreamingMovies", "Streaming movies"),
    ]
    for name, label in addon_fields:
        if has_internet:
            addons[name] = st.selectbox(label, ["No", "Yes"], key=name)
        else:
            addons[name] = "No internet service"
            st.selectbox(label, [addons[name]], disabled=True, key=name)

with col3:
    st.markdown("**Account**")
    contract = select("Contract", "Contract")
    paperless = select("PaperlessBilling", "Paperless billing")
    payment_method = select("PaymentMethod", "Payment method")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly charges ($)", 18.25, 118.75, 64.75, step=0.05)
    total_charges = st.number_input(
        "Total charges ($)",
        min_value=0.0,
        max_value=8684.80,
        value=float(round(tenure * monthly_charges, 2)),
        step=10.0,
        help="Defaults to tenure × monthly charges; override if you like.",
    )

features = {
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    **addons,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
}

st.divider()

if st.button("Predict churn", type="primary", use_container_width=True):
    try:
        result = run_prediction(features)
    except MissingFeatures as exc:
        st.error(str(exc))
    except (TypeError, ValueError) as exc:
        st.error(f"Invalid feature value: {exc}")
    else:
        probability = result["churn_probability"]
        risk = result["risk_level"]

        left, right = st.columns([1, 2])
        with left:
            st.metric("Churn probability", f"{probability:.1%}")
            st.metric("Will churn?", result["prediction"])
        with right:
            st.markdown("**Risk level**")
            banner = {"High": st.error, "Medium": st.warning, "Low": st.success}[risk]
            banner(f"{risk} risk of churn")
            st.progress(probability)
            st.caption(
                f"Flagged as churn at or above {result['threshold']:.0%}. "
                "That cutoff was tuned to catch ~81% of real churners; about "
                "half of flagged customers would have stayed anyway."
            )
