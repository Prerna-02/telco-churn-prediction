"""Streamlit UI for the telco churn model.

Calls src.predictor directly rather than going over HTTP, so the UI works
whether or not the FastAPI service is running alongside it.
"""

import pandas as pd
import streamlit as st

from src.predictor import MissingFeatures, model_is_available, predict_batch
from src.predictor import predict as run_prediction
from src.schema import ALL_FEATURE_NAMES, CATEGORICAL_FEATURES, EXAMPLE_CUSTOMER

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
    "Score one customer at a time, or upload a CSV to score a whole book."
)

if not model_is_available():
    st.error(
        "The trained model could not be loaded. Check that "
        "`models/churn_model.pkl` exists in the image."
    )
    st.stop()

single_tab, batch_tab = st.tabs(["Single customer", "Batch upload (CSV)"])

with single_tab:
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


with batch_tab:
    st.subheader("Score a CSV of customers")
    st.markdown(
        "Upload a CSV with one row per customer. Extra columns such as "
        "`customerID` or `Churn` are kept in the output but ignored by the "
        "model. Rows with blank or unreadable values are flagged rather than "
        "silently dropped."
    )

    template = pd.DataFrame([EXAMPLE_CUSTOMER])[ALL_FEATURE_NAMES]
    template.insert(0, "customerID", "EXAMPLE-0001")
    st.download_button(
        "Download a template CSV",
        template.to_csv(index=False).encode("utf-8"),
        file_name="churn_template.csv",
        mime="text/csv",
        help="One valid row showing every column the model needs.",
    )

    with st.expander("Required columns"):
        st.code(", ".join(ALL_FEATURE_NAMES))
        st.caption(
            "`gender` is not required — it was removed from the model as a "
            "protected attribute that carried no predictive signal."
        )

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read that file as CSV: {exc}")
            st.stop()

        st.info(f"Loaded **{len(raw)}** rows and **{raw.shape[1]}** columns.")

        try:
            scored = predict_batch(raw)
        except MissingFeatures as exc:
            st.error(
                f"{exc}\n\nDownload the template above to see the expected columns."
            )
        except (TypeError, ValueError) as exc:
            st.error(f"Could not score this file: {exc}")
        else:
            ok = scored["scored"].sum()
            failed = len(scored) - ok

            if ok == 0:
                st.error(
                    "No rows could be scored — every row had a blank or invalid "
                    "value in a required column."
                )
                st.stop()

            valid = scored[scored["scored"]]
            flagged = int((valid["prediction"] == "Yes").sum())
            counts = valid["risk_level"].value_counts()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Customers scored", int(ok))
            c2.metric("Flagged as churn", flagged, f"{flagged / ok:.0%} of book")
            c3.metric("High risk", int(counts.get("High", 0)))
            c4.metric("Average probability", f"{valid['churn_probability'].mean():.1%}")

            if failed:
                st.warning(
                    f"{failed} row(s) could not be scored because of blank or "
                    "invalid values. They are kept in the table below with "
                    "`scored = False`."
                )

            st.markdown("**Risk distribution**")
            st.bar_chart(
                counts.reindex(["Low", "Medium", "High"]).fillna(0).astype(int)
            )

            st.markdown("**Results**")
            only_flagged = st.checkbox("Show only customers flagged as churn")
            table = valid[valid["prediction"] == "Yes"] if only_flagged else scored

            st.dataframe(
                table.sort_values("churn_probability", ascending=False, na_position="last"),
                use_container_width=True,
                height=420,
                column_config={
                    "churn_probability": st.column_config.ProgressColumn(
                        "Churn probability", min_value=0.0, max_value=1.0, format="%.2f"
                    )
                },
            )

            st.download_button(
                "Download results as CSV",
                scored.to_csv(index=False).encode("utf-8"),
                file_name="churn_predictions.csv",
                mime="text/csv",
                type="primary",
            )
