# Telco Customer Churn Prediction

End-to-end ML project: train a churn classifier on the IBM Telco Customer Churn
dataset, package it with Docker, and deploy the same image two ways — a public
Streamlit demo on Render, and a JSON prediction API on AWS ECS Fargate.

**Live demo:** _add your Render URL here after deploying_

---

## What it does

Given a telecom customer's profile (contract type, tenure, services, billing),
the model returns the probability that the customer will churn, a Yes/No
prediction, and a Low / Medium / High risk bucket.

The Streamlit app has two tabs:

- **Single customer** — a form for one profile, with impossible service
  combinations locked out
- **Batch upload** — drop in a CSV of customers, get every row scored, with a
  risk summary and the results downloadable as CSV. Extra columns
  (`customerID`, `Churn`) pass through untouched; rows with blank or invalid
  values are flagged rather than silently dropped.

| Risk | Probability |
|------|-------------|
| Low | < 0.30 |
| Medium | 0.30 – 0.70 |
| High | ≥ 0.70 |

## Model

A scikit-learn `Pipeline`, saved as one unit so preprocessing always travels
with the model:

1. `FunctionTransformer` — engineered features (`num_addons`,
   `avg_monthly_spend`, `charges_delta`, `tenure_bucket`)
2. `ColumnTransformer` — numeric passed through, categorical one-hot encoded
3. `RandomForestClassifier(class_weight="balanced")`, depth-limited

Trained on 7,032 rows (after dropping the 11 with a blank `TotalCharges`) using
a **60/20/20 train/validation/test split**. Hyperparameters come from a 5-fold
cross-validated grid search; the decision threshold is chosen on validation; the
test set is scored exactly once.

### Results

| Metric | Original | Current |
|---|---|---|
| Recall | 0.4947 | **0.8102** |
| Precision | 0.6401 | 0.4959 |
| F1 | 0.5581 | **0.6152** |
| ROC-AUC | 0.8145 | **0.8386** |
| Accuracy | 0.7918 | 0.7306 |
| Train − test gap | 0.2070 | **0.0399** |

The model now catches **81% of churners instead of 49%**, and the overfitting
gap (previously 99.88% train vs 79.18% test) is essentially closed.

Two costs, stated plainly. Precision fell to 0.50 — half of flagged customers
were not going to leave. And accuracy is now marginally *below* the 0.7342 you
would score by predicting nobody ever churns. Accuracy is the wrong metric on an
imbalanced problem: that do-nothing baseline has a ROC-AUC of 0.5 and identifies
zero churners, while this model scores 0.8386 and finds 303 of 374. It is
penalised for false positives that are, in business terms, cheap.

### Model selection

| Model | CV ROC-AUC | Validation F1 |
|---|---|---|
| RandomForest | 0.8518 ± 0.0091 | 0.6336 |
| HistGradientBoosting | 0.8489 ± 0.0091 | 0.6279 |
| XGBoost | 0.8479 ± 0.0107 | 0.6271 |

The spread is smaller than one standard deviation, so these three are
statistically indistinguishable here — the gains came from the threshold and
the regularisation, not the algorithm. Random Forest ships because it won on
validation F1, which keeps XGBoost out of the serving image.

### Threshold

`predict()` uses **0.46**, chosen on validation to maximise F1 — not the 0.5
that `model.predict()` hardcodes. The same threshold drives both the Yes/No
answer and the risk band, so they cannot contradict each other. It is stored in
`models/model_metadata.json` alongside the metrics and fairness audit.

### Fairness

`gender` is **excluded from the model input** — cross-validated ROC-AUC was
identical to four decimal places with and without it, so a protected attribute
was carrying no signal. It is still audited (disparate impact 0.979, i.e. parity).

`SeniorCitizen`, `Partner` and `Dependents` show disparate impact of 0.41–0.60,
below the conventional 0.8 flag. In each case the flagging gap tracks a real
difference in churn rate, so it reflects the data rather than inventing a
disparity. One finding does deserve attention: recall is 0.679 for partnered
customers versus 0.883 for others. See
[notebooks/02_model_evaluation.ipynb](notebooks/02_model_evaluation.ipynb).

> The pickle is tied to **scikit-learn 1.7.2**, which is why `requirements.txt`
> pins exact versions. Loading it under a different version is not guaranteed
> to work.

## Layout

```
src/
  schema.py         feature names, value domains, example payload
  features.py       engineered features (runs inside the pipeline)
  predictor.py      model loading + prediction (shared by both frontends)
  app.py            FastAPI service  — GET /health, POST /predict
  streamlit_app.py  Streamlit UI
  train.py          training, model comparison, threshold + fairness audit
models/
  churn_model.pkl       fitted pipeline
  model_metadata.json   threshold, risk bands, metrics, fairness audit
data/raw/telco_churn.csv
notebooks/
  01_eda_telco_churn.ipynb
  02_model_evaluation.ipynb   model comparison, thresholds, SHAP, fairness
```

Both frontends call `src/predictor.py`, so the UI and the API can never
disagree about how a customer is scored.

## One image, two modes

`APP_MODE` selects what the container runs, and `PORT` is honoured so the same
image works on Render (which injects its own port) and on ECS.

| `APP_MODE` | Runs | Used by |
|-----------|------|---------|
| `ui` (default) | Streamlit on `$PORT` | Render — the public demo |
| `api` | FastAPI/uvicorn on `$PORT` | ECS Fargate — the prediction service |

## Run locally

```bash
docker compose up --build
```

- Streamlit UI → http://localhost:8501
- API docs → http://localhost:8000/docs

Or without compose:

```bash
docker build -t churn-app .
docker run -p 8501:8501 -e APP_MODE=ui  -e PORT=8501 churn-app
docker run -p 8000:8000 -e APP_MODE=api -e PORT=8000 churn-app
```

## API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 1, "PhoneService": "No",
    "MultipleLines": "No phone service", "InternetService": "DSL",
    "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85, "TotalCharges": 29.85
  }'
```

```json
{"churn_probability": 0.7783, "prediction": "Yes", "risk_level": "High", "threshold": 0.46}
```

All 18 features are required; omitting any returns `400` listing the missing
ones. `gender` is **not** accepted — it was removed from the model. `GET /health`
returns `{"status": "ok"}` and is what the ECS/ALB health check targets.

## Retraining

Training needs xgboost and shap, which are not in the serving image, so use the
dev image:

```bash
docker build -f Dockerfile.dev -t churn-dev .
docker run --rm -v "${PWD}/src:/app/src" -v "${PWD}/data:/app/data" \
  -v "${PWD}/models:/app/models" churn-dev python -m src.train
```

Or locally: `pip install -r requirements-dev.txt && python -m src.train`.

Writes `models/churn_model.pkl` and `models/model_metadata.json`, and logs to
`training.log`. If XGBoost ever wins the comparison it must be moved into
`requirements.txt`, or the serving image will fail to unpickle the model.

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for the full walkthrough — GitHub, ECR, ECS
Fargate, and Render.
