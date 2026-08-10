"""Train, compare and select a churn model.

Pipeline of work:
  1. 60/20/20 train / validation / test split
  2. 5-fold cross-validation on train for each candidate model
  3. Small hyperparameter search, scored on validation
  4. Threshold chosen on validation (never on test)
  5. Test set scored exactly once, at the end
  6. Fairness audit by group, including attributes not used as features

Run with:  python -m src.train
"""

import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from src.features import (
    ENGINEERED_CATEGORICAL,
    ENGINEERED_NUMERIC,
    add_engineered_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42

# Excluded from the feature matrix: it carries no measurable signal and is a
# protected attribute. Still retained separately for the fairness audit —
# you audit on attributes you refuse to train on.
PROTECTED_EXCLUDED = ["gender"]
AUDIT_ATTRIBUTES = ["gender", "SeniorCitizen", "Partner", "Dependents"]


def get_project_paths():
    base_dir = Path(__file__).resolve().parents[1]
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return (
        base_dir / "data" / "raw" / "telco_churn.csv",
        models_dir / "churn_model.pkl",
        models_dir / "model_metadata.json",
    )


def load_and_prepare_data(data_path: Path):
    """Load, clean, and split off the target."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Blank strings in TotalCharges belong to customers with tenure 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    logger.info(f"Dropped {before - len(df)} rows with blank TotalCharges")

    y = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Churn"])

    logger.info(f"Data shape: X={X.shape}, y={y.shape}")
    logger.info(f"Churn rate: {y.mean():.4f} (majority-class baseline accuracy)")
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Column-wise preprocessing, aware of the engineered columns."""
    engineered = add_engineered_features(X.head(1))

    numeric = [
        c
        for c in engineered.select_dtypes(include=["int64", "float64"]).columns
        if c not in ENGINEERED_CATEGORICAL
    ]
    categorical = [
        c for c in engineered.columns if c not in numeric and c not in PROTECTED_EXCLUDED
    ]

    logger.info(f"Numeric features ({len(numeric)}): {numeric}")
    logger.info(f"Categorical features ({len(categorical)}): {categorical}")
    logger.info(f"Engineered: {ENGINEERED_NUMERIC + ENGINEERED_CATEGORICAL}")
    logger.info(f"Excluded (protected, no signal): {PROTECTED_EXCLUDED}")

    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric),
            # sparse_output=False because HistGradientBoosting rejects sparse input
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
    )


def build_pipeline(X: pd.DataFrame, model) -> Pipeline:
    """Feature engineering -> preprocessing -> classifier, as one saveable unit."""
    return Pipeline(
        steps=[
            ("features", FunctionTransformer(add_engineered_features, validate=False)),
            ("preprocessor", build_preprocessor(X)),
            ("model", model),
        ]
    )


def candidate_models():
    """The three candidates, with a small search grid each."""
    candidates = {
        "RandomForest": (
            RandomForestClassifier(
                random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"
            ),
            {
                "model__n_estimators": [300],
                "model__max_depth": [6, 10, 14],
                "model__min_samples_leaf": [5, 20],
            },
        ),
        "HistGradientBoosting": (
            HistGradientBoostingClassifier(
                random_state=RANDOM_STATE, class_weight="balanced"
            ),
            {
                "model__max_depth": [3, 6],
                "model__learning_rate": [0.05, 0.1],
                "model__max_iter": [200],
            },
        ),
    }

    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.warning("xgboost not installed - skipping it in the comparison")
        return candidates

    candidates["XGBoost"] = (
        XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
        ),
        {
            "model__n_estimators": [300],
            "model__max_depth": [3, 6],
            "model__learning_rate": [0.05, 0.1],
            # XGBoost's equivalent of class_weight="balanced"
            "model__scale_pos_weight": [1.0, 2.77],
        },
    )
    return candidates


def pick_threshold(y_true, probabilities):
    """Choose the decision threshold that maximises F1 on the validation set."""
    grid = np.arange(0.05, 0.96, 0.01)
    scores = [f1_score(y_true, (probabilities >= t).astype(int)) for t in grid]
    best = int(np.argmax(scores))
    return float(grid[best]), float(scores[best])


def evaluate(y_true, probabilities, threshold: float) -> dict:
    y_pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold": round(threshold, 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, probabilities), 4),
        "pr_auc": round(average_precision_score(y_true, probabilities), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def fairness_audit(X, y_true, probabilities, threshold: float) -> dict:
    """Group metrics, including on attributes deliberately kept out of the model."""
    y_pred = (probabilities >= threshold).astype(int)
    audit = {}
    for attr in AUDIT_ATTRIBUTES:
        if attr not in X.columns:
            continue
        groups = {}
        for value in sorted(X[attr].unique(), key=str):
            mask = (X[attr] == value).values
            if mask.sum() < 30:
                continue
            groups[str(value)] = {
                "n": int(mask.sum()),
                "actual_churn_rate": round(float(y_true[mask].mean()), 4),
                "selection_rate": round(float(y_pred[mask].mean()), 4),
                "recall": round(float(recall_score(y_true[mask], y_pred[mask])), 4),
                "precision": round(
                    float(precision_score(y_true[mask], y_pred[mask], zero_division=0)), 4
                ),
            }
        rates = [g["selection_rate"] for g in groups.values()]
        if rates and max(rates) > 0:
            # Disparate impact: ratio of the least- to most-selected group.
            # Below 0.8 is the conventional red flag.
            groups["_disparate_impact"] = round(min(rates) / max(rates), 4)
        audit[attr] = groups
    return audit


def main():
    data_path, model_path, metadata_path = get_project_paths()
    X, y = load_and_prepare_data(data_path)

    # 60/20/20. Test is carved off first and not touched until the very end.
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=RANDOM_STATE, stratify=y_temp
    )
    logger.info(
        f"Split: train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}"
    )

    # Drop protected attributes from the model input entirely, so they are not
    # even accepted as a request field. The full frames are kept for the
    # fairness audit — you audit on what you refuse to train on.
    X_train_m = X_train.drop(columns=PROTECTED_EXCLUDED, errors="ignore")
    X_val_m = X_val.drop(columns=PROTECTED_EXCLUDED, errors="ignore")
    X_test_m = X_test.drop(columns=PROTECTED_EXCLUDED, errors="ignore")
    logger.info(f"Model input columns ({X_train_m.shape[1]}): {list(X_train_m.columns)}")

    baseline_accuracy = float(1 - y_test.mean())
    logger.info(f"Baseline (predict nobody churns) test accuracy: {baseline_accuracy:.4f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    best_name, best_estimator, best_f1 = None, None, -1.0

    for name, (model, grid) in candidate_models().items():
        logger.info(f"--- {name}: cross-validated grid search ---")
        search = GridSearchCV(
            build_pipeline(X_train_m, model),
            param_grid=grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train_m, y_train)

        cv_mean = float(search.best_score_)
        cv_std = float(search.cv_results_["std_test_score"][search.best_index_])
        logger.info(f"{name} CV ROC-AUC: {cv_mean:.4f} +/- {cv_std:.4f}")
        logger.info(f"{name} best params: {search.best_params_}")

        val_proba = search.best_estimator_.predict_proba(X_val_m)[:, 1]
        threshold, val_f1 = pick_threshold(y_val, val_proba)
        val_metrics = evaluate(y_val, val_proba, threshold)
        logger.info(f"{name} validation @ t={threshold:.2f}: {val_metrics}")

        results[name] = {
            "cv_roc_auc_mean": round(cv_mean, 4),
            "cv_roc_auc_std": round(cv_std, 4),
            "best_params": {k: str(v) for k, v in search.best_params_.items()},
            "validation": val_metrics,
        }

        if val_f1 > best_f1:
            best_name, best_estimator, best_f1 = name, search.best_estimator_, val_f1

    logger.info(f"=== Selected model: {best_name} (validation F1 {best_f1:.4f}) ===")

    # Threshold is fixed on validation, then applied unchanged to test.
    val_proba = best_estimator.predict_proba(X_val_m)[:, 1]
    threshold, _ = pick_threshold(y_val, val_proba)

    test_proba = best_estimator.predict_proba(X_test_m)[:, 1]
    test_metrics = evaluate(y_test, test_proba, threshold)
    train_metrics = evaluate(
        y_train, best_estimator.predict_proba(X_train_m)[:, 1], threshold
    )

    logger.info(f"TRAIN metrics: {train_metrics}")
    logger.info(f"TEST  metrics: {test_metrics}")
    logger.info(
        f"Overfitting gap (train-test accuracy): "
        f"{train_metrics['accuracy'] - test_metrics['accuracy']:.4f}"
    )
    logger.info(
        f"Lift over baseline: {test_metrics['accuracy'] - baseline_accuracy:+.4f}"
    )

    audit = fairness_audit(X_test, y_test.values, test_proba, threshold)
    logger.info(f"Fairness audit: {json.dumps(audit, indent=2)}")

    # Anything at or above the threshold is "Yes"; High is the upper half of the
    # remaining range. One threshold drives both, so they cannot contradict.
    high_band = round(threshold + (1.0 - threshold) / 2, 4)

    joblib_dump(best_estimator, model_path)
    metadata = {
        "model_type": best_name,
        "trained_on": date.today().isoformat(),
        "threshold": round(threshold, 4),
        "risk_bands": {"medium_from": round(threshold, 4), "high_from": high_band},
        "excluded_features": PROTECTED_EXCLUDED,
        "baseline_accuracy": round(baseline_accuracy, 4),
        "split": {
            "train": int(X_train.shape[0]),
            "validation": int(X_val.shape[0]),
            "test": int(X_test.shape[0]),
        },
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "model_comparison": results,
        "fairness_audit": audit,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logger.info(f"Saved model to {model_path} and metadata to {metadata_path}")


def joblib_dump(estimator, path: Path):
    import joblib

    joblib.dump(estimator, path, compress=3)


if __name__ == "__main__":
    main()
