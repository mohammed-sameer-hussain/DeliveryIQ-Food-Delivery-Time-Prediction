
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import (
    train_test_split,
    KFold,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline

from src.data.loader import (
    ROOT,
    load_config,
    load_dataset,
    clean_dataset,
)
from src.models.train import build_preprocessor


# ============================================================
# 1. PREPARE DATA
# ============================================================

def prepare_training_data():

    config = load_config()
    target = config["target_column"]

    print("\n[1/5] Loading dataset...", flush=True)

    raw = load_dataset()
    data, quality = clean_dataset(raw, target)

    # Remove target leakage.
    forbidden = {
        str(column).strip().lower()
        for column in config.get("leakage_columns", [])
    }

    forbidden.add(target.strip().lower())

    # Remove identifiers.
    identifiers = {
        "id",
        "order_id",
        "delivery_id",
        "row_id",
        "delivery_person_id",
    }

    drop_columns = [
        column for column in data.columns
        if column.strip().lower() in forbidden
        or column.strip().lower() in identifiers
    ]

    X = data.drop(columns=drop_columns)
    y = data[target].astype(float)

    # Remove columns that contain no usable values.
    empty_columns = X.columns[X.isna().all()].tolist()

    if empty_columns:
        X = X.drop(columns=empty_columns)
        drop_columns.extend(empty_columns)

    if X.empty:
        raise ValueError("No usable feature columns remain.")

    print("Dataset rows:", len(data))
    print("Features:", X.shape[1])
    print("Excluded columns:", drop_columns)

    # Use the same split settings as the baseline.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(config.get("test_size", 0.20)),
        random_state=int(config.get("random_state", 42)),
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        config,
        quality,
        drop_columns,
    )


# ============================================================
# 2. DEFINE LIGHTWEIGHT HISTGRADIENT SEARCH
# ============================================================

def build_search(X_train, config):

    estimator = HistGradientBoostingRegressor(
        random_state=42,
    )

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("model", estimator),
    ])

    parameter_space = {
        "model__max_iter": [80, 100, 140, 180],
        "model__learning_rate": [0.03, 0.05, 0.08, 0.10],
        "model__max_leaf_nodes": [15, 20, 31],
        "model__min_samples_leaf": [10, 20, 30],
        "model__l2_regularization": [0.0, 0.5, 1.0, 2.0],
    }

    folds = min(
        int(config.get("tuning_cv_folds", 3)),
        len(X_train),
    )

    if folds < 2:
        raise ValueError(
            "Insufficient training rows for cross-validation."
        )

    cv = KFold(
        n_splits=folds,
        shuffle=True,
        random_state=42,
    )

    # Three candidates initially: a manageable tuning workload.
    n_iter = int(config.get("tuning_n_iter", 3))

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_space,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=cv,
        random_state=42,
        n_jobs=1,
        verbose=1,
        refit=False,
        error_score="raise",
        return_train_score=False,
    )

    return search


# ============================================================
# 3. SAVE RESULTS
# ============================================================

def save_results(
    model,
    metrics,
    search_results,
    predictions,
    X_test,
    y_test,
):

    model_dir = ROOT / "artifacts" / "models"
    report_dir = ROOT / "artifacts" / "reports"

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Keep the original baseline model untouched.
    joblib.dump(
        model,
        model_dir / "tuned_model.joblib",
    )

    with open(
        model_dir / "tuned_model_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
            default=str,
        )

    search_results.to_csv(
        report_dir / "tuning_comparison.csv",
        index=False,
    )

    test_results = X_test.reset_index(drop=True).copy()

    test_results["actual_minutes"] = (
        y_test.reset_index(drop=True).to_numpy()
    )

    test_results["predicted_minutes"] = predictions

    test_results["error_minutes"] = (
        predictions - y_test.reset_index(drop=True).to_numpy()
    )

    test_results["absolute_error_minutes"] = np.abs(
        test_results["error_minutes"]
    )

    test_results.to_csv(
        report_dir / "tuned_test_predictions.csv",
        index=False,
    )

    with open(
        report_dir / "tuned_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
            default=str,
        )


# ============================================================
# 4. HYPERPARAMETER TUNING
# ============================================================

def main():

    print("\n======================================")
    print(" DELIVERYIQ HISTGRADIENT OPTIMIZATION")
    print("======================================")

    (
        X_train,
        X_test,
        y_train,
        y_test,
        config,
        quality,
        drop_columns,
    ) = prepare_training_data()

    print("\n[2/5] Configuring randomized search...")

    search = build_search(X_train, config)

    print("\n[3/5] Running hyperparameter optimization...")
    print("Model: HistGradientBoosting")
    print("Scoring: Negative MAE")
    print("Cross-validation folds:", config.get("tuning_cv_folds", 3))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            search.fit(X_train, y_train)

    except KeyboardInterrupt:
        print(
            "\nTuning was interrupted. "
            "No incomplete model will be saved."
        )
        return

    print("\nBest cross-validation MAE:",
          round(-search.best_score_, 4), "minutes")

    print("\nBest hyperparameters:")
    print(json.dumps(search.best_params_, indent=4, default=str))

    # ========================================================
    # 5. FIT THE BEST MODEL ON THE COMPLETE TRAINING SET
    # ========================================================

    print("\n[4/5] Fitting optimized model...")

    best_model = HistGradientBoostingRegressor(
        random_state=42,
    )

    best_model.set_params(
        **{
            key.replace("model__", ""): value
            for key, value in search.best_params_.items()
        }
    )

    final_pipeline = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("model", best_model),
    ])

    final_pipeline.fit(X_train, y_train)

    # Evaluate against the held-out test data once.
    predictions = final_pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    median_error = median_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    metrics = {
        "project": "DeliveryIQ",
        "stage": "tuned",
        "selected_model": "HistGradientBoosting",
        "test_rows": int(len(y_test)),
        "cv_mae_minutes": float(-search.best_score_),
        "mae_minutes": float(mae),
        "rmse_minutes": float(rmse),
        "median_absolute_error_minutes": float(median_error),
        "r2": float(r2),
        "features": [str(c) for c in X_train.columns],
        "feature_types": {
            str(c): (
                "numeric"
                if c in X_train.select_dtypes(
                    include=["number", "bool"]
                ).columns
                else "categorical"
            )
            for c in X_train.columns
        },
        "dropped_columns": [str(c) for c in drop_columns],
        "data_quality": quality,
        "best_parameters": search.best_params_,
    }

    search_results = pd.DataFrame({
        "model": ["HistGradientBoosting"],
        "best_cv_mae_minutes": [float(-search.best_score_)],
        "best_parameters": [
            json.dumps(search.best_params_, default=str)
        ],
    })

    print("\n[5/5] Saving optimized model and reports...")

    save_results(
        final_pipeline,
        metrics,
        search_results,
        predictions,
        X_test,
        y_test,
    )

    print("\n======================================")
    print(" DELIVERYIQ TUNING COMPLETED")
    print("======================================")

    print(f"CV MAE: {mae if False else -search.best_score_:.3f} minutes")
    print(f"Test MAE: {mae:.3f} minutes")
    print(f"Test RMSE: {rmse:.3f} minutes")
    print(f"Median absolute error: {median_error:.3f} minutes")
    print(f"R2 score: {r2:.4f}")

    print("\nSaved tuned model:")
    print("artifacts/models/tuned_model.joblib")

    print("\nSaved evaluation files:")
    print("artifacts/reports/tuning_comparison.csv")
    print("artifacts/reports/tuned_metrics.json")
    print("artifacts/reports/tuned_test_predictions.csv")


if __name__ == "__main__":
    main()
