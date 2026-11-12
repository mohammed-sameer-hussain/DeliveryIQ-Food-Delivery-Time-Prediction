
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.loader import ROOT, load_config, load_dataset, clean_dataset


# ============================================================
# 1. PREPROCESSING PIPELINE
# ============================================================

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:

    numeric_features = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_features = [
        col for col in X.columns
        if col not in numeric_features
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            )
        ),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",

        # Dense output is required by HistGradientBoosting.
        sparse_threshold=0.0,
    )

    return preprocessor


# ============================================================
# 2. MACHINE LEARNING MODEL DEFINITIONS
# ============================================================

def get_models():

    models = {

        "DummyMedian": DummyRegressor(
            strategy="median"
        ),

        "Ridge": Ridge(
            alpha=1.0
        ),

        "DecisionTree": DecisionTreeRegressor(
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),

        "RandomForest": RandomForestRegressor(
            n_estimators=80,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=42,
            n_jobs=1
        ),

        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=100,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=42,
            n_jobs=1
        ),

        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.08,
            max_leaf_nodes=20,
            l2_regularization=1.0,
            random_state=42
        ),
    }

    return models


# ============================================================
# 3. DATA PREPARATION
# ============================================================

def prepare_data(df=None):

    config = load_config()

    target = config["target_column"]

    raw = df if df is not None else load_dataset()

    print("\n[1/6] Loading and cleaning dataset...", flush=True)

    data, data_quality = clean_dataset(
        raw,
        target
    )

    print(f"Original rows: {data_quality['rows_before']}")
    print(f"Cleaned rows: {data_quality['rows_after']}")

    # Remove target and known leakage features.
    forbidden = {
        str(column).strip().lower()
        for column in config.get("leakage_columns", [])
    }

    forbidden.add(target.strip().lower())

    # Remove identifier fields that may cause memorization.
    identifier_names = {
        "id",
        "order_id",
        "delivery_id",
        "row_id",
        "delivery_person_id",
    }

    drop_cols = []

    for column in data.columns:

        normalized = column.strip().lower()

        if normalized in forbidden:
            drop_cols.append(column)

        elif normalized in identifier_names:
            drop_cols.append(column)

    X = data.drop(columns=drop_cols)

    y = data[target].astype(float)

    # Remove columns with no usable values.
    empty_cols = X.columns[
        X.isna().all()
    ].tolist()

    if empty_cols:

        X = X.drop(columns=empty_cols)

        drop_cols.extend(empty_cols)

    if X.shape[1] == 0:

        raise ValueError(
            "No usable feature columns remain after cleaning."
        )

    if len(X) < 10:

        raise ValueError(
            "Dataset is too small. At least 10 valid rows are required."
        )

    print(f"Features selected: {X.shape[1]}")
    print(f"Excluded columns: {drop_cols}")

    return X, y, data_quality, drop_cols, config


# ============================================================
# 4. TRAINING AND CROSS-VALIDATION
# ============================================================

def train_model(df=None):

    print("\n========================================")
    print("       DELIVERYIQ MODEL TRAINING")
    print("========================================")

    X, y, data_quality, drop_cols, config = prepare_data(df)

    print("\n[2/6] Splitting training and testing data...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(config.get("test_size", 0.20)),
        random_state=int(config.get("random_state", 42))
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    models = get_models()

    folds = min(
        int(config.get("cv_folds", 3)),
        len(X_train)
    )

    if folds < 2:

        raise ValueError(
            "At least two training folds are required."
        )

    cv = KFold(
        n_splits=folds,
        shuffle=True,
        random_state=42
    )

    # ========================================================
    # 5. MODEL COMPARISON
    # ========================================================

    print("\n[3/6] Starting model comparison...")

    comparison_results = []

    for name, estimator in models.items():

        print(
            f"\nEvaluating {name}...",
            flush=True
        )

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("model", estimator),
        ])

        with warnings.catch_warnings():

            warnings.simplefilter("ignore")

            scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring="neg_mean_absolute_error",
                n_jobs=1,
                error_score="raise"
            )

        cv_mae = float(-scores.mean())

        cv_std = float(scores.std())

        comparison_results.append({

            "model": name,

            "cv_mae_minutes": cv_mae,

            "cv_mae_std": cv_std,

        })

        print(
            f"{name} | "
            f"CV MAE: {cv_mae:.3f} minutes",
            flush=True
        )

    comparison = pd.DataFrame(
        comparison_results
    )

    comparison = comparison.sort_values(
        by="cv_mae_minutes",
        ascending=True
    ).reset_index(drop=True)

    # ========================================================
    # 6. SELECT THE BEST MODEL
    # ========================================================

    print("\n[4/6] Selecting the best model...")

    best_name = str(
        comparison.iloc[0]["model"]
    )

    print(f"\nSelected model: {best_name}")

    best_estimator = models[best_name]

    # Fit only the selected model on all training data.
    best_pipeline = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("model", best_estimator),
    ])

    print("\nTraining selected model...", flush=True)

    best_pipeline.fit(
        X_train,
        y_train
    )

    print("Final model training completed!", flush=True)

    # ========================================================
    # 7. FINAL TEST EVALUATION
    # ========================================================

    print("\n[5/6] Evaluating on held-out test data...")

    predictions = best_pipeline.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    median_error = median_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    metrics = {

        "project": "DeliveryIQ",

        "selected_model": best_name,

        "test_rows": int(len(y_test)),

        "mae_minutes": float(mae),

        "mse": float(mse),

        "rmse_minutes": float(rmse),

        "median_absolute_error_minutes": float(median_error),

        "r2": float(r2),

        "cv_mae_minutes": float(
            comparison.iloc[0]["cv_mae_minutes"]
        ),

        "features": [
            str(column)
            for column in X.columns
        ],

        "dropped_columns": sorted(
            set(map(str, drop_cols))
        ),

        "data_quality": data_quality,
    }

    # ========================================================
    # 8. SAVE MODEL AND REPORTS
    # ========================================================

    print("\n[6/6] Saving model and evaluation reports...")

    model_dir = ROOT / "artifacts" / "models"

    report_dir = ROOT / "artifacts" / "reports"

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save complete preprocessing + model pipeline.
    joblib.dump(
        best_pipeline,
        model_dir / "best_model.joblib"
    )

    # Save metadata.
    with open(
        model_dir / "model_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # Save model comparison.
    comparison.to_csv(
        report_dir / "model_comparison.csv",
        index=False
    )

    # Save evaluation metrics.
    with open(
        report_dir / "metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # Save actual vs predicted values.
    prediction_results = pd.DataFrame({

        "actual_minutes": y_test.to_numpy(),

        "predicted_minutes": predictions,

        "absolute_error_minutes": np.abs(
            y_test.to_numpy() - predictions
        ),

    })

    prediction_results.to_csv(
        report_dir / "test_predictions.csv",
        index=False
    )

    print("\n========================================")
    print("       DELIVERYIQ TRAINING COMPLETE")
    print("========================================")

    return {

        "metrics": metrics,

        "comparison": comparison,

    }


# ============================================================
# 9. MAIN EXECUTION
# ============================================================

def main():

    result = train_model()

    print("\nMODEL COMPARISON")
    print("----------------------------------------")

    print(
        result["comparison"].to_string(
            index=False
        )
    )

    print("\nFINAL TEST PERFORMANCE")
    print("----------------------------------------")

    metrics = result["metrics"]

    print(
        f"Selected model: {metrics['selected_model']}"
    )

    print(
        f"MAE: {metrics['mae_minutes']:.3f} minutes"
    )

    print(
        f"RMSE: {metrics['rmse_minutes']:.3f} minutes"
    )

    print(
        f"Median absolute error: "
        f"{metrics['median_absolute_error_minutes']:.3f} minutes"
    )

    print(
        f"R2 Score: {metrics['r2']:.4f}"
    )

    print("\nModel saved successfully.")

    print(
        "Reports saved inside artifacts/reports/"
    )


if __name__ == "__main__":

    main()
