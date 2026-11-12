import json
import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.data.loader import ROOT


def main():

    model_dir = ROOT / "artifacts" / "models"
    report_dir = ROOT / "artifacts" / "reports"
    plot_dir = report_dir / "plots"

    plot_dir.mkdir(parents=True, exist_ok=True)

    # Prefer tuned model when one exists.
    if (model_dir / "tuned_model.joblib").exists():
        model_path = model_dir / "tuned_model.joblib"
        metadata_path = model_dir / "tuned_model_metadata.json"
        prediction_path = report_dir / "tuned_test_predictions.csv"
        stage = "tuned"
    else:
        model_path = model_dir / "best_model.joblib"
        metadata_path = model_dir / "model_metadata.json"
        prediction_path = report_dir / "test_predictions.csv"
        stage = "baseline"

    if not model_path.exists() or not prediction_path.exists():
        raise FileNotFoundError(
            "Run python train.py first, or python tune.py for the tuned model."
        )

    model = joblib.load(model_path)

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    data = pd.read_csv(prediction_path)

    actual = data["actual_minutes"].to_numpy(dtype=float)
    predicted = data["predicted_minutes"].to_numpy(dtype=float)

    residuals = predicted - actual

    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    r2 = float(r2_score(actual, predicted))

    # Actual vs predicted
    plt.figure(figsize=(7, 6))
    plt.scatter(actual, predicted, alpha=0.5)

    lower = min(actual.min(), predicted.min())
    upper = max(actual.max(), predicted.max())

    plt.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--"
    )

    plt.xlabel("Actual delivery time (minutes)")
    plt.ylabel("Predicted delivery time (minutes)")
    plt.title("Actual vs Predicted Delivery Time")
    plt.tight_layout()

    plt.savefig(
        plot_dir / f"{stage}_actual_vs_predicted.png",
        dpi=150
    )
    plt.close()

    # Residual analysis
    plt.figure(figsize=(7, 5))
    plt.scatter(predicted, residuals, alpha=0.5)
    plt.axhline(0, linestyle="--")

    plt.xlabel("Predicted delivery time")
    plt.ylabel("Prediction residual")
    plt.title("Residual Analysis")
    plt.tight_layout()

    plt.savefig(
        plot_dir / f"{stage}_residuals.png",
        dpi=150
    )
    plt.close()

    # Error distribution
    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=30)

    plt.xlabel("Prediction error (minutes)")
    plt.ylabel("Number of orders")
    plt.title("Prediction Error Distribution")
    plt.tight_layout()

    plt.savefig(
        plot_dir / f"{stage}_error_distribution.png",
        dpi=150
    )
    plt.close()

    # Permutation importance is available when the prediction
    # report contains the original test feature columns.
    feature_columns = metadata.get("features", [])

    available_features = [
        column for column in feature_columns
        if column in data.columns
    ]

    if available_features:

        X_test = data[available_features].copy()
        y_test = data["actual_minutes"]

        sample_size = min(500, len(X_test))

        if len(X_test) > sample_size:
            sample = X_test.sample(
                sample_size,
                random_state=42
            )
            y_test = y_test.loc[sample.index]
            X_test = sample

        importance = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="neg_mean_absolute_error",
            n_repeats=3,
            random_state=42,
            n_jobs=1
        )

        importance_data = pd.DataFrame({
            "feature": available_features,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std
        }).sort_values(
            "importance_mean",
            ascending=False
        )

        importance_data.to_csv(
            report_dir / f"{stage}_permutation_importance.csv",
            index=False
        )

    summary = {
        "stage": stage,
        "model": metadata.get("selected_model"),
        "test_rows": int(len(actual)),
        "mae_minutes": mae,
        "rmse_minutes": rmse,
        "r2": r2,
        "note": "R2 is not an accuracy percentage."
    }

    (report_dir / f"{stage}_evaluation_summary.json").write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8"
    )

    print("\nDELIVERYIQ EVALUATION")
    print(json.dumps(summary, indent=4))
    print("\nPlots saved in:", plot_dir)


if __name__ == "__main__":
    main()
