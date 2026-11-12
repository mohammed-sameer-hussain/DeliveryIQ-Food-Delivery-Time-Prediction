
from pathlib import Path
import io
import json
import zipfile

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# 1. PROJECT PATHS AND CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "artifacts" / "models"
REPORT_DIR = ROOT / "artifacts" / "reports"
DATA_DIR = ROOT / "data" / "raw"

st.set_page_config(
    page_title="DeliveryIQ | Food Delivery ETA",
    page_icon="🚚",
    layout="wide",
)


# =========================================================
# 2. LOAD THE BEST AVAILABLE MODEL
# =========================================================

def get_model_paths():

    tuned_model = MODEL_DIR / "tuned_model.joblib"
    tuned_metadata = MODEL_DIR / "tuned_model_metadata.json"

    if tuned_model.exists() and tuned_metadata.exists():
        return tuned_model, tuned_metadata, "Optimized"

    baseline_model = MODEL_DIR / "best_model.joblib"
    baseline_metadata = MODEL_DIR / "model_metadata.json"

    if baseline_model.exists() and baseline_metadata.exists():
        return baseline_model, baseline_metadata, "Baseline"

    raise FileNotFoundError(
        "No trained model found. Run python train.py "
        "and python tune.py first."
    )


@st.cache_resource
def load_model(model_path, metadata_path):

    model = joblib.load(model_path)

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return model, metadata


# =========================================================
# 3. LOAD REFERENCE DATA FOR INPUT DEMONSTRATION
# =========================================================

def get_reference_data():

    candidates = [
        DATA_DIR / "Dataset.csv",
        DATA_DIR / "cleaned_dataset (1).csv",
        ROOT / "data" / "processed" / "cleaned_dataset.csv",
        ROOT / "data" / "Dataset.csv",
    ]

    for path in candidates:

        if path.exists():

            frame = pd.read_csv(path)
            frame.columns = frame.columns.str.strip()

            return frame

    return None


def get_default_value(reference, feature, data_type):

    if reference is not None and feature in reference.columns:

        series = reference[feature].dropna()

        if not series.empty:

            if data_type == "numeric":

                numeric = pd.to_numeric(
                    series,
                    errors="coerce"
                ).dropna()

                if not numeric.empty:
                    return float(numeric.median())

            else:

                mode = series.astype(str).mode()

                if not mode.empty:
                    return str(mode.iloc[0])

    return 0.0 if data_type == "numeric" else ""


# =========================================================
# 4. PREPARE REPORT DOWNLOAD
# =========================================================

def create_report_markdown(metadata, stage, audit=None):

    lines = [
        "# DeliveryIQ — Model Evaluation Report",
        "",
        "## Model details",
        "",
        f"- Project: DeliveryIQ",
        f"- Model: {metadata.get('selected_model', 'N/A')}",
        f"- Model stage: {stage}",
        f"- Test records: {metadata.get('test_rows', 'N/A')}",
        f"- Prediction features: {len(metadata.get('features', []))}",
        "",
        "## Evaluation results",
        "",
        f"- Test MAE: {metadata.get('mae_minutes', 'N/A')} minutes",
        f"- Test RMSE: {metadata.get('rmse_minutes', 'N/A')} minutes",
        f"- Median absolute error: "
        f"{metadata.get('median_absolute_error_minutes', 'N/A')} minutes",
        f"- R²: {metadata.get('r2', 'N/A')}",
        f"- Cross-validation MAE: "
        f"{metadata.get('cv_mae_minutes', 'N/A')} minutes",
        "",
        "## Metric interpretation",
        "",
        "- MAE: Average absolute prediction error in minutes.",
        "- RMSE: Gives greater weight to large prediction errors.",
        "- Median absolute error: Typical middle absolute prediction error.",
        "- R²: Measures explained variance. It is not an accuracy percentage.",
        "",
        "## Input features",
        "",
    ]

    for feature in metadata.get("features", []):
        lines.append(f"- {feature}")

    lines.extend([
        "",
        "## Excluded features",
        "",
    ])

    for feature in metadata.get("dropped_columns", []):
        lines.append(f"- {feature}")

    lines.extend([
        "",
        "## Data quality",
        "",
        "```json",
        json.dumps(
            metadata.get("data_quality", {}),
            indent=4
        ),
        "```",
        "",
        "## Validation warning",
        "",
        "Confirm that every feature is available at the time the "
        "delivery ETA is requested. Exclude information generated "
        "after delivery or after the prediction point.",
        "",
        "These metrics represent the saved evaluation split and are "
        "not guarantees for future orders.",
    ])

    if audit is not None:

        lines.extend([
            "",
            "## Leakage audit",
            "",
            "```json",
            json.dumps(audit, indent=4),
            "```",
        ])

    return "\n".join(lines)


def create_report_zip(metadata, metadata_path, stage):

    memory_file = io.BytesIO()

    audit_path = REPORT_DIR / "leakage_audit.json"

    audit = None

    if audit_path.exists():

        audit = json.loads(
            audit_path.read_text(encoding="utf-8")
        )

    markdown_report = create_report_markdown(
        metadata,
        stage,
        audit
    )

    with zipfile.ZipFile(
        memory_file,
        mode="w",
        compression=zipfile.ZIP_DEFLATED
    ) as archive:

        archive.writestr(
            "DeliveryIQ_Model_Evaluation_Report.md",
            markdown_report
        )

        archive.writestr(
            "model_metadata.json",
            metadata_path.read_text(encoding="utf-8")
        )

        # Include all saved reports and charts.
        if REPORT_DIR.exists():

            for path in REPORT_DIR.rglob("*"):

                if path.is_file():

                    relative = path.relative_to(REPORT_DIR)

                    archive.write(
                        path,
                        f"reports/{relative}"
                    )

    return memory_file.getvalue(), markdown_report


# =========================================================
# 5. LOAD APPLICATION ASSETS
# =========================================================

try:

    model_path, metadata_path, model_stage = get_model_paths()

    model, metadata = load_model(
        str(model_path),
        str(metadata_path)
    )

except Exception as exc:

    st.error(str(exc))
    st.stop()


features = metadata["features"]

feature_types = metadata.get("feature_types", {})

reference = get_reference_data()


# =========================================================
# 6. APPLICATION HEADER
# =========================================================

st.title("🚚 DeliveryIQ")

st.subheader(
    "Intelligent Food Delivery ETA Prediction System"
)

st.write(
    "Estimate food delivery time using machine learning, "
    "order information and delivery-related features."
)

st.caption(
    f"Active model: {metadata['selected_model']} | "
    f"Model stage: {model_stage}"
)


# =========================================================
# 7. MODEL PERFORMANCE CARDS
# =========================================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Test MAE",
    f"{metadata['mae_minutes']:.2f} minutes"
)

col2.metric(
    "Test RMSE",
    f"{metadata['rmse_minutes']:.2f} minutes"
)

col3.metric(
    "Test R²",
    f"{metadata['r2']:.4f}"
)

st.caption(
    "These are historical held-out evaluation metrics, "
    "not guaranteed results for an individual delivery."
)

st.divider()


# =========================================================
# 8. APPLICATION TABS
# =========================================================

single_tab, batch_tab, reports_tab = st.tabs([
    "📝 Single Prediction",
    "📁 Batch CSV Prediction",
    "📊 Reports & Downloads",
])


# =========================================================
# 9. SINGLE DELIVERY PREDICTION
# =========================================================

with single_tab:

    st.header("Single Delivery Prediction")

    st.write(
        "Enter the delivery information to calculate "
        "the estimated delivery time."
    )

    # Populate the form with one real example row.
    if reference is not None:

        if st.button(
            "Load example values from my dataset",
            key="load_example"
        ):

            sample = reference.iloc[0]

            for feature in features:

                key = f"single_{feature}"

                value = (
                    sample[feature]
                    if feature in sample.index
                    else get_default_value(
                        reference,
                        feature,
                        feature_types.get(feature, "categorical")
                    )
                )

                if pd.isna(value):

                    value = get_default_value(
                        reference,
                        feature,
                        feature_types.get(feature, "categorical")
                    )

                if feature_types.get(feature) == "numeric":

                    converted = pd.to_numeric(
                        value,
                        errors="coerce"
                    )

                    if pd.isna(converted):
                        value = get_default_value(
                            reference,
                            feature,
                            "numeric"
                        )
                    else:
                        value = float(converted)

                else:

                    value = str(value)

                st.session_state[key] = value

            st.rerun()

    else:

        st.info(
            "The original Dataset.csv was not found. "
            "You can still enter values manually."
        )

    with st.form("single_prediction_form"):

        st.subheader("Enter all required model features")

        user_values = {}

        col_a, col_b = st.columns(2)

        for index, feature in enumerate(features):

            data_type = feature_types.get(
                feature,
                "categorical"
            )

            key = f"single_{feature}"

            default = get_default_value(
                reference,
                feature,
                data_type
            )

            with (col_a if index % 2 == 0 else col_b):

                if data_type == "numeric":

                    if key not in st.session_state:
                        st.session_state[key] = float(default)

                    user_values[feature] = st.number_input(
                        label=feature,
                        key=key,
                        step=0.1,
                        format="%.3f"
                    )

                else:

                    options = []

                    if (
                        reference is not None
                        and feature in reference.columns
                    ):

                        options = (
                            reference[feature]
                            .dropna()
                            .astype(str)
                            .unique()
                            .tolist()
                        )

                    if 0 < len(options) <= 30:

                        options = sorted(options)

                        if key not in st.session_state:

                            value = str(default)

                            st.session_state[key] = (
                                value if value in options
                                else options[0]
                            )

                        user_values[feature] = st.selectbox(
                            label=feature,
                            options=options,
                            key=key
                        )

                    else:

                        if key not in st.session_state:

                            st.session_state[key] = str(default)

                        user_values[feature] = st.text_input(
                            label=feature,
                            key=key
                        )

        submitted = st.form_submit_button(
            "Predict Delivery Time",
            type="primary"
        )

    if submitted:

        empty_fields = [
            feature
            for feature, value in user_values.items()
            if str(value).strip() == ""
        ]

        if empty_fields:

            st.error(
                f"Please complete these fields: {empty_fields}"
            )

        else:

            try:

                input_frame = pd.DataFrame(
                    [user_values],
                    columns=features
                )

                prediction = float(
                    model.predict(input_frame)[0]
                )

                if prediction <= 0:

                    st.warning(
                        "The model returned a non-positive ETA. "
                        "Please review the input values."
                    )

                else:

                    st.success("Prediction completed successfully.")

                    st.metric(
                        "Estimated Delivery Time",
                        f"{prediction:.2f} minutes"
                    )

                    result = input_frame.copy()

                    result["predicted_delivery_time_min"] = prediction

                    st.download_button(
                        "Download Single Prediction CSV",
                        result.to_csv(index=False).encode("utf-8"),
                        file_name="deliveryiq_single_prediction.csv",
                        mime="text/csv"
                    )

            except Exception as exc:

                st.error(f"Prediction failed: {exc}")


# =========================================================
# 10. BATCH CSV PREDICTION
# =========================================================

with batch_tab:

    st.header("Multiple Delivery Predictions")

    st.write(
        "Upload a CSV file containing multiple delivery records. "
        "The system will calculate an ETA for each row."
    )

    # Download a CSV template with the exact feature names.
    template_values = {}

    for feature in features:

        template_values[feature] = get_default_value(
            reference,
            feature,
            feature_types.get(feature, "categorical")
        )

    template = pd.DataFrame([template_values])

    st.subheader("Step A — Download the input template")

    st.download_button(
        "Download CSV Input Template",
        template.to_csv(index=False).encode("utf-8"),
        file_name="deliveryiq_input_template.csv",
        mime="text/csv"
    )

    st.caption(
        "The template contains the exact feature names required "
        "by the currently selected model."
    )

    st.subheader("Step B — Upload your completed CSV")

    uploaded_file = st.file_uploader(
        "Choose your delivery CSV file",
        type=["csv"],
        key="batch_csv"
    )

    if uploaded_file is not None:

        try:

            batch = pd.read_csv(uploaded_file)

            batch.columns = batch.columns.str.strip()

            st.write("Uploaded file preview:")

            st.dataframe(
                batch.head(10),
                use_container_width=True
            )

            missing = [
                feature
                for feature in features
                if feature not in batch.columns
            ]

            if missing:

                st.error(
                    f"Your CSV is missing these required features: {missing}"
                )

                st.info(
                    "Use the downloadable template and keep its "
                    "column names unchanged."
                )

            elif batch.empty:

                st.warning("The uploaded CSV contains no records.")

            else:

                input_frame = batch[features].copy()

                # Validate numeric values and allow empty cells
                # to be handled by the trained imputation pipeline.
                for feature in features:

                    if feature_types.get(feature) == "numeric":

                        original = input_frame[feature]

                        converted = pd.to_numeric(
                            original,
                            errors="coerce"
                        )

                        invalid = (
                            original.notna()
                            & original.astype(str).str.strip().ne("")
                            & converted.isna()
                        )

                        if invalid.any():

                            rows = (
                                invalid[invalid]
                                .index.tolist()[:10]
                            )

                            raise ValueError(
                                f"Invalid numeric values in {feature}. "
                                f"Check CSV row indexes: {rows}"
                            )

                        input_frame[feature] = converted

                    else:

                        input_frame[feature] = (
                            input_frame[feature]
                            .replace(r"^\s*$", np.nan, regex=True)
                        )

                if input_frame.isna().any().any():

                    st.warning(
                        "Some cells are empty. The trained "
                        "preprocessing pipeline will impute missing values."
                    )

                # Run predictions.
                predictions = model.predict(input_frame)

                result = batch.copy()

                result["predicted_delivery_time_min"] = predictions

                st.success(
                    f"Successfully predicted {len(result)} deliveries."
                )

                st.subheader("Step C — View prediction results")

                st.dataframe(
                    result.head(50),
                    use_container_width=True
                )

                average_eta = float(
                    np.mean(predictions)
                )

                st.metric(
                    "Average Estimated Delivery Time",
                    f"{average_eta:.2f} minutes"
                )

                st.subheader("Step D — Download results")

                st.download_button(
                    "Download Complete Batch Predictions CSV",
                    result.to_csv(index=False).encode("utf-8"),
                    file_name="deliveryiq_batch_predictions.csv",
                    mime="text/csv",
                    type="primary"
                )

        except Exception as exc:

            st.error(f"Batch prediction failed: {exc}")


# =========================================================
# 11. REPORTS AND DOWNLOADS
# =========================================================

with reports_tab:

    st.header("DeliveryIQ Model Reports")

    st.write(
        "View evaluation results, model comparisons, "
        "feature importance and downloadable reports."
    )

    # Baseline model comparison.
    comparison_path = REPORT_DIR / "model_comparison.csv"

    if comparison_path.exists():

        st.subheader("Baseline Model Comparison")

        comparison = pd.read_csv(comparison_path)

        st.dataframe(
            comparison,
            use_container_width=True
        )

        st.download_button(
            "Download Baseline Comparison CSV",
            comparison.to_csv(index=False).encode("utf-8"),
            file_name="deliveryiq_baseline_comparison.csv",
            mime="text/csv"
        )

    # Tuning results.
    tuning_path = REPORT_DIR / "tuning_comparison.csv"

    if tuning_path.exists():

        st.subheader("Hyperparameter Tuning Results")

        tuning = pd.read_csv(tuning_path)

        st.dataframe(
            tuning,
            use_container_width=True
        )

        st.download_button(
            "Download Tuning Results CSV",
            tuning.to_csv(index=False).encode("utf-8"),
            file_name="deliveryiq_tuning_results.csv",
            mime="text/csv"
        )

    # Feature importance.
    importance_path = (
        REPORT_DIR / "tuned_permutation_importance.csv"
    )

    if importance_path.exists():

        st.subheader("Feature Importance")

        importance = pd.read_csv(importance_path)

        st.dataframe(
            importance,
            use_container_width=True
        )

        st.download_button(
            "Download Feature Importance CSV",
            importance.to_csv(index=False).encode("utf-8"),
            file_name="deliveryiq_feature_importance.csv",
            mime="text/csv"
        )

    # Evaluation charts.
    st.subheader("Model Evaluation Visualizations")

    plot_dir = REPORT_DIR / "plots"

    plots = [
        (
            "tuned_actual_vs_predicted.png",
            "Actual vs Predicted Delivery Time"
        ),
        (
            "tuned_residuals.png",
            "Residual Analysis"
        ),
        (
            "tuned_error_distribution.png",
            "Prediction Error Distribution"
        ),
        (
            "tuned_permutation_importance.png",
            "Permutation Feature Importance"
        ),
    ]

    for filename, caption in plots:

        path = plot_dir / filename

        if path.exists():

            st.write(f"**{caption}**")

            st.image(
                str(path),
                use_container_width=True
            )

    # Leakage audit.
    audit_path = REPORT_DIR / "leakage_audit.json"

    if audit_path.exists():

        with open(
            audit_path,
            "r",
            encoding="utf-8"
        ) as file:

            audit = json.load(file)

        with st.expander("View Data Leakage Audit"):

            st.json(audit)

        st.download_button(
            "Download Leakage Audit JSON",
            json.dumps(audit, indent=4).encode("utf-8"),
            file_name="deliveryiq_leakage_audit.json",
            mime="application/json"
        )

    # Complete report package.
    st.divider()

    st.subheader("Download Complete Project Evaluation Report")

    report_zip, report_markdown = create_report_zip(
        metadata,
        metadata_path,
        model_stage
    )

    st.download_button(
        "Download Complete Report Bundle (ZIP)",
        report_zip,
        file_name="DeliveryIQ_Complete_Evaluation_Report.zip",
        mime="application/zip",
        type="primary"
    )

    st.download_button(
        "Download Detailed Model Report (Markdown)",
        report_markdown.encode("utf-8"),
        file_name="DeliveryIQ_Model_Evaluation_Report.md",
        mime="text/markdown"
    )

    st.info(
        "The ZIP report includes the evaluation summary, model metadata, "
        "leakage audit, available CSV reports and generated charts."
    )


# =========================================================
# 12. FOOTER
# =========================================================

st.divider()

st.caption(
    "DeliveryIQ | Intelligent Food Delivery ETA Prediction System | "
    "Machine Learning Portfolio Project"
)
