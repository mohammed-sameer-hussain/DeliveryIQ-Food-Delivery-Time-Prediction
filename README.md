# DeliveryIQ — Intelligent Food Delivery Time Prediction & Analytics

**An end-to-end machine-learning application for food-delivery ETA prediction, model evaluation, batch inference, and API-based predictions.**

![Python](https://img.shields.io/badge/Python-3.14-blue)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-HistGradientBoosting-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Tests](https://img.shields.io/badge/Tests-pytest-brightgreen)

---

## 1. Overview

DeliveryIQ is a food-delivery ETA prediction and analytics project built with Python and Scikit-learn. It uses historical order and delivery attributes to estimate delivery duration in minutes.

The project goes beyond a standalone model-training notebook. It includes:

- A reusable data-loading and cleaning layer.
- Dataset quality and potential leakage auditing.
- Baseline model benchmarking.
- HistGradientBoosting hyperparameter search.
- Held-out test evaluation with regression metrics.
- Serialized preprocessing-and-model pipelines.
- A Streamlit interface for single and batch predictions.
- A FastAPI service for programmatic predictions.
- Downloadable prediction and evaluation outputs.
- Automated tests with pytest.

**Prediction target:** `Time_taken_min`  
**Problem type:** Supervised regression  
**Selected estimator:** `HistGradientBoostingRegressor`

> **Metric note:** R² is a coefficient of determination, not a classification accuracy percentage. MAE and RMSE are reported in minutes.

## 2. Results at a glance

The ZIP supplied for review contains saved baseline and tuned model metadata for the same configured 80/20 random train-test split (`random_state=42`). Each saved test evaluation contains 10,000 records.

| Metric | Baseline | Tuned | Change |
|---|---:|---:|---:|
| Test MAE | 2.7832 min | 2.6306 min | 5.49% lower |
| Test RMSE | 3.6817 min | 3.4268 min | 6.92% lower |
| Test R² | 0.98933 | 0.99076 | +0.143 percentage points |
| Cross-validation MAE | 2.8375 min | 2.6971 min | 4.95% lower |

The percentage reductions are calculated relative to the baseline metric:

- **MAE reduction:** `(2.783235 - 2.630647) / 2.783235 × 100 ≈ 5.49%`
- **RMSE reduction:** `(3.681678 - 3.426796) / 3.681678 × 100 ≈ 6.92%`
- **Cross-validation MAE reduction:** `(2.837481 - 2.697063) / 2.837481 × 100 ≈ 4.95%`

The tuned model's reported test MAE is approximately **2.63 minutes**. This is an average absolute error on the saved test set; it is not a guarantee for every delivery or for future operating conditions.

### Baseline model benchmark

The saved `model_comparison.csv` records the following cross-validation MAE values:

| Model | Cross-validation MAE |
|---|---:|
| HistGradientBoosting | 2.8375 min |
| ExtraTrees | 3.0605 min |
| RandomForest | 3.2426 min |
| DecisionTree | 4.5665 min |
| Ridge | 6.4144 min |
| DummyMedian | 27.0496 min |

Lower MAE is better. HistGradientBoosting was selected as the best baseline candidate in this comparison.

### Tuned model configuration

The saved tuned metadata records these selected parameters:

| Parameter | Value |
|---|---:|
| `max_iter` | 180 |
| `learning_rate` | 0.08 |
| `max_leaf_nodes` | 20 |
| `min_samples_leaf` | 10 |
| `l2_regularization` | 0.0 |

The tuning script uses `RandomizedSearchCV`, negative MAE scoring, and 3-fold shuffled K-fold cross-validation. The current configuration evaluates **3 randomized parameter candidates**. This is a compact search, not an exhaustive optimization.

## 3. Dataset

The saved audit and model metadata report:

| Dataset property | Value |
|---|---:|
| Rows | 50,000 |
| Columns | 24 |
| Model input features | 21 |
| Target | `Time_taken_min` |
| Held-out test rows | 10,000 |

The 21 model features recorded in the tuned metadata are:

`Order_Date`, `Order_Hour`, `Day_of_Week`, `Is_Weekend`, `Is_Festival`, `Weather`, `Pickup_Zone`, `Dropoff_Zone`, `Vehicle_Type`, `Rider_Experience_Years`, `Rider_Rating`, `Restaurant_Rating`, `Cuisine_Type`, `Order_Items`, `Restaurant_Load`, `Preparation_Time_Min`, `Road_Distance_km`, `Delivery_Distance_Category`, `Traffic_Level`, `Number_of_Signals`, and `Delivery_Priority`.

### Data quality and leakage checks

The saved audit reports:

- 50,000 rows before cleaning.
- 50,000 rows after cleaning.
- No duplicate or invalid-target rows removed by the cleaning step.
- No missing values reported by the saved cleaning summary.
- No high-correlation review flags at the configured threshold.
- `Average_Speed_kmph` identified as a suspicious/known leakage column and excluded from model inputs.
- `Order_ID` excluded as an identifier.
- `Time_taken_min` excluded from features because it is the target.

The exclusion of `Average_Speed_kmph` is important because average speed may be calculated using distance and actual delivery duration. Every feature should also be checked to confirm it is available at the exact time an ETA is requested.

A clean-data audit does **not** prove that all forms of leakage or bias are absent. The current evaluation uses a random split; chronological and location-based validation are recommended before making production claims.

## 4. Features

### Data preparation and audit

- Loads the configured CSV dataset.
- Checks that the target column exists.
- Removes exact duplicate rows.
- Converts the target to numeric values.
- Removes rows with missing or non-positive target values.
- Reports missing-value counts and cleaning statistics.
- Excludes configured leakage columns and identifier columns from model features.

### Model training and benchmarking

The training pipeline compares:

- Dummy median baseline.
- Ridge regression.
- Decision Tree.
- Random Forest.
- Extra Trees.
- HistGradientBoosting.

The best baseline candidate is selected using cross-validation MAE on the training portion. The selected pipeline is then evaluated on the held-out test portion.

### Hyperparameter optimization

The tuning workflow applies randomized search to HistGradientBoosting using negative MAE as its scoring metric. The best configuration is refitted on the training data and evaluated against the same configured held-out test split.

### Streamlit application

The Streamlit application (`app/streamlit_app.py`) provides three tabs:

1. **Single Prediction** — enter delivery features and estimate delivery duration.
2. **Batch CSV Prediction** — upload a CSV containing the model's required feature columns, generate predictions, and download results.
3. **Reports & Downloads** — review available model information and download a report bundle containing a Markdown evaluation summary, model metadata, and saved report files.

The app prefers the tuned model when its model and metadata files exist; otherwise, it falls back to the baseline model.

### FastAPI service

The FastAPI application (`api/main.py`) exposes:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | Model readiness check |
| GET | `/model-info` | Active model, feature list, and saved metrics |
| POST | `/predict` | Predict delivery time from feature values |

Interactive API documentation is available at `/docs` after starting the API.

### Evaluation outputs

The evaluation script generates:

- Actual-versus-predicted plot.
- Residual plot.
- Prediction-error distribution.
- Permutation-importance CSV when the required test features are available.
- Evaluation summary JSON.

## 5. Technology stack

| Area | Technology |
|---|---|
| Language | Python |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Model persistence | Joblib |
| Visualization | Matplotlib |
| Interactive application | Streamlit |
| REST API | FastAPI |
| API server | Uvicorn |
| Validation | Pytest |

## 6. Project structure

```text
DeliveryIQ/
├── api/
│   └── main.py
├── app/
│   └── streamlit_app.py
├── artifacts/
│   ├── models/                 # generated model files (not committed by default)
│   └── reports/                # generated metrics, predictions and plots
├── configs/
│   └── config.json
├── data/
│   ├── raw/                    # place Dataset.csv here
│   └── processed/
├── notebooks/
│   └── README.md
├── src/
│   ├── data/
│   │   ├── audit.py
│   │   └── loader.py
│   └── models/
│       ├── train.py
│       ├── tune.py
│       └── evaluate.py
├── tests/
│   ├── test_data.py
│   └── test_pipeline.py
├── audit_data.py
├── train.py
├── tune.py
├── evaluate.py
├── requirements.txt
├── requirements-optional-boosting.txt
├── requirements-optional-explainability.txt
├── requirements-optional-xgboost.txt
├── pytest.ini
└── README.md
```

Generated model binaries, raw datasets, and many generated report files are excluded by `.gitignore` in the supplied project. Run the pipeline locally to recreate them.

## 7. Installation — Windows PowerShell

### Prerequisites

- Git.
- Python compatible with the dependency versions in `requirements.txt`.
- VS Code or another Python IDE.

The project README supplied in the ZIP documents Python 3.14. If that interpreter is installed, use the commands below. Otherwise, select an installed Python version compatible with the listed packages.

### 1. Clone the repository

```powershell
git clone YOUR_GITHUB_REPOSITORY_URL
cd DeliveryIQ
```

### 2. Create and activate a virtual environment

```powershell
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4. Add the dataset

Place the dataset at:

```text
data/raw/Dataset.csv
```

The target column must be named `Time_taken_min`, unless you change `target_column` in `configs/config.json`.

The dataset is not included in a clean GitHub clone when excluded by `.gitignore`. Obtain it from a source for which you have permission to use and redistribute it. Do not commit private or restricted data.

## 8. Run the project

Run commands from the repository root (the directory containing `train.py`).

### Step 1 — Audit the dataset

```powershell
python audit_data.py
```

Creates:

```text
artifacts/reports/leakage_audit.json
```

### Step 2 — Train and benchmark models

```powershell
python train.py
```

Creates the baseline model and its metadata, model-comparison report, metrics, and test predictions under `artifacts/`.

### Step 3 — Tune HistGradientBoosting

```powershell
python tune.py
```

Creates the tuned model, tuned metadata, tuning comparison, tuned metrics, and tuned test predictions.

### Step 4 — Generate evaluation plots and feature importance

```powershell
python evaluate.py
```

Creates evaluation summaries and plots under:

```text
artifacts/reports/
artifacts/reports/plots/
```

### Step 5 — Run automated tests

```powershell
pytest -q
```

The supplied ZIP contains two test functions. The previously reported run showed `2 passed`; rerun the tests in your environment to confirm.

### Step 6 — Launch the Streamlit application

```powershell
streamlit run app/streamlit_app.py
```

Streamlit prints the local application URL in the terminal.

### Step 7 — Launch the FastAPI service

Open a second terminal, activate the same virtual environment, and run:

```powershell
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

- API root: `http://127.0.0.1:8000/`
- Health: `http://127.0.0.1:8000/health`
- Model information: `http://127.0.0.1:8000/model-info`
- Interactive API docs: `http://127.0.0.1:8000/docs`

### Example API request

Use the feature names and valid example values from `artifacts/models/tuned_model_metadata.json`. The API expects all required model features inside the `features` object.

```json
{
  "features": {
    "Order_Date": "2025-01-15",
    "Order_Hour": 13,
    "Day_of_Week": "Wednesday",
    "Is_Weekend": 0,
    "Is_Festival": 0,
    "Weather": "Clear",
    "Pickup_Zone": "Zone_1",
    "Dropoff_Zone": "Zone_2",
    "Vehicle_Type": "Bike",
    "Rider_Experience_Years": 3,
    "Rider_Rating": 4.7,
    "Restaurant_Rating": 4.5,
    "Cuisine_Type": "Indian",
    "Order_Items": 2,
    "Restaurant_Load": "Medium",
    "Preparation_Time_Min": 10,
    "Road_Distance_km": 5.2,
    "Delivery_Distance_Category": "Medium",
    "Traffic_Level": "Medium",
    "Number_of_Signals": 4,
    "Delivery_Priority": "Normal"
  }
}
```

This is an illustrative request shape. Replace example categories with values supported by the dataset and model preprocessing.

## 9. Configuration

The main configuration is stored in `configs/config.json`.

Current saved settings include:

- Target column: `Time_taken_min`
- Test size: `0.20`
- Random state: `42`
- Baseline cross-validation folds: `3`
- Tuning iterations: `3`
- Tuning cross-validation folds: `3`
- Explicit leakage-column exclusions, including `Average_Speed_kmph`

Change these settings carefully and rerun the complete training and evaluation pipeline after any change.

## 10. What makes DeliveryIQ different?

Many introductory ML demonstrations stop after fitting a model and printing one score. DeliveryIQ brings several stages of an ML workflow together:

| Typical introductory project | DeliveryIQ implementation |
|---|---|
| One model | Multiple baseline regressors benchmarked |
| Manual data checks | Reusable cleaning and audit scripts |
| Features passed directly to estimator | Preprocessing and estimator saved as a pipeline |
| One metric | MAE, RMSE, median absolute error and R² |
| No tuning comparison | Randomized hyperparameter search |
| Notebook-only predictions | Streamlit single and batch prediction interface |
| No service endpoint | FastAPI prediction API |
| Manual result inspection | Saved metrics, predictions, plots and report bundle |
| No automated checks | Pytest tests for cleaning and preprocessing |

The key differentiator is the integration of **data validation, model benchmarking, optimization, application-based inference, API access, and evaluation reporting** in one project.

## 11. Limitations and responsible interpretation

The current results are promising, but the following limitations should be addressed before describing DeliveryIQ as production-ready:

1. **Random split:** The current evaluation uses a shuffled random split. A chronological split is more representative when predicting future orders.
2. **Geographic generalization:** Test the model on unseen pickup/drop-off zones to measure location generalization.
3. **Feature availability:** Confirm every input is known when the ETA is requested. In particular, preparation time or rider information may not be available at order placement.
4. **Tuning budget:** The current randomized search evaluates only three parameter candidates. Expand the search using training-only cross-validation.
5. **Test-set discipline:** Keep the held-out test set untouched during model selection and tuning.
6. **Data provenance:** Document the dataset source, license, collection method, and any synthetic-data assumptions.
7. **Test coverage:** The included test suite is small. Add tests for CSV schema validation, unknown categories, missing features, API responses, model loading, and batch predictions.
8. **Reproducibility:** Record package versions, data version, split strategy, and model parameters with each experiment.

A high R² on one random split does not guarantee the same performance on future deliveries.

## 12. Future enhancements

The following are proposed roadmap items, not features claimed as implemented in the current ZIP.

### Explainable AI
- Add SHAP explanations for global and individual predictions.
- Show which input features increase or decrease an ETA.
- Provide understandable explanations in the dashboard.

### Delay-risk classification
- Add a separate classification model for on-time, moderately delayed, and highly delayed orders.
- Evaluate it with precision, recall, F1-score, and a confusion matrix.

### Data and model drift monitoring
- Compare production input distributions with training data.
- Monitor prediction error when actual outcomes become available.
- Alert when drift or performance degradation is detected.

### Prediction intervals
- Add conformal prediction or quantile-based methods.
- Return a plausible ETA range alongside the point estimate.

### Stronger model experimentation
- Benchmark XGBoost, LightGBM, and CatBoost where dependencies and licensing permit.
- Use more extensive cross-validation and documented experiment tracking.
- Compare models using the same validation strategy.

### Production deployment
- Containerize the API using Docker.
- Add database-backed prediction history.
- Add authentication, request validation, structured logging, and monitoring.
- Deploy the service to a cloud platform.

### Real-world operational data
- Integrate permitted traffic, weather, and route information.
- Validate data freshness and API reliability.
- Reassess model performance using real delivery outcomes.

## 13. Contributing

Contributions and suggestions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make a focused change.
4. Run the tests with `pytest -q`.
5. Submit a pull request describing the change and its validation.

## 14. Author and project information

**Project:** DeliveryIQ — Intelligent Food Delivery Time Prediction & Analytics  
**Domain:** Machine Learning, Predictive Analytics, Logistics  
**Core stack:** Python, Pandas, NumPy, Scikit-learn, Streamlit, FastAPI, Pytest

## 15. Conclusion

DeliveryIQ demonstrates an end-to-end machine-learning workflow for food-delivery time estimation. The saved experiment results show that the tuned HistGradientBoosting model reduced test MAE by approximately **5.49%** and test RMSE by approximately **6.92%** compared with the saved baseline evaluation, while increasing R² by approximately **0.143 percentage points**.

Its value is not limited to the model score: the project also includes data auditing, model comparison, a Streamlit application, batch inference, a FastAPI service, and evaluation reporting.

**DeliveryIQ — Predict. Evaluate. Improve.**
