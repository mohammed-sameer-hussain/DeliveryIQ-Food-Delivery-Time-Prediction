
from pathlib import Path
from functools import lru_cache
import json
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "artifacts" / "models"


app = FastAPI(
    title="DeliveryIQ API",
    description="Food Delivery ETA Prediction API",
    version="2.0.0"
)


class PredictionRequest(BaseModel):

    features: dict[str, object] = Field(
        ...,
        description="Feature name and corresponding input value"
    )


@lru_cache(maxsize=1)
def load_assets():

    tuned_model = MODEL_DIR / "tuned_model.joblib"

    tuned_metadata = MODEL_DIR / "tuned_model_metadata.json"

    if tuned_model.exists() and tuned_metadata.exists():

        model_path = tuned_model

        metadata_path = tuned_metadata

        stage = "Optimized"

    else:

        model_path = MODEL_DIR / "best_model.joblib"

        metadata_path = MODEL_DIR / "model_metadata.json"

        stage = "Baseline"

    if not model_path.exists():

        raise FileNotFoundError(
            "Model not found. Run python train.py and python tune.py."
        )

    model = joblib.load(model_path)

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    return model, metadata, stage


@app.get("/")
def home():

    return {
        "project": "DeliveryIQ",
        "description": "Food Delivery Time Prediction",
        "documentation": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health():

    try:

        model, metadata, stage = load_assets()

        return {
            "status": "healthy",
            "model_ready": True,
            "model": metadata["selected_model"],
            "stage": stage
        }

    except Exception as exc:

        return {
            "status": "not_ready",
            "model_ready": False,
            "error": str(exc)
        }


@app.get("/model-info")
def model_info():

    try:

        model, metadata, stage = load_assets()

        return {
            "project": "DeliveryIQ",
            "model": metadata["selected_model"],
            "stage": stage,
            "features": metadata["features"],
            "test_mae_minutes": metadata["mae_minutes"],
            "test_rmse_minutes": metadata["rmse_minutes"],
            "r2": metadata["r2"]
        }

    except Exception as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc)
        )


@app.post("/predict")
def predict(request: PredictionRequest):

    try:

        model, metadata, stage = load_assets()

    except Exception as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc)
        )

    required_features = metadata["features"]

    missing_features = [
        feature
        for feature in required_features
        if feature not in request.features
    ]

    if missing_features:

        raise HTTPException(
            status_code=422,
            detail={
                "missing_features": missing_features
            }
        )

    try:

        row = {
            feature: request.features[feature]
            for feature in required_features
        }

        input_data = pd.DataFrame(
            [row],
            columns=required_features
        )

        feature_types = metadata.get(
            "feature_types",
            {}
        )

        for feature, data_type in feature_types.items():

            if data_type == "numeric":

                input_data[feature] = pd.to_numeric(
                    input_data[feature],
                    errors="raise"
                )

        prediction = float(
            model.predict(input_data)[0]
        )

        return {
            "project": "DeliveryIQ",
            "predicted_delivery_time_minutes": round(
                prediction,
                2
            ),
            "model": metadata["selected_model"],
            "stage": stage,
            "unit": "minutes"
        }

    except Exception as exc:

        raise HTTPException(
            status_code=422,
            detail=f"Prediction failed: {exc}"
        )
