"""Inference helpers and validation for the trained housing price model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path("models/best_model.joblib")
META_PATH = Path("models/model_metadata.json")


class ValidationError(ValueError):
    """Raised when inference inputs do not satisfy the saved model schema."""


def load_model(model_path: Path = MODEL_PATH, metadata_path: Path = META_PATH):
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata artifact: {metadata_path}")

    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata


def load_artifacts(model_path: Path = MODEL_PATH, metadata_path: Path = META_PATH):
    return load_model(model_path=model_path, metadata_path=metadata_path)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return bool(pd.isna(value)) if not isinstance(value, (list, dict, tuple, set)) else False


def _clean_numeric_value(column: str, value: Any, numeric_ranges: dict[str, dict[str, float]]) -> float:
    try:
        cleaned = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{column} must be numeric.") from exc

    bounds = numeric_ranges.get(column)
    if bounds:
        min_value = float(bounds["min"])
        max_value = float(bounds["max"])
        if cleaned < min_value or cleaned > max_value:
            raise ValidationError(f"{column} must be between {min_value:g} and {max_value:g}.")

    return cleaned


def _clean_categorical_value(column: str, value: Any, categorical_values: dict[str, list[str]]) -> str:
    cleaned = str(value).strip()
    allowed_values = categorical_values.get(column, [])
    if allowed_values and cleaned not in allowed_values:
        allowed_display = ", ".join(allowed_values)
        raise ValidationError(f"{column} must be one of: {allowed_display}.")
    return cleaned


def validate_input(payload: dict[str, Any], metadata: dict | None = None) -> dict[str, Any]:
    if metadata is None:
        _, metadata = load_model()

    feature_cols = metadata["feature_columns"]
    defaults = metadata.get("defaults", {})
    numeric_features = set(metadata.get("numeric_features", []))
    numeric_ranges = metadata.get("numeric_ranges", {})
    categorical_values = metadata.get("categorical_values", {})

    cleaned_payload: dict[str, Any] = {}

    for column in feature_cols:
        value = payload.get(column, defaults.get(column))
        if _is_missing(value):
            raise ValidationError(f"{column} is required.")

        if column in numeric_features:
            cleaned_payload[column] = _clean_numeric_value(column, value, numeric_ranges)
        else:
            cleaned_payload[column] = _clean_categorical_value(column, value, categorical_values)

    return cleaned_payload


def prepare_input(payload: dict[str, Any], metadata: dict) -> pd.DataFrame:
    cleaned_payload = validate_input(payload, metadata=metadata)
    return pd.DataFrame([cleaned_payload], columns=metadata["feature_columns"])


def predict_one(payload: dict[str, Any], model=None, metadata: dict | None = None) -> float:
    if model is None or metadata is None:
        model, metadata = load_model()

    input_df = prepare_input(payload, metadata)
    pred_log = model.predict(input_df)
    return float(np.expm1(pred_log)[0])


def predict_batch(records: pd.DataFrame | list[dict[str, Any]], model=None, metadata: dict | None = None) -> list[float]:
    if model is None or metadata is None:
        model, metadata = load_model()

    if isinstance(records, pd.DataFrame):
        raw_records = records.to_dict(orient="records")
    else:
        raw_records = records

    cleaned_records = [validate_input(record, metadata=metadata) for record in raw_records]
    input_df = pd.DataFrame(cleaned_records, columns=metadata["feature_columns"])
    pred_log = model.predict(input_df)
    return [float(value) for value in np.expm1(pred_log)]


def predict_price(payload: dict[str, Any]) -> float:
    return predict_one(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Malaysian housing price from JSON payload")
    parser.add_argument(
        "--payload",
        required=True,
        help='JSON string with feature values, e.g. \'{"Bedroom": 3, "Bathroom": 2}\'',
    )
    args = parser.parse_args()

    payload = json.loads(args.payload)
    prediction = predict_one(payload)
    print(json.dumps({"predicted_price_rm": prediction}, indent=2))


if __name__ == "__main__":
    main()
