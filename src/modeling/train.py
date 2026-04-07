"""Train housing price models, save the best artifact, and publish evaluation outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional external gradient boosting libraries.
try:
    from xgboost import XGBRegressor  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    CatBoostRegressor = None

RANDOM_STATE = 42
TARGET_COL = "price"
MODEL_VERSION = "v1"

# High-cardinality text/id-like columns that make v1 harder to generalize.
EXCLUDE_COLS = {
    TARGET_COL,
    "Building Name",
    "Address",
    "Facilities",
    "Developer",
    "Railway Station",
    "Bus Stop",
    "School",
    "Nearby Mall",
}

DATA_PATH = Path("data/final/house_model_ready.csv")
MODEL_PATH = Path("models/best_model.joblib")
META_PATH = Path("models/model_metadata.json")
REPORTS_DIR = Path("reports/model")
COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
TEST_METRICS_PATH = REPORTS_DIR / "test_metrics.json"
SPLIT_INDICES_PATH = REPORTS_DIR / "split_indices.csv"
TEST_PREDICTIONS_PATH = REPORTS_DIR / "test_predictions.csv"
GROUPED_ERRORS_PATH = REPORTS_DIR / "grouped_error_analysis.csv"
RESIDUAL_PLOT_PATH = REPORTS_DIR / "residuals_vs_actual.png"
PREDICTION_PLOT_PATH = REPORTS_DIR / "predictions_vs_actual.png"


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "median_abs_error": float(np.median(np.abs(y_true - y_pred))),
    }


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=10,
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_cols, categorical_cols


def build_candidates(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    candidates: dict[str, Pipeline] = {
        "dummy_median": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", DummyRegressor(strategy="median")),
            ]
        ),
        "linear_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LinearRegression()),
            ]
        ),
        "ridge": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "lasso": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", Lasso(alpha=0.001, max_iter=10000, random_state=RANDOM_STATE)),
            ]
        ),
        "elastic_net": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=10000, random_state=RANDOM_STATE)),
            ]
        ),
        "decision_tree": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", DecisionTreeRegressor(max_depth=18, min_samples_leaf=3, random_state=RANDOM_STATE)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=600,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "bagging": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    BaggingRegressor(
                        estimator=DecisionTreeRegressor(max_depth=18, random_state=RANDOM_STATE),
                        n_estimators=250,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", GradientBoostingRegressor(random_state=RANDOM_STATE)),
            ]
        ),
        "adaboost": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    AdaBoostRegressor(
                        estimator=DecisionTreeRegressor(max_depth=4, random_state=RANDOM_STATE),
                        n_estimators=300,
                        learning_rate=0.03,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        random_state=RANDOM_STATE,
                        learning_rate=0.05,
                        max_depth=8,
                        max_iter=500,
                    ),
                ),
            ]
        ),
        "knn": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", KNeighborsRegressor(n_neighbors=15, weights="distance")),
            ]
        ),
        "svr_rbf": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", SVR(kernel="rbf", C=50.0, epsilon=0.05, gamma="scale")),
            ]
        ),
    }

    if XGBRegressor is not None:
        candidates["xgboost"] = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=1000,
                        learning_rate=0.03,
                        max_depth=6,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_alpha=0.0,
                        reg_lambda=1.0,
                        objective="reg:squarederror",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        )

    if LGBMRegressor is not None:
        candidates["lightgbm"] = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=1200,
                        learning_rate=0.03,
                        num_leaves=31,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        )

    if CatBoostRegressor is not None:
        candidates["catboost"] = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    CatBoostRegressor(
                        iterations=1200,
                        learning_rate=0.03,
                        depth=6,
                        loss_function="RMSE",
                        random_seed=RANDOM_STATE,
                        verbose=False,
                    ),
                ),
            ]
        )

    return candidates


def build_schema_defaults(X: pd.DataFrame) -> tuple[dict, dict, dict]:
    defaults: dict[str, object] = {}
    numeric_ranges: dict[str, dict[str, float]] = {}
    categorical_values: dict[str, list[str]] = {}

    for col in X.columns:
        series = X[col]
        if pd.api.types.is_numeric_dtype(series):
            defaults[col] = float(series.median()) if series.notna().any() else 0.0
            numeric_ranges[col] = {
                "min": float(series.min()) if series.notna().any() else 0.0,
                "max": float(series.max()) if series.notna().any() else 1.0,
            }
        else:
            mode = series.mode(dropna=True)
            defaults[col] = str(mode.iloc[0]) if not mode.empty else "Unknown"
            vals = series.dropna().astype(str).value_counts().head(25).index.tolist()
            categorical_values[col] = vals

    return defaults, numeric_ranges, categorical_values


def find_constant_columns(X: pd.DataFrame) -> list[str]:
    """Return columns with <=1 unique non-null value (no predictive signal)."""
    constant_cols: list[str] = []
    for col in X.columns:
        if X[col].nunique(dropna=True) <= 1:
            constant_cols.append(col)
    return constant_cols


def save_split_indices(train_idx: np.ndarray, valid_idx: np.ndarray, test_idx: np.ndarray) -> None:
    split_df = pd.concat(
        [
            pd.DataFrame({"source_row_index": train_idx, "split": "train"}),
            pd.DataFrame({"source_row_index": valid_idx, "split": "valid"}),
            pd.DataFrame({"source_row_index": test_idx, "split": "test"}),
        ],
        ignore_index=True,
    ).sort_values(["split", "source_row_index"])
    split_df.to_csv(SPLIT_INDICES_PATH, index=False)


def build_price_bands(y_true: np.ndarray) -> pd.Series:
    labels = ["Budget", "Mid-range", "Premium", "Luxury"]
    unique_values = np.unique(y_true)
    if len(unique_values) < 4:
        return pd.Series(np.repeat("Overall", len(y_true)))

    try:
        return pd.qcut(y_true, q=4, labels=labels, duplicates="drop").astype(str)
    except ValueError:
        return pd.Series(np.repeat("Overall", len(y_true)))


def build_test_predictions(X_test: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    test_predictions = X_test.reset_index(drop=True).copy()
    test_predictions["actual_price"] = y_true
    test_predictions["predicted_price"] = y_pred
    test_predictions["residual"] = y_true - y_pred
    test_predictions["absolute_error"] = np.abs(y_true - y_pred)
    test_predictions["absolute_percentage_error"] = np.where(
        y_true != 0,
        np.abs((y_true - y_pred) / y_true) * 100,
        np.nan,
    )
    test_predictions["price_band"] = build_price_bands(y_true)
    return test_predictions


def summarize_group_errors(test_predictions: pd.DataFrame) -> pd.DataFrame:
    grouped_frames: list[pd.DataFrame] = []
    group_specs = [
        ("Property Type", "Property Type"),
        ("Tenure Type", "Tenure Type"),
        ("price_band", "Price Band"),
    ]

    for column, label in group_specs:
        if column not in test_predictions.columns:
            continue

        grouped = (
            test_predictions.groupby(column, dropna=False)
            .agg(
                rows=("actual_price", "size"),
                actual_price_mean=("actual_price", "mean"),
                predicted_price_mean=("predicted_price", "mean"),
                mae=("absolute_error", "mean"),
                median_abs_error=("absolute_error", "median"),
                rmse=("residual", lambda values: float(np.sqrt(np.mean(np.square(values))))),
                mean_abs_percentage_error=("absolute_percentage_error", "mean"),
            )
            .reset_index()
            .rename(columns={column: "group_value"})
        )
        grouped.insert(0, "group_type", label)
        grouped_frames.append(grouped)

    if not grouped_frames:
        return pd.DataFrame(columns=["group_type", "group_value"])

    return pd.concat(grouped_frames, ignore_index=True)


def save_error_plots(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    residuals = y_true - y_pred

    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, residuals, alpha=0.55, color="#1f77b4", edgecolors="none")
    plt.axhline(0.0, color="#d62728", linestyle="--", linewidth=1.5)
    plt.title("Residuals vs Actual Price")
    plt.xlabel("Actual Price (RM)")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.tight_layout()
    plt.savefig(RESIDUAL_PLOT_PATH, dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.55, color="#2ca02c", edgecolors="none")
    min_value = float(min(y_true.min(), y_pred.min()))
    max_value = float(max(y_true.max(), y_pred.max()))
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="#d62728", linewidth=1.5)
    plt.title("Predicted vs Actual Price")
    plt.xlabel("Actual Price (RM)")
    plt.ylabel("Predicted Price (RM)")
    plt.tight_layout()
    plt.savefig(PREDICTION_PLOT_PATH, dpi=180)
    plt.close()


def save_error_artifacts(X_test: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    test_predictions = build_test_predictions(X_test, y_true, y_pred)
    grouped_errors = summarize_group_errors(test_predictions)

    test_predictions.to_csv(TEST_PREDICTIONS_PATH, index=False)
    grouped_errors.to_csv(GROUPED_ERRORS_PATH, index=False)
    save_error_plots(y_true, y_pred)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    raw_df = pd.read_csv(DATA_PATH)
    if TARGET_COL not in raw_df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {DATA_PATH}")

    filtered_df = raw_df.loc[raw_df[TARGET_COL] > 0].copy()
    source_row_index = filtered_df.index.to_numpy()
    df = filtered_df.reset_index(drop=True)

    initial_feature_cols = [col for col in df.columns if col not in EXCLUDE_COLS]
    X = df[initial_feature_cols].copy()
    dropped_constant_cols = find_constant_columns(X)
    if dropped_constant_cols:
        X = X.drop(columns=dropped_constant_cols)

    feature_cols = X.columns.tolist()
    if not feature_cols:
        raise ValueError("No usable features remain after dropping constant columns.")

    y = df[TARGET_COL].astype(float).to_numpy()
    y_log = np.log1p(y)

    (
        X_trainval,
        X_test,
        y_trainval_log,
        _y_test_log,
        y_trainval_raw,
        y_test_raw,
        trainval_idx,
        test_idx,
    ) = train_test_split(
        X,
        y_log,
        y,
        source_row_index,
        test_size=0.15,
        random_state=RANDOM_STATE,
    )

    (
        X_train,
        X_valid,
        y_train_log,
        y_valid_log,
        y_train_raw,
        y_valid_raw,
        train_idx,
        valid_idx,
    ) = train_test_split(
        X_trainval,
        y_trainval_log,
        y_trainval_raw,
        trainval_idx,
        test_size=0.1764705882,
        random_state=RANDOM_STATE,
    )

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)
    candidates = build_candidates(preprocessor)
    print(f"Candidate models: {', '.join(candidates.keys())}")

    comparison_rows: list[dict[str, float | str]] = []
    best_name = ""
    best_valid_rmse = float("inf")
    best_pipeline: Pipeline | None = None

    for name, pipeline in candidates.items():
        fit_started = perf_counter()
        pipeline.fit(X_train, y_train_log)
        fit_time_seconds = perf_counter() - fit_started

        valid_pred_raw = np.expm1(pipeline.predict(X_valid))
        valid_metrics = evaluate_predictions(y_valid_raw, valid_pred_raw)
        model_params = pipeline.named_steps["model"].get_params()

        comparison_rows.append(
            {
                "model": name,
                "fit_time_seconds": round(fit_time_seconds, 3),
                "valid_rmse": valid_metrics["rmse"],
                "valid_mae": valid_metrics["mae"],
                "valid_r2": valid_metrics["r2"],
                "valid_median_abs_error": valid_metrics["median_abs_error"],
                "params": json.dumps(model_params, sort_keys=True, default=str),
            }
        )

        if valid_metrics["rmse"] < best_valid_rmse:
            best_valid_rmse = valid_metrics["rmse"]
            best_name = name
            best_pipeline = pipeline

    assert best_pipeline is not None

    best_pipeline.fit(X_trainval, y_trainval_log)
    test_pred_raw = np.expm1(best_pipeline.predict(X_test))
    test_metrics = evaluate_predictions(y_test_raw, test_pred_raw)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pipeline, MODEL_PATH)

    comparison_df = pd.DataFrame(comparison_rows).sort_values("valid_rmse", ascending=True)
    comparison_df.to_csv(COMPARISON_PATH, index=False)

    save_split_indices(train_idx, valid_idx, test_idx)
    save_error_artifacts(X_test, y_test_raw, test_pred_raw)

    defaults, numeric_ranges, categorical_values = build_schema_defaults(X_trainval)
    generated_at = datetime.now(timezone.utc).isoformat()

    metadata = {
        "version": MODEL_VERSION,
        "trained_at_utc": generated_at,
        "target": TARGET_COL,
        "transform": "log1p",
        "best_model": best_name,
        "random_state": RANDOM_STATE,
        "train_rows": int(len(X_train)),
        "valid_rows": int(len(X_valid)),
        "test_rows": int(len(X_test)),
        "feature_columns": feature_cols,
        "dropped_constant_columns": dropped_constant_cols,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "excluded_columns": sorted(list(EXCLUDE_COLS)),
        "defaults": defaults,
        "numeric_ranges": numeric_ranges,
        "categorical_values": categorical_values,
        "validation_metrics": {
            "rmse": float(best_valid_rmse),
        },
        "test_metrics": test_metrics,
        "artifacts": {
            "model_path": str(MODEL_PATH),
            "comparison_path": str(COMPARISON_PATH),
            "test_metrics_path": str(TEST_METRICS_PATH),
            "split_indices_path": str(SPLIT_INDICES_PATH),
            "test_predictions_path": str(TEST_PREDICTIONS_PATH),
            "grouped_error_analysis_path": str(GROUPED_ERRORS_PATH),
            "residual_plot_path": str(RESIDUAL_PLOT_PATH),
            "prediction_plot_path": str(PREDICTION_PLOT_PATH),
        },
    }

    META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    TEST_METRICS_PATH.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    print("Training complete")
    print(f"Dropped constant columns: {dropped_constant_cols if dropped_constant_cols else 'None'}")
    print(f"Best model: {best_name}")
    print(f"Validation RMSE: {best_valid_rmse:,.2f}")
    print(f"Test RMSE: {test_metrics['rmse']:,.2f}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metadata: {META_PATH}")
    print(f"Saved comparison: {COMPARISON_PATH}")
    print(f"Saved test metrics: {TEST_METRICS_PATH}")
    print(f"Saved split indices: {SPLIT_INDICES_PATH}")
    print(f"Saved test predictions: {TEST_PREDICTIONS_PATH}")
    print(f"Saved grouped errors: {GROUPED_ERRORS_PATH}")
    print(f"Saved residual plot: {RESIDUAL_PLOT_PATH}")
    print(f"Saved prediction plot: {PREDICTION_PLOT_PATH}")


if __name__ == "__main__":
    main()
