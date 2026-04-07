"""Streamlit app for Malaysian housing price prediction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.predict import ValidationError, load_model, predict_one

REPORTS_DIR = PROJECT_ROOT / "reports" / "model"

st.set_page_config(
    page_title="Malaysian Housing Price Predictor",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        color: #1f77b4;
        text-align: left;
        padding: 1rem 0;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        border-radius: 0.3rem;
        margin: 1rem 0;
        color: #123548;
    }
    .insight-box strong {
        color: #0b4f7d;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-left: 4px solid #ffc107;
        border-radius: 0.3rem;
        margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_bundle():
    return load_model()


@st.cache_data
def load_report_data() -> dict[str, pd.DataFrame | dict | None]:
    data: dict[str, pd.DataFrame | dict | None] = {
        "comparison": None,
        "grouped_errors": None,
        "test_predictions": None,
        "test_metrics": None,
    }

    csv_files = {
        "comparison": REPORTS_DIR / "model_comparison.csv",
        "grouped_errors": REPORTS_DIR / "grouped_error_analysis.csv",
        "test_predictions": REPORTS_DIR / "test_predictions.csv",
    }

    for key, path in csv_files.items():
        if path.exists():
            data[key] = pd.read_csv(path)

    test_metrics_path = REPORTS_DIR / "test_metrics.json"
    if test_metrics_path.exists():
        data["test_metrics"] = json.loads(test_metrics_path.read_text(encoding="utf-8"))

    return data


def format_label(column: str) -> str:
    label_map = {
        "Bedroom": "Bedrooms",
        "Bathroom": "Bathrooms",
        "Property Size": "Property Size (sq ft)",
        "Tenure Type": "Tenure Type",
        "Completion Year": "Completion Year",
        "# of Floors": "Building Floors",
        "Total Units": "Total Units",
        "Property Type": "Property Type",
        "Parking Lot": "Parking Lots",
        "Floor Range": "Floor Range",
        "Land Title": "Land Title",
        "Firm Type": "Firm Type",
        "num_facilities": "Number of Facilities",
    }
    return label_map.get(column, column)


def build_numeric_input(column: str, metadata: dict, container) -> float:
    bounds = metadata["numeric_ranges"][column]
    default = float(metadata["defaults"][column])
    min_value = float(bounds["min"])
    max_value = float(bounds["max"])
    integer_like = default.is_integer() and min_value.is_integer() and max_value.is_integer()

    if integer_like:
        return float(
            container.number_input(
                format_label(column),
                min_value=int(min_value),
                max_value=int(max_value),
                value=int(default),
                step=1,
            )
        )

    return float(
        container.number_input(
            format_label(column),
            min_value=min_value,
            max_value=max_value,
            value=default,
            step=1.0,
        )
    )


def build_categorical_input(column: str, metadata: dict, container) -> str:
    options = metadata["categorical_values"][column]
    default = metadata["defaults"][column]
    default_index = options.index(default) if default in options else 0
    return container.selectbox(format_label(column), options, index=default_index)


def render_prediction_form(model, metadata: dict) -> None:
    st.markdown("### 🧮 Prediction Form")
    st.write("Enter the property details below to estimate a housing price in RM.")
    st.caption("Defaults are pulled from the saved model metadata and numeric values are limited to the training-data ranges.")

    payload: dict[str, float | str] = {}

    with st.form("prediction_form", clear_on_submit=False):
        left_col, right_col = st.columns(2)

        numeric_features = metadata.get("numeric_features", [])
        categorical_features = metadata.get("categorical_features", [])

        for index, column in enumerate(numeric_features):
            target_column = left_col if index % 2 == 0 else right_col
            payload[column] = build_numeric_input(column, metadata, target_column)

        for index, column in enumerate(categorical_features):
            target_column = left_col if index % 2 == 0 else right_col
            payload[column] = build_categorical_input(column, metadata, target_column)

        submitted = st.form_submit_button("Predict Price", use_container_width=True)

    if submitted:
        try:
            prediction = predict_one(payload, model=model, metadata=metadata)
        except ValidationError as exc:
            st.error(str(exc))
        except Exception as exc:  # pragma: no cover - UI guard
            st.error(f"Prediction failed: {exc}")
        else:
            st.success("Prediction completed.")
            st.metric("Predicted Price", f"RM {prediction:,.0f}")
            st.markdown(
                """
                <div class="insight-box">
                    <strong>Interpretation:</strong> This v1 app returns a point prediction from the saved LightGBM model.
                    Use it as a model estimate, not as a formal valuation.
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_overview(metadata: dict, reports: dict[str, pd.DataFrame | dict | None]) -> None:
    st.markdown('<h1 class="main-header">🤖 Overview of The ML Predictive Model</h1>', unsafe_allow_html=True)
    st.markdown("---")

    test_metrics = metadata.get("test_metrics", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Model", str(metadata.get("best_model", "Unknown")).title())
    col2.metric("Test RMSE", f"RM {test_metrics.get('rmse', 0):,.0f}")
    col3.metric("Test MAE", f"RM {test_metrics.get('mae', 0):,.0f}")
    col4.metric("Median Abs Error", f"RM {test_metrics.get('median_abs_error', 0):,.0f}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 💡 Key Insight(s)")
        st.write("➡️ **LightGBM is the current best v1 model** based on the saved validation comparison output.")
        st.write("➡️ **Median absolute error is easier to explain in RM** than R² when showing business-facing performance.")
        st.write("➡️ **The app uses the same saved pipeline artifact as local inference**, so UI predictions match the trained model.")

    with col2:
        st.markdown("#### ‼️ Model Summary")
        st.write(f"📌 **Target**: {metadata.get('target', 'price')}")
        st.write(f"📌 **Transform**: {metadata.get('transform', 'log1p')}")
        st.write(f"📌 **Feature Count**: {len(metadata.get('feature_columns', []))}")
        st.write(f"📌 **Trained At (UTC)**: {metadata.get('trained_at_utc', 'Unknown')}")

    st.markdown("---")
    st.markdown("### 📋 Inference Inputs")
    st.dataframe(
        pd.DataFrame(
            {
                "Feature": metadata.get("feature_columns", []),
                "Default": [metadata.get("defaults", {}).get(feature) for feature in metadata.get("feature_columns", [])],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    if isinstance(reports.get("comparison"), pd.DataFrame):
        st.markdown("---")
        st.markdown("### 🏁 Model Ranking Snapshot")
        comparison_df = reports["comparison"].copy()
        st.dataframe(comparison_df.head(5), use_container_width=True, hide_index=True)


def render_model_evaluation(reports: dict[str, pd.DataFrame | dict | None]) -> None:
    st.markdown('<h1 class="main-header">📈 Model Evaluation and Error Analysis</h1>', unsafe_allow_html=True)
    st.markdown("---")

    residual_plot = REPORTS_DIR / "residuals_vs_actual.png"
    prediction_plot = REPORTS_DIR / "predictions_vs_actual.png"

    if residual_plot.exists() and prediction_plot.exists():
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Residuals vs Actual")
            st.image(Image.open(residual_plot), use_container_width=True)
        with col2:
            st.markdown("### Predicted vs Actual")
            st.image(Image.open(prediction_plot), use_container_width=True)

    grouped_errors = reports.get("grouped_errors")
    if isinstance(grouped_errors, pd.DataFrame):
        st.markdown("---")
        st.markdown("### 📊 Grouped Error Analysis")
        st.dataframe(grouped_errors, use_container_width=True, hide_index=True)

    test_predictions = reports.get("test_predictions")
    if isinstance(test_predictions, pd.DataFrame):
        st.markdown("---")
        st.markdown("### 📋 Test Prediction Sample")
        st.dataframe(test_predictions.head(20), use_container_width=True)


def render_reports(reports: dict[str, pd.DataFrame | dict | None]) -> None:
    st.markdown('<h1 class="main-header">📋 ML Reports and Downloads</h1>', unsafe_allow_html=True)
    st.markdown("---")

    report_options = {
        "Model Comparison": "comparison",
        "Grouped Error Analysis": "grouped_errors",
        "Test Predictions": "test_predictions",
    }

    selected_report = st.selectbox("Select a report to view", list(report_options.keys()))
    report_key = report_options[selected_report]
    report_df = reports.get(report_key)

    if isinstance(report_df, pd.DataFrame):
        st.dataframe(report_df, use_container_width=True)
        csv_data = report_df.to_csv(index=False)
        st.download_button(
            label=f"📥 Download {selected_report}",
            data=csv_data,
            file_name=f"{report_key}.csv",
            mime="text/csv",
        )
    else:
        st.warning(f"Report not available: {selected_report}")


try:
    model, metadata = load_model_bundle()
except Exception as exc:  # pragma: no cover - UI guard
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

reports = load_report_data()

st.sidebar.markdown("# ⏩ Navigation Menu")
st.sidebar.markdown("Use the menu below to explore the predictive-model dashboard.")
st.sidebar.markdown("---")

st.sidebar.markdown("## 📂 Sections")
page = st.sidebar.radio(
    "Choose a section:",
    [
        "📊 Overview",
        "🧮 Predict Price",
        "📈 Model Evaluation",
        "📋 Full Reports",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("## ℹ️ About This Dashboard")
st.sidebar.markdown(
    "This dashboard presents the trained machine learning model, local inference form, and error-analysis reports for the Malaysian housing price prediction workflow."
)

if page == "📊 Overview":
    render_overview(metadata, reports)
elif page == "🧮 Predict Price":
    st.markdown('<h1 class="main-header">🧮 Predict Malaysian Housing Price</h1>', unsafe_allow_html=True)
    st.markdown("---")
    render_prediction_form(model, metadata)
elif page == "📈 Model Evaluation":
    render_model_evaluation(reports)
elif page == "📋 Full Reports":
    render_reports(reports)

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 2rem;'>
        <p>Data Analysis By <a href="https://github.com/piqim" target="_blank" style="font-weight: bold; text-decoration: none; color: inherit;">Mustaqim Burhanuddin</a></p>
        <p>ML Modeling and Prediction App By <a href="https://github.com/Eliot-2006" target="_blank" style="font-weight: bold; text-decoration: none; color: inherit;">Eliot Boda</a></p>
        <p>Built with Python Streamlit, Pandas, and scikit-learn</p>
    </div>
    """,
    unsafe_allow_html=True,
)
