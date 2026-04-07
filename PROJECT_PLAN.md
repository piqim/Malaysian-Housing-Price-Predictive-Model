# Remaining Work Plan: ML Website, App Integration, and Project Handover

## Summary
- Treat Part 5 as partially complete, not fully complete.
- The repo already contains a selected best model, saved artifacts, saved metrics, and a reusable inference module.
- The current implementation priority is the website and deployment path: keep the analysis dashboard as-is, deploy the separate ML Streamlit app, and update the GitHub Pages landing page once the final Streamlit URL exists.

## Current State
- Analysis dashboard is already implemented in `app/streamlit_app.py`.
- ML inference app is implemented in `app/ml_predict_app.py` and is ready to run locally.
- Model training and artifact generation live in `src/modeling/train.py`.
- Inference helpers and validation live in `src/modeling/predict.py`.
- Current v1 best model is `lightgbm`, based on the saved metadata and comparison report.

## Remaining Tasks
1. Deploy `app/ml_predict_app.py` to Streamlit Community Cloud from this repository.
2. Replace the temporary ML card link in `index.html` with the final deployed Streamlit URL.
3. Update the README top link for the ML app once the deployment URL exists.
4. Run `python src/modeling/train.py` whenever the dataset or feature selection changes so the reports and metadata stay current.
5. Decide whether Part 5 v2 should add cross-validation search, more extensive tuning, or explainability features.

## Outputs Already Added
- `models/best_model.joblib`
- `models/model_metadata.json`
- `reports/model/model_comparison.csv`
- `reports/model/test_metrics.json`
- `reports/model/split_indices.csv`
- `reports/model/test_predictions.csv`
- `reports/model/grouped_error_analysis.csv`
- `reports/model/residuals_vs_actual.png`
- `reports/model/predictions_vs_actual.png`

## Acceptance Checks
- `streamlit run app/streamlit_app.py` still works for the analysis dashboard.
- `streamlit run app/ml_predict_app.py` works for the prediction UI.
- `python src/modeling/train.py` regenerates the model artifact and reporting outputs.
- The GitHub Pages landing page has no dead ML card link.
- The final deployed Streamlit URL is reflected in both `README.md` and `index.html`.

## Assumptions
- The ML app remains separate from the descriptive-analysis app.
- Streamlit Community Cloud is the deployment target for v1.
- The current LightGBM pipeline remains the selected production candidate unless retraining changes the ranking.
- Advanced explainability and prediction intervals are deferred until after deployment.
