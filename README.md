# 🏢 Malaysian Housing Price Predictive Model
- Main Application: https://piqim.github.io/Malaysian-Housing-Price-Predictive-Model/
- Analysis Streamlit Application: https://myhopr2.streamlit.app/
- Machine Learning Predictive Model Application: Deployment pending, run locally with `streamlit run app/ml_predict_app.py`

---

## 🎯 Overview

This project analyzes **4,000+ Malaysian housing listings** to uncover pricing patterns, feature relationships, and market insights. The analysis pipeline includes data cleaning, comprehensive statistical analysis, and an interactive Streamlit dashboard for data exploration.

### Key Objectives:
- Clean and standardize raw property data
- Conduct univariate, bivariate, and multivariate analysis
- Identify key price drivers and feature correlations
- Assess amenity impacts on pricing
- Provide ML-ready dataset for predictive modeling
- Create interactive visualizations for stakeholder insights
- Train and compare regression models for price prediction
- Deploy a separate Streamlit app for the final ML predictor

---

## 📊 Dataset

### Source
The dataset used in this project contains Malaysian housing listings scraped from property portals. => Dataset: https://www.kaggle.com/datasets/mcpenguin/raw-malaysian-housing-prices-data/data

**Data Credit**: Original dataset mined and compiled from Malaysian real estate platforms. We acknowledge and thank the original data collectors for making this analysis possible.

### Dataset Overview
- **Total Records**: 4,000 properties
- **Features**: 32 columns
- **Target Variable**: `price` (RM)
- **Time Period**: Current market listings (as of data collection)

### Key Features:
- **Numerical**: Property Size, Bedrooms, Bathrooms, Completion Year, Floors, Total Units, Parking
- **Categorical**: Property Type, Tenure Type, Land Title, Category, Developer
- **Location**: Address, Building Name
- **Amenities**: Nearby School, Mall, Park, Hospital, Highway, Railway Station
- **Facilities**: Building facilities (counted and analyzed)

---

## 🛠️ Technologies Used

### Core Libraries:
- **Python 3.8+** - Programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib** - Static visualizations
- **Seaborn** - Statistical data visualization
- **Plotly** - Interactive visualizations

### Dashboard:
- **Streamlit** - Interactive web dashboard

### Development:
- **VS Code** - IDE
- **Git** - Version control

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Git (optional)

### Step 1: Clone or Download Repository

```bash
# Clone repository (if using Git)
git clone https://github.com/piqim/Malaysian-Housing-Price-Predictive-Model.git
cd Malaysian-Housing-Price-Predictive-Model

# Or download and extract ZIP file
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\Activate.ps1
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt contents:**
```txt
pandas>=2.2.0
numpy>=1.26.0
matplotlib>=3.8.2
seaborn>=0.13.0
plotly>=5.18.0
scikit-learn>=1.3.2
lightgbm==4.6.0
streamlit>=1.29.0
joblib>=1.4.0
openpyxl>=3.1.2
Pillow>=11.0.0
```

### Step 4: Prepare Dataset

Place your `house.csv` file in the `data/raw/` directory.

---

## 📖 Usage Guide

### Quick Start (3 Steps)

```bash
# 0.5 Activate venv: venv\Scripts\Activate.ps1

# 1. Clean the data
python src/data_cleaning.py

# 2. Run descriptive analysis
python src/descriptive_analysis.py

# 3. Launch dashboard
streamlit run app/streamlit_app.py
```

Your dashboard will open automatically at `http://localhost:8501`

### ML Prediction App

Run the separate Streamlit app for the predictive model:

```bash
streamlit run app/ml_predict_app.py
```

### Retraining the Model

Retrain the saved predictive-model artifacts and evaluation outputs:

```bash
python src/modeling/train.py
```

---

## 🔬 Project Phases

### Phase 1: Data Cleaning

**Script:** `src/data_cleaning.py`

#### What it does:
1. **Standardizes numeric fields** - Converts price (removes "RM", spaces), property size (removes "sq.ft"), etc.
2. **Handles binary indicators** - Standardizes Yes/No values to 1/0
3. **Cleans categorical features** - Formats text (Title case, strips whitespace)
4. **Processes text columns** - Extracts facility counts, removes descriptions
5. **Removes non-informative columns** - Drops Firm Number, REN Number, Ad List
6. **Validates data** - Removes impossible values (negative prices, invalid years)
7. **Imputes missing values** - Fills with median for numerical features

#### Output:
- `data/processed/house_cleaned.csv` - Clean, standardized dataset

#### Key Statistics:
- **Rows processed**: 4,000
- **Columns cleaned**: 32 → 29 (after removing non-informative columns)
- **Price format**: "RM 340 000" → 340000 (numeric)
- **Binary features**: Standardized to 0/1

---

### Phase 2: Descriptive Analysis

**Script:** `src/descriptive_analysis.py`

#### Analysis Types:

**1. Univariate Analysis**
- Price distribution (mean, median, skewness, kurtosis)
- Numerical feature distributions
- Outlier detection using IQR method
- Distribution interpretations

**2. Bivariate Analysis**
- Price vs numerical features (correlation, scatter plots)
- Price vs categorical features (box plots, median comparisons)
- Price vs amenities (impact percentage)
- Number of facilities vs price

**3. Multivariate Analysis**
- Correlation matrix (all numerical features)
- Multicollinearity detection
- Property Type × Tenure Type analysis
- Feature interaction effects

#### Outputs:

**7 Visualizations:**
- `01_price_univariate.png` - Price distribution (histogram, box plot, log scale)
- `02_numerical_features.png` - All numerical feature distributions
- `03_price_vs_numerical.png` - Scatter plots with trend lines
- `04_price_vs_categorical.png` - Box plots by categories
- `05_amenity_impact.png` - Amenity price impact bar chart
- `05b_facilities_vs_price.png` - Facilities correlation analysis
- `06_correlation_heatmap.png` - Full correlation matrix

**13 Detailed CSV Reports:**
1. `analysis_price_detailed.csv` - Price statistics with interpretations
2. `analysis_numerical_features_detailed.csv` - All numerical feature stats
3. `analysis_ml_feature_quality.csv` - Feature quality assessment
4. `analysis_correlations_detailed.csv` - Correlation strengths and ML importance
5. `analysis_categorical_features_detailed.csv` - Category-level price analysis
6. `analysis_amenity_impact_detailed.csv` - Amenity impact with recommendations
7. `analysis_facilities_detailed.csv` - Facility count analysis
8. `analysis_facilities_summary.csv` - Facilities correlation summary
9. `analysis_correlation_matrix_full.csv` - Complete correlation matrix
10. `analysis_multicollinearity.csv` - Highly correlated feature pairs
11. `analysis_grouped_property_tenure.csv` - Combined feature analysis
12. `analysis_ml_feature_summary.csv` - Master feature reference
13. `analysis_metadata.csv` - Analysis metadata

**ML-Ready Dataset:**
- `data/final/house_model_ready.csv` - Cleaned data for modeling

---

### Phase 3: Interactive Dashboard

**Script:** `app/streamlit_app.py`

#### Dashboard Pages:

**1. 📊 Overview**
- Key statistics and metrics
- Price distribution overview
- Dataset summary
- Quick data preview

**2. 💰 Price Analysis**
- Detailed price statistics with interpretations
- Interactive price histograms
- Box plots for outlier detection
- Price vs numerical features

**3. 📈 Feature Correlations**
- Correlation table with ML importance ratings
- Interactive correlation bar charts
- Multicollinearity warnings
- Full correlation heatmap

**4. 📑 Categorical Features**
- Interactive category selector
- Dynamic price charts by category
- Property Type × Tenure Type heatmap
- Category insights (premium vs budget)

**5. 🏪 Amenity & Facilities**
- Amenity impact visualization
- Facilities correlation analysis
- Price increase per facility
- High/medium/low impact levels

**6. 📋 Full Reports**
- Access all 13+ analysis CSVs
- Download individual reports
- View detailed statistics

---

### Phase 4: Predictive Modeling and ML App

**Scripts:**
- `src/modeling/train.py`
- `src/modeling/predict.py`
- `app/ml_predict_app.py`

#### Current Status
- Best current v1 model: `lightgbm`
- Saved model artifact: `models/best_model.joblib`
- Saved model metadata: `models/model_metadata.json`
- Model comparison report: `reports/model/model_comparison.csv`
- Test metrics report: `reports/model/test_metrics.json`
- Reproducible split manifest: `reports/model/split_indices.csv`
- Test-level predictions: `reports/model/test_predictions.csv`
- Grouped error analysis: `reports/model/grouped_error_analysis.csv`
- Error plots:
  - `reports/model/residuals_vs_actual.png`
  - `reports/model/predictions_vs_actual.png`

#### What Part 5 Already Covers
- Multiple regression models trained and compared
- Best model selected and serialized for inference
- Prediction helper module for single-row and batch inference
- Test metrics saved for handover and app display

#### What Still Remains
- Publish the ML Streamlit app on Streamlit Community Cloud
- Replace the temporary ML card link on the GitHub Pages site with the deployed Streamlit URL
- Iterate on cross-validation and hyperparameter search if you want a stronger v2 model

---

## ML Predictive Model App

The predictive-model app is intentionally separate from the descriptive-analysis dashboard.

### Features
- Input form generated from saved model metadata
- Numeric bounds based on training-data ranges
- Categorical dropdowns limited to allowed model values
- Validation guardrails for invalid inference inputs
- Predicted house price in RM
- Inline display of saved test metrics

### Local Run Command

```bash
streamlit run app/ml_predict_app.py
```

### Deployment Note
- Deploy `app/ml_predict_app.py` on Streamlit Community Cloud from this repository.
- After deployment, replace the temporary ML card link in `index.html` with the final Streamlit URL.
- Until that URL exists, the website links to this repository guide instead of a fake deployment.

---

## 👥 Contributors

### Data Analysis & Cleaning:
- **Mustaqim Bin Burhanuddin** - Data cleaning, descriptive analysis, dashboard development

### Machine Learning:
- **Eliot Boda** - Predictive modeling, inference app, and deployment handover

### Data Source:
- Original dataset compiled from Malaysian property portals
- Dataset: https://www.kaggle.com/datasets/mcpenguin/raw-malaysian-housing-prices-data/data
- Credit to the original data miners and property listing platforms

---

## 📄 License

This project is for educational and analytical purposes. The dataset is used under fair use for academic research.

**Note**: If you use this code or analysis, please:
1. Credit the original data source
2. Acknowledge this repository
3. Not to be used for commercial purposes without permission


**Last Updated**: April 7, 2026

**Version**: 1.1.0 (Analysis Complete, ML App Local-Ready)
