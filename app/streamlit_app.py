"""
Malaysian Condominium Price Predictive Model
Interactive Streamlit Dashboard - Descriptive Analysis Results
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from PIL import Image

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Malaysian Condo Price Analysis",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
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
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-left: 4px solid #ffc107;
        border-radius: 0.3rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# LOAD DATA FUNCTIONS
# ============================================================================

@st.cache_data
def load_data():
    """Load all necessary data files"""
    data_dict = {}
    
    # Main dataset
    try:
        data_dict['main'] = pd.read_csv('data/final/house_model_ready.csv')
    except FileNotFoundError:
        st.error("Main dataset not found: data/final/house_model_ready.csv")
        return None
    
    # Load all CSV files with error handling
    files_to_load = {
        'summary': 'data/processed/summary_statistics.csv',
        'price_detailed': 'data/processed/analysis_price_detailed.csv',
        'numerical_detailed': 'data/processed/analysis_numerical_features_detailed.csv',
        'ml_quality': 'data/processed/analysis_ml_feature_quality.csv',
        'correlations': 'data/processed/analysis_correlations_detailed.csv',
        'categorical': 'data/processed/analysis_categorical_features_detailed.csv',
        'amenities': 'data/processed/amenity_impact.csv',
        'facilities_detailed': 'data/processed/analysis_facilities_detailed.csv',
        'facilities_summary': 'data/processed/analysis_facilities_summary.csv',
        'multicollinearity': 'data/processed/analysis_multicollinearity.csv',
        'grouped_analysis': 'data/processed/analysis_grouped_property_tenure.csv',
        'ml_summary': 'data/processed/analysis_ml_feature_summary.csv',
        'metadata': 'data/processed/analysis_metadata.csv',
        'price_correlations': 'data/processed/price_correlations.csv',
        'numerical_summary': 'data/processed/numerical_features_summary.csv'
    }
    
    for key, filepath in files_to_load.items():
        try:
            data_dict[key] = pd.read_csv(filepath)
        except FileNotFoundError:
            st.warning(f"Optional file not found: {filepath}")
            data_dict[key] = None
    
    return data_dict

data = load_data()

if data is None or data['main'] is None:
    st.error("❌ Could not load data files. Please run the analysis first:")
    st.code("python src/descriptive_analysis.py", language="bash")
    st.stop()

df = data['main']

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.markdown("# 🏢 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Choose a section:",
    [
        "📊 Overview",
        "💰 Price Analysis", 
        "📈 Feature Correlations",
        "📑 Categorical Features",
        "🏪 Amenity & Facilities",
        "📋 Full Reports"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Dataset Info")
st.sidebar.metric("Total Properties", f"{len(df):,}")
st.sidebar.metric("Features", f"{len(df.columns)}")

if data['metadata'] is not None:
    meta = data['metadata']
    st.sidebar.metric("Numerical Features", int(meta['Numerical_Features'].values[0]))
    st.sidebar.metric("Categorical Features", int(meta['Categorical_Features'].values[0]))

st.sidebar.markdown("---")

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================

if page == "📊 Overview":
    st.markdown('<p class="main-header">🏢 Malaysian Condominium Price Analysis</p>', unsafe_allow_html=True)
    st.markdown("### Comprehensive Descriptive Analysis Dashboard")
    
    st.markdown("---")
    
    # Key Metrics
    st.markdown("## 📊 Key Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Average Price",
            f"RM {df['price'].mean():,.0f}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Median Price",
            f"RM {df['price'].median():,.0f}",
            delta=None
        )
    
    with col3:
        if 'Property Size' in df.columns:
            st.metric(
                "Avg Property Size",
                f"{df['Property Size'].mean():,.0f} sq ft",
                delta=None
            )
    
    with col4:
        if 'Bedroom' in df.columns:
            st.metric(
                "Most Common Bedrooms",
                f"{int(df['Bedroom'].mode().values[0])}",
                delta=None
            )
    
    st.markdown("---")
    
    # Price Distribution Visualization
    st.markdown("## 💰 Price Distribution Overview")
    
    if os.path.exists('reports/figures/01_price_univariate.png'):
        image = Image.open('reports/figures/01_price_univariate.png')
        st.image(image, use_container_width=True)
    
    # Key Insights
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("### 💡 Key Insights")
    
    skewness = df['price'].skew()
    if skewness > 1:
        st.write("📌 **Price Distribution**: Highly right-skewed - market has many luxury condominiums")
    elif skewness > 0.5:
        st.write("📌 **Price Distribution**: Moderately right-skewed - some high-end properties")
    else:
        st.write("📌 **Price Distribution**: Fairly symmetric distribution")
    
    st.write(f"📌 **Price Range**: RM {df['price'].min():,.0f} to RM {df['price'].max():,.0f}")
    st.write(f"📌 **Standard Deviation**: RM {df['price'].std():,.0f}")
    st.write(f"📌 **Skewness**: {skewness:.2f}")
    st.write(f"📌 **Kurtosis**: {df['price'].kurtosis():.2f}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dataset Summary
    if data['metadata'] is not None:
        st.markdown("## 📋 Analysis Summary")
        meta = data['metadata']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Analysis Date**: {meta['Analysis_Date'].values[0]}")
            st.write(f"**Total Records**: {int(meta['Total_Records'].values[0]):,}")
            st.write(f"**Total Features**: {int(meta['Total_Features'].values[0])}")
            st.write(f"**Numerical Features**: {int(meta['Numerical_Features'].values[0])}")
            st.write(f"**Categorical Features**: {int(meta['Categorical_Features'].values[0])}")
        
        with col2:
            st.write(f"**Target Variable**: {meta['Target_Variable'].values[0]}")
            st.write(f"**Target Mean**: RM {meta['Target_Mean'].values[0]:,.0f}")
            st.write(f"**Target Median**: RM {meta['Target_Median'].values[0]:,.0f}")
            st.write(f"**Features (High Correlation)**: {int(meta['Features_High_Correlation'].values[0])}")
    
    st.markdown("---")
    
    # Quick Data Preview
    st.markdown("## 📋 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

# ============================================================================
# PAGE 2: PRICE ANALYSIS
# ============================================================================

elif page == "💰 Price Analysis":
    st.markdown('<p class="main-header">💰 Price Analysis</p>', unsafe_allow_html=True)
    
    # Detailed Price Statistics
    st.markdown("## 📊 Detailed Price Statistics")
    
    if data['price_detailed'] is not None:
        st.dataframe(data['price_detailed'], use_container_width=True)
        
        # Highlight key interpretations
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("### 🎯 Interpretations")
        for idx, row in data['price_detailed'].iterrows():
            if row['Metric'] in ['Skewness', 'Kurtosis', 'IQR']:
                st.write(f"**{row['Metric']}**: {row['Interpretation']}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Price Distribution Visualizations
    st.markdown("## 📈 Price Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        fig = px.histogram(
            df, 
            x='price', 
            nbins=50,
            title='Price Distribution',
            labels={'price': 'Price (RM)', 'count': 'Frequency'}
        )
        fig.add_vline(x=df['price'].mean(), line_dash="dash", line_color="red", 
                     annotation_text="Mean", annotation_position="top")
        fig.add_vline(x=df['price'].median(), line_dash="dash", line_color="green", 
                     annotation_text="Median", annotation_position="top")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Box plot
        fig = px.box(
            df, 
            y='price',
            title='Price Box Plot (Outlier Detection)',
            labels={'price': 'Price (RM)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Saved visualizations
    st.markdown("## 📊 Price vs Numerical Features")
    
    if os.path.exists('reports/figures/03_price_vs_numerical.png'):
        image = Image.open('reports/figures/03_price_vs_numerical.png')
        st.image(image, use_container_width=True)

# ============================================================================
# PAGE 3: FEATURE CORRELATIONS
# ============================================================================

elif page == "📈 Feature Correlations":
    st.markdown('<p class="main-header">📈 Feature Correlations with Price</p>', unsafe_allow_html=True)
    
    # Correlation Analysis
    st.markdown("## 🔗 Detailed Correlation Analysis")
    
    if data['correlations'] is not None:
        # Display correlation table
        st.dataframe(
            data['correlations'].sort_values('Abs_Correlation', ascending=False),
            use_container_width=True
        )
        
        # Bar chart of correlations
        fig = px.bar(
            data['correlations'].sort_values('Correlation', ascending=False),
            x='Correlation',
            y='Feature',
            orientation='h',
            title='Feature Correlations with Price',
            color='Correlation',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            labels={'Correlation': 'Correlation Coefficient'},
            hover_data=['Strength', 'ML_Importance']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("### 💡 Top Predictive Features")
        top_features = data['correlations'].nlargest(5, 'Abs_Correlation')
        for idx, row in top_features.iterrows():
            st.write(f"**{row['Feature']}**: {row['Interpretation']} (Correlation: {row['Correlation']:.3f})")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Numerical Features
    st.markdown("## 📊 Numerical Features Distribution")
    
    if os.path.exists('reports/figures/02_numerical_features.png'):
        image = Image.open('reports/figures/02_numerical_features.png')
        st.image(image, use_container_width=True)
    
    if data['numerical_detailed'] is not None:
        with st.expander("📋 View Detailed Numerical Features Statistics"):
            st.dataframe(data['numerical_detailed'], use_container_width=True)
    
    st.markdown("---")
    
    # Multicollinearity Check
    st.markdown("## ⚠️ Multicollinearity Detection")
    
    if data['multicollinearity'] is not None and len(data['multicollinearity']) > 0:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.write("**Features with high correlation detected (potential multicollinearity):**")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.dataframe(data['multicollinearity'], use_container_width=True)
    else:
        st.success("✅ No significant multicollinearity detected!")
    
    # Correlation Heatmap
    st.markdown("## 🔥 Correlation Heatmap")
    
    if os.path.exists('reports/figures/06_correlation_heatmap.png'):
        image = Image.open('reports/figures/06_correlation_heatmap.png')
        st.image(image, use_container_width=True)

# ============================================================================
# PAGE 4: CATEGORICAL FEATURES
# ============================================================================

elif page == "📑 Categorical Features":
    st.markdown('<p class="main-header">📑 Categorical Features Analysis</p>', unsafe_allow_html=True)
    
    st.markdown("## 📊 Price by Categories")
    
    if os.path.exists('reports/figures/04_price_vs_categorical.png'):
        image = Image.open('reports/figures/04_price_vs_categorical.png')
        st.image(image, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed categorical analysis
    st.markdown("## 📋 Detailed Category Analysis")
    
    if data['categorical'] is not None:
        # Filter selector
        features = data['categorical']['Feature'].unique()
        selected_feature = st.selectbox(
            "Select Feature to Analyze",
            features
        )
        
        filtered_cat = data['categorical'][data['categorical']['Feature'] == selected_feature]
        
        # Display table
        st.dataframe(filtered_cat, use_container_width=True)
        
        # Visualization
        fig = px.bar(
            filtered_cat.sort_values('Median_Price', ascending=False),
            x='Category',
            y='Median_Price',
            title=f'Median Price by {selected_feature}',
            labels={'Median_Price': 'Median Price (RM)'},
            color='Diff_from_Overall_Median_Pct',
            color_continuous_scale='RdYlGn',
            hover_data=['Count', 'Mean_Price', 'Interpretation']
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown(f"### 💡 Insights for {selected_feature}")
        
        highest = filtered_cat.nlargest(1, 'Median_Price').iloc[0]
        lowest = filtered_cat.nsmallest(1, 'Median_Price').iloc[0]
        
        st.write(f"**Highest Price**: {highest['Category']} - RM {highest['Median_Price']:,.0f}")
        st.write(f"  ↳ {highest['Interpretation']}")
        st.write(f"**Lowest Price**: {lowest['Category']} - RM {lowest['Median_Price']:,.0f}")
        st.write(f"  ↳ {lowest['Interpretation']}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Grouped Analysis
    st.markdown("## 🔀 Combined Feature Analysis")
    
    if data['grouped_analysis'] is not None:
        st.markdown("### Property Type × Tenure Type Analysis")
        st.dataframe(data['grouped_analysis'], use_container_width=True)
        
        # Heatmap
        pivot_data = data['grouped_analysis'].pivot_table(
            values='median',
            index='Property Type',
            columns='Tenure Type',
            aggfunc='first'
        )
        
        fig = px.imshow(
            pivot_data,
            labels=dict(x="Tenure Type", y="Property Type", color="Median Price"),
            title="Price Heatmap: Property Type × Tenure Type",
            color_continuous_scale='YlOrRd',
            aspect="auto"
        )
        fig.update_xaxes(side="top")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 5: AMENITY & FACILITIES
# ============================================================================

elif page == "🏪 Amenity & Facilities":
    st.markdown('<p class="main-header">🏪 Amenity & Facilities Impact</p>', unsafe_allow_html=True)
    
    # Amenity Impact
    st.markdown("## 📊 Amenity Impact on Price")
    
    if data['amenities'] is not None and len(data['amenities']) > 0:
        # Display amenity table
        st.dataframe(data['amenities'], use_container_width=True)
        
        # Visualization
        fig = px.bar(
            data['amenities'].sort_values('Difference (%)', ascending=False),
            x='Amenity',
            y='Difference (%)',
            title='Price Impact of Amenities (%)',
            labels={'Difference (%)': 'Price Difference (%)'},
            color='Difference (%)',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0
        )
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("### 💡 Amenity Insights")
        
        positive = data['amenities'][data['amenities']['Difference (%)'] > 0].sort_values('Difference (%)', ascending=False)
        if len(positive) > 0:
            st.write("**Most Valuable Amenities:**")
            for idx, row in positive.head(3).iterrows():
                st.write(f"• **{row['Amenity']}**: +{row['Difference (%)']:.1f}% price increase")
        
        negative = data['amenities'][data['amenities']['Difference (%)'] < 0].sort_values('Difference (%)')
        if len(negative) > 0:
            st.write("\n**Amenities Associated with Lower Prices:**")
            for idx, row in negative.head(3).iterrows():
                st.write(f"• **{row['Amenity']}**: {row['Difference (%)']:.1f}% price decrease")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Saved visualization
    if os.path.exists('reports/figures/05_amenity_impact.png'):
        st.markdown("### 📊 Amenity Impact Visualization")
        image = Image.open('reports/figures/05_amenity_impact.png')
        st.image(image, use_container_width=True)
    
    st.markdown("---")
    
    # Facilities Analysis
    st.markdown("## 🏗️ Number of Facilities Analysis")
    
    if data['facilities_summary'] is not None:
        # Summary
        summary = data['facilities_summary'].iloc[0]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Correlation", f"{summary['Correlation']:.3f}")
        with col2:
            st.metric("Strength", summary['Strength'])
        with col3:
            st.metric("Avg Price Increase", f"RM {summary['Avg_Price_Increase_Per_Facility']:,.0f}")
        
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.write(f"**Interpretation**: {summary['Interpretation']}")
        st.write(f"**ML Recommendation**: {summary['ML_Recommendation']}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    if data['facilities_detailed'] is not None:
        st.markdown("### Detailed Facilities Breakdown")
        st.dataframe(data['facilities_detailed'], use_container_width=True)
    
    # Facilities visualization
    if os.path.exists('reports/figures/05b_facilities_vs_price.png'):
        st.markdown("### 📊 Facilities vs Price Visualization")
        image = Image.open('reports/figures/05b_facilities_vs_price.png')
        st.image(image, use_container_width=True)

# ============================================================================
# PAGE 6: FULL REPORTS
# ============================================================================

elif page == "📋 Full Reports":
    st.markdown('<p class="main-header">📋 Comprehensive Analysis Reports</p>', unsafe_allow_html=True)
    
    st.markdown("## 📊 Available Reports")
    st.markdown("Select a report to view detailed analysis results.")
    
    st.markdown("---")
    
    # Report selector
    report_options = {
        "Price Detailed Analysis": 'price_detailed',
        "Numerical Features Analysis": 'numerical_detailed',
        "Feature Quality Assessment": 'ml_quality',
        "Correlation Analysis": 'correlations',
        "Categorical Features Analysis": 'categorical',
        "Amenity Impact Analysis": 'amenities',
        "Facilities Detailed Analysis": 'facilities_detailed',
        "Facilities Summary": 'facilities_summary',
        "Multicollinearity Detection": 'multicollinearity',
        "Grouped Analysis (Property × Tenure)": 'grouped_analysis',
        "ML Feature Summary": 'ml_summary',
        "Analysis Metadata": 'metadata'
    }
    
    selected_report = st.selectbox(
        "Select Report",
        list(report_options.keys())
    )
    
    # Display selected report
    report_key = report_options[selected_report]
    
    if data[report_key] is not None:
        st.markdown(f"### {selected_report}")
        st.dataframe(data[report_key], use_container_width=True)
        
        # Download button
        csv = data[report_key].to_csv(index=False)
        st.download_button(
            label=f"📥 Download {selected_report}",
            data=csv,
            file_name=f"{report_key}.csv",
            mime='text/csv'
        )
    else:
        st.warning(f"Report not available: {selected_report}")
    
    st.markdown("---")
    
    # Files location info
    st.markdown("## 📁 File Locations")
    st.info("""
    All analysis files are stored in:
    - **Visualizations**: `reports/figures/`
    - **Data CSVs**: `data/processed/`
    - **ML-Ready Data**: `data/final/house_model_ready.csv`
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; padding: 2rem;'>
        <p>Malaysian Condominium Price Predictive Model | Descriptive Analysis Dashboard</p>
        <p>Built with Streamlit 🎈 | Data Analysis with Python 🐍</p>
    </div>
    """, unsafe_allow_html=True)