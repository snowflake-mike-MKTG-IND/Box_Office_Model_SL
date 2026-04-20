"""
V17 Opening Weekend Prediction Model Visualization
Interactive dashboard for data scientists
"""

import streamlit as st
from cortex_badge import show_cortex_badge

st.set_page_config(
    page_title="V17 OW Prediction Model",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("V17 Opening Weekend Prediction Model")
st.subheader("3-Tier Cascade Architecture + TMDB Popularity Override")

st.divider()

st.markdown("""
Welcome to the **V17 Box Office Prediction Model** visualization dashboard.

This model predicts movie opening weekend (OW) revenue using a **3-tier cascade architecture** with an
orthogonal **TMDB popularity override system** (Rule C):

| Tier | Revenue Range | Training Films |
|------|---------------|----------------|
| **SMALL** | < $15M | 143 |
| **MID** | $15M - $50M | 84 |
| **LARGE+** | > $50M | 50 |
""")

st.header("Navigation")
st.markdown("""
Use the sidebar to explore:

0. **Cortex Code** - AI-assisted development story
1. **Architecture** - Model cascade flow and tier configurations
2. **Features** - Feature importance by category
3. **Performance** - Classification accuracy, MAE, confusion matrix
4. **Predictions** - Interactive prediction tool
5. **Errors** - Model limitations and biggest misses
6. **Timeline** - Model development history
7. **Recent Predictions** - Live tracking of predictions vs actual results
""")

st.divider()

st.header("Quick Stats")
st.caption("V17 cross-validation at -7 days prediction horizon")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Training Films", "277", "+3 new trends features")
col2.metric("Features", "59", "+3 vs V16 (ROLLING_14D, ROLLING_21D, TRENDS_EARLIEST)")
col3.metric("CV Accuracy (-7d)", "73.2%", "5-fold GroupKFold")
col4.metric("CV MAE (-7d)", "$11.74M", "Across all tiers")

st.divider()

st.header("V17 Improvements over V16")

col1, col2 = st.columns(2)

with col1:
    st.success("**What Changed in V17**")
    st.markdown("""
    - **+3 new trends features**: `ROLLING_14D`, `ROLLING_21D`, `TRENDS_EARLIEST`
    - **59 total features** (36 static + 23 Google Trends)
    - **5-fold GroupKFold cross-validation** grouped by MOVIE_ID
    - **277 training films** (cleaned dataset)
    - ROLLING_21D ranks #4 in SMALL regressor at -14d (importance=3.93)
    """)

with col2:
    st.markdown("""
    | Metric | V16 | V17 | Change |
    |--------|-----|-----|--------|
    | Training Films | 285 | **277** | Cleaned |
    | Features | 56 | **59** | +3 trends |
    | CV Accuracy (-7d) | 73.2% | **73.2%** | Same |
    | CV MAE (-7d) | $11.78M | **$11.74M** | -$0.04M |
    | CV MAE (-14d) | $11.57M | **$11.56M** | -$0.01M |
    """)
    st.caption("V17 vs V16 comparison from fair 5-fold GroupKFold CV on identical data and folds.")

st.divider()

st.header("Model Version")
st.markdown("""
- **Version**: V17 (April 2026)
- **Type**: 3-Tier Cascade with Tier-Specific Regressors + TMDB Override (Rule C)
- **Algorithm**: CatBoost
- **Features**: 59 (36 static + 23 Google Trends)
- **Override**: Rule C — TMDB D14>=25 → LARGE+, D14>=15 + momentum>=1.3 → MID
- **Training Source**: Snowflake feature view (production schema)
- **Model File**: `ow_pipeline_v17_production.joblib.gz`
""")

show_cortex_badge()
