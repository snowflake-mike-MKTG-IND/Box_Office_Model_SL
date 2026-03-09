"""
V15 Opening Weekend Prediction Model Visualization
Interactive dashboard for data scientists
"""

import streamlit as st
from cortex_badge import show_cortex_badge

st.set_page_config(
    page_title="V15 OW Prediction Model",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("V15 Opening Weekend Prediction Model")
st.subheader("3-Tier Cascade Architecture Visualization")

st.divider()

st.markdown("""
Welcome to the **V15 Box Office Prediction Model** visualization dashboard.

This model predicts movie opening weekend (OW) revenue using a **3-tier cascade architecture**:

| Tier | Revenue Range | Training Films |
|------|---------------|----------------|
| **SMALL** | < $15M | 137 |
| **MID** | $15M - $50M | 84 |
| **LARGE+** | > $50M | 48 |
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
st.caption("V15 at -7 days prediction horizon")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Classification Accuracy", "77.3%", "+5.8% vs V14")
col2.metric("Mean Absolute Error", "$11.0M", "-$2.1M vs V14")
col3.metric("LARGE+ Accuracy", "77.1%", "+12.1% vs V14")
col4.metric("Training Films", "269", "+30 vs V14")

st.divider()

st.header("V15 Improvements over V14")

col1, col2 = st.columns(2)

with col1:
    st.success("**What Changed**")
    st.markdown("""
    - **+30 training films** (239 → 269) from data cleanup & new scoring
    - **+1 new feature**: `PREDECESSOR_OW_LOG` (sequel predecessor performance)
    - **52 total features** (32 static + 20 Google Trends)
    - Comprehensive data quality fixes (duplicate cleanup, scoring gaps filled)
    """)

with col2:
    st.markdown("""
    | Metric | V14 | V15 | Change |
    |--------|-----|-----|--------|
    | Classification (-7d) | 71.5% | **77.3%** | +5.8% |
    | MAE (-7d) | $13.1M | **$11.0M** | -$2.1M |
    | LARGE+ Accuracy | 65.0% | **77.1%** | +12.1% |
    | Training Films | 239 | **269** | +30 |
    | Features | 51 | **52** | +1 |
    """)

st.divider()

st.header("Model Version")
st.markdown("""
- **Version**: V15 (March 2026)
- **Type**: 3-Tier Cascade with Tier-Specific Regressors
- **Algorithm**: CatBoost
- **Features**: 52 (32 static + 20 Google Trends)
- **Training Source**: Snowflake feature view (production schema)
""")

show_cortex_badge()
