"""
V14 Opening Weekend Prediction Model Visualization
Interactive dashboard for data scientists
"""

import streamlit as st

st.set_page_config(
    page_title="V14 OW Prediction Model",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("V14 Opening Weekend Prediction Model")
st.subheader("3-Tier Cascade Architecture Visualization")

st.divider()

st.markdown("""
Welcome to the **V14 Box Office Prediction Model** visualization dashboard.

This model predicts movie opening weekend (OW) revenue using a **3-tier cascade architecture**:

| Tier | Revenue Range | Training Films |
|------|---------------|----------------|
| **SMALL** | < $15M | 115 |
| **MID** | $15M - $50M | 78 |
| **LARGE+** | > $50M | 46 |
""")

st.header("Navigation")
st.markdown("""
Use the sidebar to explore:

1. **Architecture** - Model cascade flow and tier configurations
2. **Features** - Feature importance by category
3. **Performance** - Classification accuracy, MAE, confusion matrix
4. **Predictions** - Interactive prediction tool
5. **Errors** - Model limitations and biggest misses
""")

st.divider()

st.header("Quick Stats")
st.caption("V14 at -7 days prediction horizon")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Classification Accuracy", "71.5%", "+3.7% vs V13 4-tier")
col2.metric("Mean Absolute Error", "$13.1M", "-$0.9M vs V13")
col3.metric("LARGE+ Accuracy", "65%", "+38% vs old LARGE")
col4.metric("Training Films", "239", "")

st.divider()

st.header("Model Version")
st.markdown("""
- **Version**: V14 (February 2026)
- **Type**: 3-Tier Cascade with Tier-Specific Regressors
- **Algorithm**: CatBoost
- **Features**: 51 (31 static + 20 Google Trends)
""")
