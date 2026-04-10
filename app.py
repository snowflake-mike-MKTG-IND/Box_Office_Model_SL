"""
V16 Opening Weekend Prediction Model Visualization
Interactive dashboard for data scientists
"""

import streamlit as st
from cortex_badge import show_cortex_badge

st.set_page_config(
    page_title="V16 OW Prediction Model",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("V16 Opening Weekend Prediction Model")
st.subheader("3-Tier Cascade Architecture + TMDB Popularity Override")

st.divider()

st.markdown("""
Welcome to the **V16 Box Office Prediction Model** visualization dashboard.

This model predicts movie opening weekend (OW) revenue using a **3-tier cascade architecture** with an
orthogonal **TMDB popularity override system** (Rule C):

| Tier | Revenue Range | Training Films |
|------|---------------|----------------|
| **SMALL** | < $15M | 148 |
| **MID** | $15M - $50M | 86 |
| **LARGE+** | > $50M | 51 |
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
st.caption("V16 at -7 days prediction horizon")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Training Films", "285", "+16 vs V15")
col2.metric("Features", "56", "+4 vs V15 (IS_MAJOR_STUDIO, TMDB D14/D7, Momentum)")
col3.metric("TMDB Override Accuracy", "100%", "4/4 correct on holdout")
col4.metric("Override Tier Boost", "63% → 84%", "+21pp on blind holdout")

st.divider()

st.header("V16 Improvements over V15")

col1, col2 = st.columns(2)

with col1:
    st.success("**What Changed in V16**")
    st.markdown("""
    - **+16 training films** (269 → 285) — full studio coverage
    - **+4 new features**: `IS_MAJOR_STUDIO`, `TMDB_POPULARITY_D14`, `TMDB_POPULARITY_D7`, `TMDB_POP_MOMENTUM`
    - **56 total features** (36 static + 20 Google Trends)
    - **TMDB Override (Rule C)**: Orthogonal post-model tier override using live TMDB popularity
    - Fixes V15's biggest misses: Project Hail Mary, Hoppers, Reminders of Him
    """)

with col2:
    st.markdown("""
    | Metric | V15 | V16 | Change |
    |--------|-----|-----|--------|
    | Training Films | 269 | **285** | +16 |
    | Features | 52 | **56** | +4 |
    | Holdout Tier Acc (w/ override) | 63.2% | **84.2%** | +21pp |
    | Override Precision | — | **4/4 (100%)** | New |
    | PHM Error | -$54.09M | **+$0.05M** | Fixed |
    | Hoppers Error | -$35.23M | **-$2.97M** | Fixed |
    """)

st.divider()

st.header("Model Version")
st.markdown("""
- **Version**: V16 (April 2026)
- **Type**: 3-Tier Cascade with Tier-Specific Regressors + TMDB Override (Rule C)
- **Algorithm**: CatBoost
- **Features**: 56 (36 static + 20 Google Trends)
- **Override**: Rule C — TMDB D14>=25 → LARGE+, D14>=15 + momentum>=1.3 → MID
- **Training Source**: Snowflake feature view (production schema)
- **SP**: `SPARK_PAR_DEMO.ML_MODEL_TEST.PREDICT_MOVIE_V16`
""")

show_cortex_badge()
