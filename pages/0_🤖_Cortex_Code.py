import streamlit as st
import pandas as pd
from cortex_badge import show_cortex_badge

st.set_page_config(page_title="AI-Assisted Development", page_icon="🤖", layout="wide")

st.title("🤖 AI-Assisted ML Development")
st.subheader("How Cortex Code Accelerated the Entire ML Lifecycle")

st.divider()

st.markdown("""
This model and dashboard were built collaboratively using **[Cortex Code](https://docs.snowflake.com/en/user-guide/cortex-code/cortex-code)**, 
Snowflake's AI development environment. From SQL pipeline construction to model architecture 
iteration to this dashboard — Cortex Code accelerated every phase of the ML lifecycle.
""")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Development Time", "~32.5 hrs", "7 active sessions")
col2.metric("Snowflake Artifacts", "663", "Tables + Views + Procs")
col3.metric("Model Iterations", "9 versions", "76 experiments")
col4.metric("HP Configurations", "104", "Grid + random search")

st.info(
    "**One person, one AI, one week.** What would traditionally require a data engineer, "
    "ML engineer, and dashboard developer over 4–6 weeks was completed in ~32.5 hours of "
    "active development across 7 sessions."
)

st.divider()

st.header("What Cortex Code Did")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Engineering")
    st.markdown("""
    - Built 441 tables, 152 views, 70 stored procedures in Snowflake
    - Wrote and debugged complex SQL for Google Trends normalization (scaled against Movie Showtimes baseline)
    - Built YouTube sentiment pipeline and external API integrations
    - Managed canonical movie ID mapping — the single source of truth for all movie IDs
    """)

    st.subheader("Feature Engineering")
    st.markdown("""
    - Discovered and implemented 52 features across 8 categories
    - Created rolling window calculations (3D, 5D, 7D) for Google Trends data
    - Built star power metrics from actor box office history (4,588 actors indexed)
    - Engineered interaction features (Trends × IP, Star × Rolling, Sentiment × Trends)
    """)

with col2:
    st.subheader("Model Development")
    st.markdown("""
    - Iterated through 9 model versions and 76 experiments
    - Managed CatBoost hyperparameter tuning (104 HP configurations)
    - Implemented 3-tier cascade architecture with tier-specific loss functions
    - Handled cross-validation, train/test splits, and model evaluation
    """)

    st.subheader("Pipeline & Deployment")
    st.markdown("""
    - Built prediction scripts for weekly production runs
    - Created automated data fetching and enrichment pipelines
    - Deployed models to Snowflake stages
    - Built this entire 8-page Streamlit dashboard
    """)

st.divider()

st.header("The Velocity Multiplier")

col1, col2 = st.columns(2)

with col1:
    comparison_data = {
        "Metric": [
            "Calendar Time",
            "Active Work Time",
            "Team Size",
            "Snowflake Artifacts",
            "ML Experiments",
            "Dashboard Pages",
            "Production Models",
        ],
        "With Cortex Code": [
            "7 active days",
            "~32.5 hours",
            "1 person",
            "663 (441T + 152V + 70P)",
            "76 + 104 HP",
            "8 pages",
            "4 horizon models",
        ],
        "Traditional Estimate": [
            "4–6 weeks",
            "200–300 hours",
            "2–3 engineers",
            "Similar scope",
            "Similar scope",
            "Similar scope",
            "Similar scope",
        ],
    }
    st.dataframe(
        pd.DataFrame(comparison_data),
        use_container_width=True,
        hide_index=True,
    )

with col2:
    st.success(
        "**One person. One AI. One week.**\n\n"
        "What would traditionally require a data engineer, ML engineer, and "
        "dashboard developer over 4–6 weeks was completed in ~32.5 hours of "
        "active development across 7 sessions.\n\n"
        "Cortex Code didn't just write code — it managed the full ML lifecycle: "
        "data exploration, hypothesis generation, experiment management, "
        "debugging, pipeline orchestration, and visualization."
    )

    st.markdown("""
    **Key acceleration areas:**
    - **SQL generation**: Complex joins, window functions, CTEs written in seconds
    - **Debugging**: Traced data pipeline issues across 441 tables instantly
    - **Iteration speed**: V2→V5 feature engineering in 25 minutes (Feb 15)
    - **Architecture pivots**: 4-tier → 3-tier redesign in one session
    - **Dashboard**: 8 interactive pages with Plotly charts in hours, not days
    """)

show_cortex_badge()
