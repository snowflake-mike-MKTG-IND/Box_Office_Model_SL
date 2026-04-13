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
col1.metric("Development Time", "~35.5 hrs", "8 active sessions")
col2.metric("Snowflake Artifacts", "664", "Tables + Views + Procs")
col3.metric("Model Iterations", "10 versions", "76 experiments + 5 override rules")
col4.metric("HP Configurations", "104", "Grid + random search")

st.info(
    "**One person, one AI, one week of active work.** What would traditionally require a data engineer, "
    "ML engineer, and dashboard developer over 4-6 weeks was completed in ~35.5 hours of "
    "active development across 8 sessions — including the V16 TMDB override system, which went from "
    "hypothesis to validated production deployment in a single session."
)

st.divider()

st.header("What Cortex Code Did")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Engineering")
    st.markdown("""
    - Built 441 tables, 152 views, 71 stored procedures in Snowflake
    - Wrote and debugged complex SQL for Google Trends normalization (scaled against Movie Showtimes baseline)
    - Built YouTube sentiment pipeline and external API integrations
    - Managed canonical movie ID mapping — the single source of truth for all movie IDs
    - Integrated TMDB daily popularity exports for V16 override system
    """)

    st.subheader("Feature Engineering")
    st.markdown("""
    - Discovered and implemented 56 features across 9 categories
    - Created rolling window calculations (3D, 5D, 7D) for Google Trends data
    - Built star power metrics from actor box office history (4,588 actors indexed)
    - Engineered interaction features (Trends x IP, Star x Rolling, Sentiment x Trends)
    - V16: Added IS_MAJOR_STUDIO, TMDB daily popularity features
    """)

with col2:
    st.subheader("Model Development")
    st.markdown("""
    - Iterated through 10 model versions and 76 experiments
    - Managed CatBoost hyperparameter tuning (104 HP configurations)
    - Implemented 3-tier cascade architecture with tier-specific loss functions
    - Handled cross-validation, train/test splits, and model evaluation
    - Designed and validated TMDB override system (5 rule variants tested)
    """)

    st.subheader("Pipeline & Deployment")
    st.markdown("""
    - Built prediction scripts for weekly production runs
    - Created automated data fetching and enrichment pipelines
    - Deployed models to Snowflake stages with stored procedures
    - Built this entire 8-page Streamlit dashboard
    - V16: Full production deployment including SP, model file, dashboard update
    """)

st.divider()

st.header("V16 Override: One-Session Iteration")
st.markdown(
    "The V16 TMDB override development is a case study in **Cortex Code velocity**. "
    "The entire workflow — from identifying the signal to validated production deployment — "
    "happened in a single ~3 hour Cortex Code session on April 10, 2026."
)

col1, col2 = st.columns([2, 1])

with col1:
    v16_steps = pd.DataFrame({
        "Step": [
            "1. Signal Discovery",
            "2. Rule Design",
            "3. Holdout Validation",
            "4. Rule Selection",
            "5. Production Bake",
            "6. SP Creation",
            "7. Dashboard Update",
        ],
        "What Cortex Code Did": [
            "Analyzed TMDB D14 vs actual OW: found Spearman r=0.817 across 22 films with data",
            "Designed 5 rule variants (A-D + baseline) with different thresholds and momentum gates",
            "Built proper holdout test: 266 train / 19 test split, tested all 5 rules blind",
            "Compared results: Rule C wins (84.2%, 4/4 correct, 0 wrong, momentum gate filters Primate)",
            "Updated V16 model joblib with override config, uploaded to Snowflake stage",
            "Created PREDICT_MOVIE_V16 stored procedure with override logic built in",
            "Updated all 8 dashboard pages to reflect V16 as production model",
        ],
        "Output": [
            "Correlation analysis",
            "test_override_rules.py",
            "5-way comparison table",
            "Rule C selected",
            "ow_pipeline_v16_production.joblib.gz",
            "PREDICT_MOVIE_V16(NUMBER)",
            "8 updated Streamlit pages",
        ]
    })

    st.dataframe(v16_steps, use_container_width=True, hide_index=True)

with col2:
    st.metric("Session Duration", "~3 hours", "One Cortex Code session")
    st.metric("Rules Tested", "5", "On proper holdout")
    st.metric("Holdout Improvement", "+21pp", "63.2% -> 84.2%")
    st.metric("Override Precision", "4/4", "100% correct")
    st.metric("Artifacts Updated", "8+", "Model + SP + Dashboard")

st.markdown(
    "**The key insight**: TMDB daily popularity data exists for only ~30 training films — too sparse "
    "for CatBoost to learn meaningful splits. But Cortex Code identified that the raw signal is "
    "extremely strong, designed a rule-based override that operates orthogonally to the model, "
    "tested it on a proper holdout, and deployed it to production — all without changing the base "
    "model architecture. This kind of rapid hypothesis-to-production iteration is what makes "
    "Cortex Code a force multiplier."
)

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
            "V16 Override (hypothesis to prod)",
        ],
        "With Cortex Code": [
            "8 active days over 73 calendar days",
            "~35.5 hours",
            "1 person",
            "664 (441T + 152V + 71P)",
            "76 + 104 HP + 5 override rules",
            "8 pages",
            "5 production models",
            "~3 hours (one session)",
        ],
        "Traditional Estimate": [
            "4-6 weeks",
            "200-300 hours",
            "2-3 engineers",
            "Similar scope",
            "Similar scope",
            "Similar scope",
            "Similar scope",
            "1-2 weeks (design, review, test, deploy)",
        ],
    }
    st.dataframe(
        pd.DataFrame(comparison_data),
        use_container_width=True,
        hide_index=True,
    )

with col2:
    st.success(
        "**One person. One AI. One week of active work.**\n\n"
        "What would traditionally require a data engineer, ML engineer, and "
        "dashboard developer over 4-6 weeks was completed in ~35.5 hours of "
        "active development across 8 sessions.\n\n"
        "Cortex Code didn't just write code — it managed the full ML lifecycle: "
        "data exploration, hypothesis generation, experiment management, "
        "debugging, pipeline orchestration, and visualization."
    )

    st.markdown("""
    **Key acceleration areas:**
    - **SQL generation**: Complex joins, window functions, CTEs written in seconds
    - **Debugging**: Traced data pipeline issues across 441 tables instantly
    - **Iteration speed**: V2->V5 feature engineering in 25 minutes (Feb 15)
    - **Architecture pivots**: 4-tier -> 3-tier redesign in one session
    - **V16 override**: Hypothesis -> holdout validation -> production in 3 hours
    - **Dashboard**: 8 interactive pages with Plotly charts in hours, not days
    """)

show_cortex_badge()
