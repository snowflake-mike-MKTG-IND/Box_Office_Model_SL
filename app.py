"""V23b Opening Weekend Prediction Model — Home / navigation hub."""
import streamlit as st

from theme import (
    SF_BLUE,
    apply_page_config,
    freshness_caption,
    kpi_row,
    page_header,
    section,
    show_cortex_badge,
)

apply_page_config("V23b OW Prediction Model", icon="🎬")

page_header(
    "V23b Opening Weekend Prediction Model",
    "Horror-first 2-bucket routing · Snowflake Model Registry · Feature Store · Dual Rule C · Rule D",
)

# -- Cortex Code velocity hero ----------------------------------------------
st.markdown(
    f"""
    <div style="background: linear-gradient(135deg, {SF_BLUE} 0%, #11567F 100%);
                padding: 1.75rem 2rem; border-radius: 14px; color: white;
                margin-bottom: 1.5rem;">
      <div style="font-size: 0.82rem; letter-spacing: 0.08em; text-transform: uppercase;
                  opacity: 0.85; margin-bottom: 0.35rem;">
        ❄️ Built with Cortex Code
      </div>
      <div style="font-size: 1.55rem; font-weight: 700; line-height: 1.25;
                  margin-bottom: 0.4rem;">
        One person. One AI. ~50 hours of active work.
      </div>
      <div style="font-size: 0.98rem; opacity: 0.95; line-height: 1.45; max-width: 860px;">
        End-to-end ML product — data engineering, 24 model versions, 120+ experiments,
        140+ HP configs, 750+ Snowflake artifacts, and this 9-page dashboard — shipped in
        <b>~1 week of working hours</b>. The V18 Wikipedia sprint took <b>49 minutes</b>.
        V23b horror routing + Snowflake ML deployment took <b>one working session</b>.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Velocity stats
kpi_row([
    ("Active work",    "~50h",  "vs ~300h traditional"),
    ("Calendar span",  "15 sessions", "over 90 days"),
    ("Team size",      "1 + AI", "vs 2-3 engineers"),
    ("V22c → V23b","1 session", "Horror routing + SF Model Registry"),
])

st.caption(
    "Time savings vs traditional baseline: **~6× faster** · **~85% fewer human hours** · "
    "no procurement, no handoffs, no vendor integration delays."
)

# -- Model performance hero --------------------------------------------------
section("Model performance")
kpi_row([
    ("Training films",     "282",     "413 with GT data"),
    ("Features",           "72",      "+13 Wikipedia"),
    ("V23b CV MAE (-7d)",  "$8.74M",  "80.1% tier acc"),
    ("Horror accuracy",    "69.2%",   "breakout detection"),
])
freshness_caption("5-fold GroupKFold CV on 282 films · V23b: Horror-first routing, deployed to Snowflake Model Registry", "2026-05-27")

# -- Navigation grid ---------------------------------------------------------
section("Explore the model", "Pick a section below. Each page owns one topic.")

NAV = [
    ("Architecture", "V23b: Horror-first routing + 3-tier non-horror cascade + Rule C/D.",
     "pages/1_Architecture.py"),
    ("Features", "72 features (V23b inherits V18 feature set + D-21 horizon).",
     "pages/2_Features.py"),
    ("Performance", "V23b CV results ($8.74M MAE, 80.1% acc) with horror fix.",
     "pages/3_Performance.py"),
    ("Predict", "Cascade simulator — V23b logic (Snowflake Model Registry).",
     "pages/4_Predictions.py"),
    ("Errors", "Horror underprediction fix and Rule C/D corrections.",
     "pages/5_Errors.py"),
    ("Recent Predictions", "Live tracking of predictions vs actual weekends.",
     "pages/7_Recent_Predictions.py"),
    ("Model History", "Version-by-version evolution, HPT, velocity.",
     "pages/6_Timeline.py"),
    ("Development Story", "The 49-minute Wikipedia sprint with Cortex Code.",
     "pages/8_Wikipedia_Integration.py"),
    ("V20→V23b Model Story", "V20-Clip → V22c Hybrid → V23b Horror Routing + Snowflake ML.",
     "pages/9_V20_Model_Story.py"),
]

cols = st.columns(4)
for i, (title, desc, page) in enumerate(NAV):
    with cols[i % 4]:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(desc)
            st.page_link(page, label="Open")

# -- What's new --------------------------------------------------------------
section("What's new in V23b")
st.markdown(
    "- **Horror-first routing** — Step 1 classifies horror vs non-horror. Horror movies get "
    "dedicated 2-bucket regressors (Small/Large split at $17M via log-space KMeans). Fixes "
    "Obsession ($17.2M actual vs $9.2M V22c prediction) and Backrooms underprediction.\n"
    "- **Snowflake Model Registry** — Full ML pipeline deployed: Feature Store → CustomModel → "
    "batch inference via `mv.run()`. Model: `SPARK_PAR_DEMO.ML_PIPELINE.OW_PREDICTION_V23B/v3`.\n"
    "- **PRODUCTION data fix** — Live feature view was reading from STAGING (197 movies) instead of "
    "PRODUCTION (413 movies). 67% of training films had zero GT features. Single biggest accuracy gain.\n"
    "- **Weekend 20 results** — Mandalorian $81.7M (pred $67.4M, tier correct), Passenger $8.7M "
    "(pred $6.9M, tier correct), I Love Boosters $3.8M (pred $3.1M, tier correct). 3/3 tier accuracy.\n"
)

section("Previously")
st.markdown(
    "- **V22c** — Hybrid Original-IP Blend + Dual Rule C + Rule D. Fixed MK2 false positive.\n"
    "- **V21** — Rule D (Static Tentpole Gate) for early tentpole detection at D-18+.\n"
    "- **V18** — +13 Wikipedia pageview features lifted CV accuracy by 5.5pp to 77.2%.\n"
    "- **49-minute Wikipedia sprint** — see [Development Story](./8_📚_Wikipedia_Integration).\n"
)

show_cortex_badge()
