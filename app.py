"""V28-B Opening Weekend Prediction Model — Home / navigation hub."""
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

apply_page_config("V28-B OW Prediction Model", icon="🎬")

page_header(
    "V28-B Opening Weekend Prediction Model",
    "Horizon-normalized demand classification · CatBoost multiclass + per-tier regressors · Snowflake Model Registry + Feature Store",
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
        End-to-end ML product — data engineering, 28 model versions, 150+ experiments,
        140+ HP configs, 750+ Snowflake artifacts, and this 10-page dashboard — shipped in
        <b>~1 week of working hours</b>. The V18 Wikipedia sprint took <b>49 minutes</b>.
        The V28-A→V28-B horizon normalization took <b>one working session</b>.
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
    ("V28-A → V28-B","1 session", "Horizon-normalized demand + SF Model Registry"),
])

st.caption(
    "Time savings vs traditional baseline: **~6× faster** · **~85% fewer human hours** · "
    "no procurement, no handoffs, no vendor integration delays."
)

# -- Model performance hero --------------------------------------------------
section("Model performance")
kpi_row([
    ("Training films",       "310",     "× 3 horizons = 928 rows"),
    ("V28-B CV overall",     "76.6%",   "D-14: 75.2% · D-7: 77.0% · D-3: 77.7%"),
    ("Leak-safe backtest",   "75.3%",   "$10.96M · 288 incl. recent breakouts"),
    ("Breakout odds",        "calibrated", ">50% → 87% actual LARGE+"),
])
freshness_caption("5-fold GroupKFold CV (grouped by film) · V28-B: horizon-normalized demand classification, deployed to Snowflake Model Registry (OW_PREDICTION_V28B)", "2026-07-03")

# -- Navigation grid ---------------------------------------------------------
section("Explore the model", "Pick a section below. Each page owns one topic.")

NAV = [
    ("Architecture", "V28-B: horizon-normalized demand classification, CatBoost multiclass + per-tier regressors.",
     "pages/1_Architecture.py"),
    ("Features", "52 classifier features (36 static + 6 GT percentiles + 6 Wiki percentiles + 3 interactions + DAYS_OUT).",
     "pages/2_Features.py"),
    ("Performance", "V28-B CV: 76.6% overall accuracy across all horizons.",
     "pages/3_Performance.py"),
    ("Predict", "Classifier simulator — base logic (live V28-B runs in Snowflake).",
     "pages/4_Predictions.py"),
    ("Errors", "Where the model misses — breakouts and the measured noise floor.",
     "pages/5_Errors.py"),
    ("Recent Predictions", "Live tracking of predictions vs actual weekends.",
     "pages/7_Recent_Predictions.py"),
    ("Model History", "Version-by-version evolution, HPT, velocity.",
     "pages/6_Timeline.py"),
    ("Development Story", "The 49-minute Wikipedia sprint with Cortex Code.",
     "pages/8_Wikipedia_Integration.py"),
    ("V28-A Model Story", "Rule-free learned meta-combiner — the prior production model.",
     "pages/11_V28A_Model_Story.py"),
]

cols = st.columns(4)
for i, (title, desc, page) in enumerate(NAV):
    with cols[i % 4]:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(desc)
            st.page_link(page, label="Open")

# -- What's new --------------------------------------------------------------
section("What's new in V28-B")
st.markdown(
    "- **Horizon-normalized demand classification** — Google Trends and Wikipedia demand features "
    "are converted to horizon-relative percentiles before the classifier sees them. A film with strong "
    "D-14 demand is correctly recognized as strong *for that stage*, rather than being compared against "
    "D-7 training baselines.\n"
    "- **Multi-horizon training** — classifier trained on 928 rows (310 films × 3 horizons: D-14, D-7, D-3). "
    "CV accuracy: D-14: 75.2%, D-7: 77.0%, D-3: 77.7%, Overall: 76.6%.\n"
    "- **Range-clip** — point estimate is clamped within [bear, bull] quantile bounds, preventing "
    "impossible prediction-outside-range errors from the global/mixture blend.\n"
    "- **Snowflake Model Registry** — deployed as `SPARK_PAR_DEMO.ML_PIPELINE.OW_PREDICTION_V28B` (V1).\n"
)

section("Previously")
st.markdown(
    "- **V28-A** — rule-free learned meta-combiner: CatBoost + TabPFN soft-vote, learned combiner g. No hand rules.\n"
    "- **V27** — modern ensemble: tuned CatBoost + TabPFN soft-vote, no hand rules.\n"
    "- **V25** — demand-driven classifier; Google Trends moved into tier assignment so budget no longer dominates.\n"
    "- **V23b** — horror-first 2-bucket routing; fixed Obsession / Backrooms underprediction.\n"
    "- **V18** — +13 Wikipedia pageview features lifted CV accuracy by 5.5pp to 77.2%.\n"
)

show_cortex_badge()
