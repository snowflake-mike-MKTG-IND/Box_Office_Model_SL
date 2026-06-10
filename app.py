"""V28-A Opening Weekend Prediction Model — Home / navigation hub."""
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

apply_page_config("V28-A OW Prediction Model", icon="🎬")

page_header(
    "V28-A Opening Weekend Prediction Model",
    "Rule-free learned meta-combiner · tuned CatBoost + TabPFN · calibrated breakout odds · Snowflake Model Registry + Feature Store",
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
        The V27→V28-A rule-free meta-combiner + Snowflake ML deployment took <b>one working session</b>.
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
    ("V27 → V28-A","1 session", "Rule-free meta-combiner + SF Model Registry"),
])

st.caption(
    "Time savings vs traditional baseline: **~6× faster** · **~85% fewer human hours** · "
    "no procurement, no handoffs, no vendor integration delays."
)

# -- Model performance hero --------------------------------------------------
section("Model performance")
kpi_row([
    ("Training films",       "291",     "413 with GT data"),
    ("V28-A CV (-7d)",       "77.7%",   "$9.99M MAE · 287 films"),
    ("Leak-safe backtest",   "75.3%",   "$10.96M · 288 incl. recent breakouts"),
    ("Breakout odds",        "calibrated", ">50% → 87% actual LARGE+"),
])
freshness_caption("Nested 5-fold GroupKFold CV · V28-A: rule-free learned meta-combiner, deployed to Snowflake Model Registry (OW_PREDICTION_V28)", "2026-06-08")

# -- Navigation grid ---------------------------------------------------------
section("Explore the model", "Pick a section below. Each page owns one topic.")

NAV = [
    ("Architecture", "V28-A: rule-free learned meta-combiner over CatBoost + TabPFN base.",
     "pages/1_Architecture.py"),
    ("Features", "72 features (V18 feature set + D-21 horizon) + learned meta-features.",
     "pages/2_Features.py"),
    ("Performance", "V28-A backtest: 75.3% acc, $10.96M MAE, calibrated breakout odds.",
     "pages/3_Performance.py"),
    ("Predict", "Cascade simulator — base logic (live V28-A runs in Snowflake).",
     "pages/4_Predictions.py"),
    ("Errors", "Where the model misses — breakouts and the measured noise floor.",
     "pages/5_Errors.py"),
    ("Recent Predictions", "Live tracking of predictions vs actual weekends.",
     "pages/7_Recent_Predictions.py"),
    ("Model History", "Version-by-version evolution, HPT, velocity.",
     "pages/6_Timeline.py"),
    ("Development Story", "The 49-minute Wikipedia sprint with Cortex Code.",
     "pages/8_Wikipedia_Integration.py"),
    ("V28-A Model Story", "Rule-free learned meta-combiner + calibrated breakout odds — the current model.",
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
section("What's new in V28-A")
st.markdown(
    "- **Rule-free learned meta-combiner** — instead of hand-coded rules, a small model learns how to "
    "combine the base classifier + per-tier regressors (FINAL = 0.7·g + 0.3·mixture). Same-basis nested-CV "
    "D-7 = **77.7% / $9.99M** (287 films).\n"
    "- **Calibrated breakout odds** — every film gets a bear/base/bull range plus P(LARGE+). Calibration "
    "holds: films flagged >50% broke out **87%** of the time; '~1 in 3' flags broke out ~39%.\n"
    "- **Honest noise floor** — on the latest data (incl. the 4 hardest recent breakouts) the leak-safe "
    "backtest is **75.3% / $10.96M**; LARGE+ now sits at the measured noise floor, so the model trades "
    "point-accuracy chasing for calibrated probability.\n"
    "- **Snowflake Model Registry** — deployed as `SPARK_PAR_DEMO.ML_PIPELINE.OW_PREDICTION_V28` (default).\n"
)

section("Previously")
st.markdown(
    "- **V27** — modern ensemble: tuned CatBoost (trees) + TabPFN (transformer) soft-vote, no hand rules.\n"
    "- **V25** — demand-driven classifier; Google Trends moved into tier assignment so budget no longer dominates.\n"
    "- **V23b** — horror-first 2-bucket routing; fixed Obsession / Backrooms underprediction.\n"
    "- **V18** — +13 Wikipedia pageview features lifted CV accuracy by 5.5pp to 77.2%.\n"
)

show_cortex_badge()
