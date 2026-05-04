"""V21 Opening Weekend Prediction Model — Home / navigation hub."""
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

apply_page_config("V21 OW Prediction Model", icon="🎬")

page_header(
    "V21 Opening Weekend Prediction Model",
    "V18.7 soft-mixture cascade · V20 adaptive quantile window · Rule D (Static Tentpole Gate) · guarded Rule C",
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
        One person. One AI. ~44 hours of active work.
      </div>
      <div style="font-size: 0.98rem; opacity: 0.95; line-height: 1.45; max-width: 860px;">
        End-to-end ML product — data engineering, 18 model versions, 83 experiments,
        108 HP configs, 664 Snowflake artifacts, and this 9-page dashboard — shipped in
        <b>~1 week of working hours</b>. The V18 Wikipedia sprint took <b>49 minutes</b>.
        The V18 → V20-Clip + Rule C jump (-17.4% MAE) took <b>one working session</b>.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Velocity stats
kpi_row([
    ("Active work",    "~44h",  "vs ~250h traditional"),
    ("Calendar span",  "12 sessions", "over 90 days"),
    ("Team size",      "1 + AI", "vs 2-3 engineers"),
    ("V18 → V21","2 sessions", "Rule D + D-21 horizon"),
])

st.caption(
    "Time savings vs traditional baseline: **~6× faster** · **~85% fewer human hours** · "
    "no procurement, no handoffs, no vendor integration delays."
)

# -- Model performance hero --------------------------------------------------
section("Model performance")
kpi_row([
    ("Training films",     "276",     None),
    ("Features",           "72",      "+13 Wikipedia"),
    ("V18 CV MAE (-7d)",   "$10.96M", "77.2% tier acc"),
    ("V21 CV MAE (-7d)",   "$9.58M",  "R² 0.814 + Rule D gate"),
])
freshness_caption("5-fold GroupKFold CV on 278 films · V21: V20-Clip + Rule D + Rule C", "2026-05-04")

# -- Navigation grid ---------------------------------------------------------
section("Explore the model", "Pick a section below. Each page owns one topic.")

NAV = [
    ("Architecture", "V21 cascade: soft mixture + quantile window + Rule D + Rule C.",
     "pages/1_Architecture.py"),
    ("Features", "72 features (V21 inherits V18 feature set + D-21 horizon).",
     "pages/2_Features.py"),
    ("Performance", "V21 CV results ($9.58M MAE, R² 0.814) vs V18 baseline.",
     "pages/3_Performance.py"),
    ("Predict", "Cascade simulator — V21 logic (live in Snowflake).",
     "pages/4_Predictions.py"),
    ("Errors", "V20 base cascade biggest misses and what Rule D + Rule C fixes.",
     "pages/5_Errors.py"),
    ("Recent Predictions", "Live tracking of predictions vs actual weekends.",
     "pages/7_Recent_Predictions.py"),
    ("Model History", "Version-by-version evolution, HPT, velocity.",
     "pages/6_Timeline.py"),
    ("Development Story", "The 49-minute Wikipedia sprint with Cortex Code.",
     "pages/8_Wikipedia_Integration.py"),
    ("V20/V21 Model Story", "V20-Clip + Rule C → V21 Rule D (Static Tentpole Gate).",
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
section("What's new in V21")
st.markdown(
    "- **Rule D (Static Tentpole Gate)** — identifies obvious tentpoles at D-18+ before TMDB data arrives. "
    "Gate: Budget≥$125M + IP≥3 + Star≥9 + Predecessor≥$108M + Major Studio. 100% precision in backtest.\n"
    "- **D-21 horizon extended** — features view now starts at D-21 (was D-15) for earlier predictions.\n"
    "- **Rule C enhanced** — now checks best available TMDB (D7 priority over D14).\n"
    "- **Prediction history preserved** — all model versions tracked in `OW_PREDICTIONS_V21` (append-only). "
    "See [Recent Predictions](./Recent_Predictions) for Mandalorian (Rule D) and MK2 (Rule C).\n"
    "- **Wikipedia fix** — Mandalorian article corrected: 10,644 views/day (was 12/day due to wrong article).\n"
)

section("Previously — V18 build")
st.markdown(
    "- **+13 Wikipedia pageview features** lift CV accuracy by **5.5pp** to **77.2%** — "
    "see [Performance](./3_📈_Performance) for the full breakdown.\n"
    "- **All 276 training labels validated** against The-Numbers.com on Apr 21, correcting "
    "4 fabricated OW values and 13 corrupted release dates.\n"
    "- **49-minute end-to-end build** — see [Development Story](./8_📚_Wikipedia_Integration) "
    "for the sprint timeline."
)

show_cortex_badge()
