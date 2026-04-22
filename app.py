"""V18 Opening Weekend Prediction Model — Home / navigation hub."""
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

apply_page_config("V18 OW Prediction Model", icon="🎬")

page_header(
    "V18 Opening Weekend Prediction Model",
    "3-Tier Cascade · Google Trends · Wikipedia pageviews",
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
        One person. One AI. ~41 hours of active work.
      </div>
      <div style="font-size: 0.98rem; opacity: 0.95; line-height: 1.45; max-width: 860px;">
        End-to-end ML product — data engineering, 14 model versions, 79 experiments,
        108 HP configs, 664 Snowflake artifacts, and this 8-page dashboard — shipped in
        <b>~1 week of working hours</b>. Traditionally a 4–6 week effort for a team of
        2–3 engineers. The final V18 Wikipedia feature sprint: <b>49 minutes</b>
        from idea to tuned production model.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Velocity stats
kpi_row([
    ("Active work",    "~41h",  "vs ~250h traditional"),
    ("Calendar span",  "10 sessions", "over 83 days"),
    ("Team size",      "1 + AI", "vs 2-3 engineers"),
    ("V18 Wiki sprint","49 min", "idea → prod"),
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
    ("CV Accuracy (-7d)",  "77.2%",   "+5.5pp vs V17.2"),
    ("CV MAE (-7d)",       "$10.96M", "-$0.71M vs V17.2"),
])
freshness_caption("5-fold GroupKFold CV on 276 validated films", "2026-04-21")

# -- Navigation grid ---------------------------------------------------------
section("Explore the model", "Pick a section below. Each page owns one topic.")

NAV = [
    ("Architecture", "3-tier cascade, classifier configs, Rule C override.",
     "pages/1_Architecture.py"),
    ("Features", "Top drivers across static, Trends, and Wikipedia signals.",
     "pages/2_Features.py"),
    ("Performance", "Accuracy, MAE, per-tier results, V18 vs prior versions.",
     "pages/3_Performance.py"),
    ("Predict", "Run the V18 model interactively on any film.",
     "pages/4_Predictions.py"),
    ("Errors", "Biggest misses and known limitations.",
     "pages/5_Errors.py"),
    ("Recent Predictions", "Live tracking of predictions vs actual weekends.",
     "pages/7_Recent_Predictions.py"),
    ("Model History", "Version-by-version evolution, HPT, velocity.",
     "pages/6_Timeline.py"),
    ("Development Story", "The 49-minute Wikipedia sprint with Cortex Code.",
     "pages/8_Wikipedia_Integration.py"),
]

cols = st.columns(4)
for i, (title, desc, page) in enumerate(NAV):
    with cols[i % 4]:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(desc)
            st.page_link(page, label="Open")

# -- What's new --------------------------------------------------------------
section("What's new in V18")
st.markdown(
    "- **+13 Wikipedia pageview features** lift CV accuracy by **5.5pp** to **77.2%** — "
    "see [Performance](./3_📈_Performance) for the full breakdown.\n"
    "- **All 276 training labels validated** against The-Numbers.com on Apr 21, correcting "
    "4 fabricated OW values and 13 corrupted release dates.\n"
    "- **49-minute end-to-end build** — see [Development Story](./8_📚_Wikipedia_Integration) "
    "for the sprint timeline."
)

show_cortex_badge()
