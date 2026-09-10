"""V30 Predictions — the triple output, with live examples."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from theme import SF_BLUE, VIOLET, ORANGE, apply_page_config, page_header, section, show_cortex_badge

apply_page_config("V30 · Predict", icon="🎯")
page_header(
    "V30 Predictions",
    "Every film gets a distribution, two points, and a confidence flag — not a single guess",
)

# Live V30 scores (registered OW_PREDICTION_V30, demand-quality gate + Track B point + Track C upside, scored @ D-21)
FILMS = [
    {"title": "Practical Magic 2", "release": "2026-09-11", "point": 26.7, "bayes": 27.0,
     "lo": 15.1, "upside": 53.7, "p_large": 0.0, "tier": "MID",
     "note": "V31 V4 @ D-3. Warner Bros. legacy sequel (predecessor Practical Magic 1998, $13.1M OW). Novel IP (tier-2). Sandra Bullock + Nicole Kidman returning. GT demand steady at 30-42nd percentile across horizons, Wikipedia 54-66th. Best-estimate MID $26.7M, risk-adjusted $27.0M, 50% HDR $15.1M-$53.7M."},
    {"title": "Resident Evil", "release": "2026-09-18", "point": 21.1, "bayes": 21.4,
     "lo": 13.2, "upside": 39.4, "p_large": 0.04, "tier": "MID",
     "note": "V31 V4 @ D-14. Sony/Constantin reboot of the video game franchise (tier-3 IP, $80M budget). No predecessor OW (reboot). 35,750 trailer comments (high engagement) but net-negative viewing-intent (-7.8%). Corrected demand features reduced from V3's inflated $30.9M. Best-estimate MID $21.1M, risk-adjusted $21.4M, 50% HDR $13.2M-$39.4M."},
    {"title": "Heart of the Beast", "release": "2026-09-25", "point": 10.5, "bayes": 10.7,
     "lo": 7.3, "upside": 16.4, "p_large": 0.0, "tier": "SMALL",
     "note": "V31 V4 @ D-21. Paramount action-adventure (Brad Pitt). Early-horizon call — 5th percentile GT demand, 38th percentile Wikipedia. Best-estimate SMALL $10.5M, risk-adjusted $10.7M, 50% HDR $7.3M-$16.4M."},
]

# Recently released — prediction of record (verbatim, no hindsight) vs actual domestic OW
RELEASED = [
    {"title": "By Any Means", "release": "2026-09-04", "point": 11.8, "actual": 7.5,
     "pred_tier": "SMALL", "actual_tier": "SMALL",
     "note": "Predicted $11.8M (SMALL) at D-3; opened to $7.5M (SMALL) on 2,903 screens. Tier called correctly, abs error $4.2M."},
    {"title": "Onslaught", "release": "2026-09-04", "point": 10.1, "actual": 1.9,
     "pred_tier": "SMALL", "actual_tier": "SMALL",
     "note": "Predicted $10.1M (SMALL) at D-3; opened to $1.9M (SMALL) on 1,918 screens. Tier called correctly, but over-predicted substantially (abs error $8.2M)."},
    {"title": "Fall 2: Deadpoint", "release": "2026-09-02", "point": 15.6, "actual": 2.3,
     "pred_tier": "MID", "actual_tier": "SMALL",
     "note": "Predicted $15.6M (MID) at D-3; opened to $2.3M (SMALL) on 1,704 screens. Tier miss — opened on fewer screens than expected for a wide release."},
    {"title": "Insidious: Out of the Further", "release": "2026-08-21", "point": 12.9, "actual": 25.1,
     "pred_tier": "SMALL", "actual_tier": "MID",
     "note": "Predicted $12.9M (SMALL) at D-3; opened to $25.1M (MID) on 3,303 screens. Tier miss — the Insidious franchise's brand loyalty outran the demand signals. Abs error $12.2M."},
    {"title": "The Dog Stars", "release": "2026-08-28", "point": 13.8, "actual": 7.7,
     "pred_tier": "SMALL", "actual_tier": "SMALL",
     "note": "Predicted $13.8M (SMALL) at D-3; opened to $7.7M (SMALL) on 3,330 screens. Tier called correctly, abs error $6.1M. Against a $110M budget this was a steep miss for the studio."},
    {"title": "Mutiny", "release": "2026-08-21", "point": 11.8, "actual": 7.5,
     "pred_tier": "SMALL", "actual_tier": "SMALL",
     "note": "Predicted $11.8M (SMALL) at D-3; opened to $7.5M (SMALL) on 2,703 screens. Tier called correctly."},
    {"title": "The End of Oak Street", "release": "2026-08-14", "point": 24.2, "actual": 21.0,
     "pred_tier": "MID", "actual_tier": "MID",
     "note": "Predicted $24.2M (MID) at D-3; opened to $21.0M (MID). Tier called correctly, abs error $3.2M."},
    {"title": "Spider-Man: Brand New Day", "release": "2026-07-31", "point": 177.1, "actual": 360.1,
     "pred_tier": "LARGE+", "actual_tier": "LARGE+",
     "note": "Predicted $177.1M (LARGE+) at D-3; opened to a record $360.1M. Tier called correctly; magnitude a large-film miss consistent with V31's known ceiling on demand-quiet giants."},
    {"title": "One Night Only", "release": "2026-08-07", "point": 7.5, "actual": 5.5,
     "pred_tier": "SMALL", "actual_tier": "SMALL",
     "note": "Predicted $7.5M (SMALL) at D-3; opened to $5.5M. Tier called correctly."},
    {"title": "Super Troopers 3", "release": "2026-08-07", "point": 7.2, "actual": 4.0,
     "pred_tier": "SMALL", "actual_tier": "SMALL",
     "note": "Predicted $7.2M (SMALL) at D-3; opened to $4.0M. Tier called correctly."},
]

section("Upcoming releases (latest V31 pre-release scores)")
for f in FILMS:
    with st.container(border=True):
        tier_color = {"LARGE+": VIOLET, "MID": ORANGE, "SMALL": SF_BLUE}[f["tier"]]
        st.markdown(f"### {f['title']}  ·  <span style='color:{tier_color}'>{f['tier']}</span>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best-estimate point", f"${f['point']:.1f}M")
        c2.metric("Risk-adjusted (Bayes)", f"${f['bayes']:.1f}M")
        c3.metric("Range (floor–upside)", f"${f['lo']:.0f}–{f['upside']:.0f}M")
        c4.metric("Demand-forward P(≥$50M)", f"{f['p_large']*100:.0f}%")
        # band viz
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[f["lo"], f["upside"]], y=[0, 0], mode="lines",
                                 line=dict(color="rgba(150,150,150,0.6)", width=10), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[f["point"]], y=[0], mode="markers+text", text=["point"], textposition="top center",
                                 marker=dict(size=16, color=VIOLET), showlegend=False))
        fig.add_trace(go.Scatter(x=[f["bayes"]], y=[0], mode="markers+text", text=["Bayes"], textposition="bottom center",
                                 marker=dict(size=12, symbol="x", color="#555"), showlegend=False))
        fig.update_layout(height=120, margin=dict(l=10, r=10, t=10, b=10),
                          yaxis=dict(visible=False, range=[-1, 1]), xaxis=dict(title="Opening weekend ($M)"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f["note"])

section("Recently released — prediction of record vs. actual")
for f in RELEASED:
    with st.container(border=True):
        hit = f["pred_tier"] == f["actual_tier"]
        tier_color = {"LARGE+": VIOLET, "MID": ORANGE, "SMALL": SF_BLUE}[f["actual_tier"]]
        st.markdown(f"### {f['title']}  ·  <span style='color:{tier_color}'>{f['actual_tier']}</span>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted (point)", f"${f['point']:.1f}M")
        c2.metric("Actual opening", f"${f['actual']:.1f}M", f"{f['actual'] - f['point']:+.1f}M vs pred")
        c3.metric("Tier call", "Hit" if hit else "Miss")
        st.caption(f["note"])

section("How to read the output")
st.markdown(
    "- **Best-estimate point** — the headline number. For most films it's the density-weighted centre of the tightest "
    "50% region (flop-safe). For films the model is **confident are large** (demand-forward flag ≥ 0.4), the point is "
    "lifted toward the demand-implied ceiling (**Track B**) — this is what fixed the historical under-prediction of "
    "event films like Michael, Project Hail Mary and Mandalorian without raising the flop over-prediction rate.\n"
    "- **Risk-adjusted (Bayes, r=2)** — the P33 quantile. Use it when over-predicting is costlier than under-predicting.\n"
    "- **Range (floor–upside)** — floor = 50% HDR lower bound; **upside = P78 demand-implied ceiling** (**Track C**), the "
    "honest high end for a confident large film.\n"
    "- **Demand-forward ≥$50M flag** — calibrated large-film annotation built on demand signals only (no pedigree): "
    "100% precision / 50% recall on the 2026 holdout."
)
st.info(
    "How V30 lifts confident large films safely: the demand-**quality** gate (net audience intent × demand) separates "
    "true event films (positive intent) from look-alike high-demand flops (negative intent), so the point can be raised "
    "for the former without lifting the latter — holding the flop over-prediction rate flat.",
    icon="🎬",
)
st.caption("Live predictions are written to SPARK_PAR_DEMO.PRODUCTION.OW_PREDICTIONS_V21 (MODEL_VERSION='V31@D3' / 'V31@D7' / 'V31@D14', latest row per film by SCORED_AT).")
show_cortex_badge()
