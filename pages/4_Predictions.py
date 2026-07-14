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
    {"title": "The Odyssey", "release": "2026-07-17", "point": 99.2, "bayes": 76.0,
     "lo": 53.1, "upside": 129.6, "p_large": 0.61, "tier": "LARGE+",
     "note": "V30 @ D-3 final. 88th-pctile GT + 98th-pctile Wiki demand. RF flag confident (61%) → Track B lift from $79.2M HDR50 to $99.2M point. Nolan's Homer adaptation with Damon/Zendaya/Hathaway; predecessor Oppenheimer $82.5M."},
    {"title": "Spider-Man: Brand New Day", "release": "2026-07-31", "point": 149.6, "bayes": 95.4,
     "lo": 41.9, "upside": 233.2, "p_large": 0.54, "tier": "LARGE+",
     "note": "Mega-tentpole with a very wide, right-skewed distribution. The confident large-film flag lifts the point toward the demand-implied ceiling; risk-adjusted read ~$95M, upside ~$233M. Re-score at a tighter horizon to sharpen."},
    {"title": "Evil Dead Burn", "release": "2026-07-10", "point": 13.2, "bayes": 12.4,
     "lo": 8.9, "upside": 21.4, "p_large": 0.17, "tier": "SMALL",
     "note": "ACTUAL: $13.7M (SMALL) — near-perfect call. Point $13.2M vs actual $13.7M = $0.5M error. Tight SMALL call confirmed by low pre-release demand."},
]

section("Live examples (OW_PREDICTION_V30, latest scores)")
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
st.caption("Live predictions are written to SPARK_PAR_DEMO.PRODUCTION.OW_PREDICTIONS_V21 (MODEL_VERSION='V30').")
show_cortex_badge()
