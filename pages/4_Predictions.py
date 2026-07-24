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
     "note": "ACTUAL: $123.5M (LARGE+) — correct tier call. V30 @ D-3 point $99.2M vs actual $123.5M; the actual landed inside the model's floor–upside band ($53.1M–$129.6M), just under the $129.6M ceiling. 88th-pctile GT + 98th-pctile Wiki demand; RF flag confident (61%). Nolan's Homer adaptation (Damon/Zendaya/Hathaway), $250M budget, 3,919 theaters; $199.4M domestic total to date."},
    {"title": "Spider-Man: Brand New Day", "release": "2026-07-31", "point": 133.0, "bayes": 104.0,
     "lo": 76.9, "upside": 182.3, "p_large": 1.0, "tier": "LARGE+",
     "note": "V31 @ D-7 (re-scored 07-24 at the true 7-day horizon on fresh Google Trends thru 07-24 + refreshed Wikipedia). All demand signals top-decile (GT 98th, Wiki 91st-95th, YouTube). Best-estimate $133.0M, risk-adjusted $104M, 50% HDR $76.9M-$182.3M; demand-forward flag certain (100%). Up from the $114.9M D-14 read after the wiki refresh restored its rolling-demand percentile."},
    {"title": "Evil Dead Burn", "release": "2026-07-10", "point": 18.3, "bayes": 18.3,
     "lo": 18.3, "upside": 20.5, "p_large": 0.05, "tier": "MID",
     "note": "Prediction of record: V28-B @ D-3 — the production model live when the film opened 07-10 (V30/V31 did not exist yet). Called MID $18.3M; ACTUAL $13.7M (SMALL) — a $4.6M overshoot. Shown as the real-time pre-release call, not a hindsight re-score. This D-3 miss on a low-demand horror title is exactly what motivated the V30 flop-safety rework."},
    {"title": "Super Troopers 3", "release": "2026-08-07", "point": 6.9, "bayes": 6.4,
     "lo": 4.9, "upside": 11.3, "p_large": 0.0, "tier": "SMALL",
     "note": "V31 @ D-14 (re-scored 07-24 on refreshed trends + Wikipedia). Broken Lizard comedy sequel (Searchlight); predecessor Super Troopers 2 opened to $15.2M. Low pre-release demand at D-14 → SMALL; best-estimate $6.9M, 50% HDR $4.9M-$11.3M."},
    {"title": "One Night Only", "release": "2026-08-07", "point": 6.1, "bayes": 5.7,
     "lo": 4.4, "upside": 9.9, "p_large": 0.0, "tier": "SMALL",
     "note": "V31 @ D-14 (re-scored 07-24 on refreshed trends + Wikipedia). Universal original rom-com (Will Gluck, Barbaro/Turner). Modest pre-release demand at D-14 → SMALL; best-estimate $6.1M, 50% HDR $4.4M-$9.9M."},
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
