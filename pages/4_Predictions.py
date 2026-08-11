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
    {"title": "Spider-Man: Brand New Day", "release": "2026-07-31", "point": 177.1, "bayes": 137.4,
     "lo": 101.1, "upside": 240.6, "p_large": 1.0, "tier": "LARGE+",
     "note": "V31 V3 @ D-3 (final pre-release score, 07-28, at the true 3-day horizon on fresh Google Trends thru 07-28 + Wikipedia thru 07-27). Two data corrections lifted this read: the Wikipedia article ramp captured to release week (peak ~105K/day, R7 91st pctile) and the lead cast corrected to the confirmed top-5 billed — Zendaya had been missing, raising top-line star power. All demand signals top-decile (GT 99th, Wiki ~90th, YouTube 52.7K comments). Best-estimate $177.1M, risk-adjusted $137.4M, 50% HDR $101.1M-$240.6M; demand-forward flag certain (100%). Up from the $150.2M D-7 read as demand tightens into release."},
    {"title": "The End of Oak Street", "release": "2026-08-14", "point": 24.2, "bayes": 23.6,
     "lo": 15.8, "upside": 40.6, "p_large": 0.0, "tier": "MID",
     "note": "V31 V3 @ D-3 (final pre-release score, 08-11, at the true 3-day horizon on Google Trends refreshed thru 08-11 + Wikipedia thru 08-10). Warner Bros./Bad Robot original sci-fi thriller (David Robert Mitchell), retitled from Flowervale Street, starring Anne Hathaway and Ewan McGregor; PG-13. Best-estimate MID $24.2M, risk-adjusted $23.6M, 50% HDR $15.8M-$40.6M. Up from the $17.0M D-7 read: refreshed demand lifted the point as Google Trends velocity/acceleration reached the 97th-98th pctile and Wikipedia peaked ~51K/day, letting the star/studio pedigree re-enter via the model's gated interactions. Still MID against an $85M production budget \u2014 meaningful downside risk."},
    {"title": "Super Troopers 3", "release": "2026-08-07", "point": 7.2, "bayes": 6.8,
     "lo": 5.1, "upside": 11.9, "p_large": 0.0, "tier": "SMALL",
     "note": "V31 V3 @ D-3 (final pre-release score, 08-07). Broken Lizard comedy sequel; predecessor Super Troopers 2 opened to $15.2M. Pre-release demand stays bottom-decile (Google Trends ~9th pctile) and the real Broken Lizard cast carries no lifetime box office → SMALL; best-estimate $7.2M, 50% HDR $5.1M-$11.9M. Essentially flat against the $6.4M D-7 read."},
    {"title": "One Night Only", "release": "2026-08-07", "point": 7.5, "bayes": 6.9,
     "lo": 5.3, "upside": 12.3, "p_large": 0.0, "tier": "SMALL",
     "note": "V31 V3 @ D-3 (final pre-release score, 08-07). Universal original rom-com (Will Gluck, Barbaro/Turner). Modest pre-release demand (Google Trends ~25th pctile) → SMALL; best-estimate $7.5M, 50% HDR $5.3M-$12.3M. Firmed from the $6.3M D-7 read."},
    {"title": "The Dog Stars", "release": "2026-08-28", "point": 16.2, "bayes": 15.7,
     "lo": 11.2, "upside": 27.6, "p_large": 0.0, "tier": "MID",
     "note": "V31 V3 @ D-21 (early forecast, 08-07). Ridley Scott sci-fi survival drama adapted from Peter Heller's novel. Mid-tier early demand (Google Trends ~50th pctile, Wikipedia ~64th, peak ~14.3K/day) → MID; best-estimate $16.2M, 50% HDR $11.2M-$27.6M. First read; refines at D-14/-7."},
    {"title": "Finding Emily", "release": "2026-08-28", "point": 3.4, "bayes": 3.4,
     "lo": 1.7, "upside": 10.3, "p_large": 0.0, "tier": "SMALL",
     "note": "V31 V3 @ D-21 (early forecast, 08-07). Bottom-decile early demand (Google Trends ~6th pctile, Wikipedia ~10th) → SMALL; best-estimate $3.4M, 50% HDR $1.7M-$10.3M. First read."},
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
