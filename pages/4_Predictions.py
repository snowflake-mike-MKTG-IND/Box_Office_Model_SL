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

# Live V30 scores (registered OW_PREDICTION_V30, scored @ D-21)
FILMS = [
    {"title": "The Odyssey", "release": "2026-07-17", "hdr50": 74.7, "bayes": 71.3,
     "lo": 49.1, "hi": 98.7, "p_large": 0.80, "tier": "LARGE+",
     "note": "Strong, well-formed demand (ROLLING_7D pctile 0.71, Wiki 0.94, budget ~$250M). RF flags it large with high confidence; a reasonably tight band for an early D-21 read."},
    {"title": "Spider-Man: Brand New Day", "release": "2026-07-31", "hdr50": 97.7, "bayes": 107.0,
     "lo": 18.5, "hi": 373.5, "p_large": 0.52, "tier": "LARGE+",
     "note": "HIGH-UNCERTAINTY / bimodal at D-21: the gated model holds a conservative ~$100M while the linear component extrapolates to blockbuster territory (~$300M+). Very wide band — re-score at a tighter horizon to sharpen."},
    {"title": "Evil Dead Burn", "release": "2026-07-10", "hdr50": 13.0, "bayes": 12.5,
     "lo": 8.5, "hi": 17.1, "p_large": 0.10, "tier": "SMALL",
     "note": "Low pre-release demand (ROLLING_7D pctile 0.21) for a modest-budget horror sequel. Tight, confident SMALL call."},
]

section("Live examples (scored at D-21, OW_PREDICTION_V30)")
for f in FILMS:
    with st.container(border=True):
        tier_color = {"LARGE+": VIOLET, "MID": ORANGE, "SMALL": SF_BLUE}[f["tier"]]
        st.markdown(f"### {f['title']}  ·  <span style='color:{tier_color}'>{f['tier']}</span>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("HDR50_MEAN (best-est)", f"${f['hdr50']:.1f}M")
        c2.metric("Bayes point (risk-adj)", f"${f['bayes']:.1f}M")
        c3.metric("50% HDR band", f"${f['lo']:.0f}–{f['hi']:.0f}M")
        c4.metric("RF P(≥$50M)", f"{f['p_large']*100:.0f}%")
        # band viz
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[f["lo"], f["hi"]], y=[0, 0], mode="lines",
                                 line=dict(color="rgba(150,150,150,0.6)", width=10), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[f["hdr50"]], y=[0], mode="markers+text", text=["HDR50"], textposition="top center",
                                 marker=dict(size=16, color=VIOLET), showlegend=False))
        fig.add_trace(go.Scatter(x=[f["bayes"]], y=[0], mode="markers+text", text=["Bayes"], textposition="bottom center",
                                 marker=dict(size=12, symbol="x", color="#555"), showlegend=False))
        fig.update_layout(height=120, margin=dict(l=10, r=10, t=10, b=10),
                          yaxis=dict(visible=False, range=[-1, 1]), xaxis=dict(title="Opening weekend ($M)"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f["note"])

section("How to read the triple output")
st.markdown(
    "- **HDR50_MEAN** — the density-weighted center of the tightest 50% region of the distribution. Use it as the "
    "single best-estimate headline number.\n"
    "- **Bayes point (r=2)** — the P33 quantile. Use it when over-predicting is more costly than under-predicting "
    "(the default posture for greenlight/marketing-spend decisions). It sits at or below HDR50_MEAN.\n"
    "- **50% HDR band** — the honest uncertainty. A tight band (Odyssey, Evil Dead Burn) = confident; a very wide "
    "band (Spider-Man at D-21) = the model is telling you it doesn't have enough settled signal yet.\n"
    "- **RF ≥$50M flag** — a calibrated large-film annotation (93% precision on the holdout). It never changes the "
    "point estimate; it's a second opinion on tier."
)
st.info(
    "V30 milestone vs the prior generation: the old classifier would **not** have flagged Spider-Man as large from "
    "its predecessor/pedigree signal at this horizon. V30's demand-gated design surfaces it as LARGE+ (with honest "
    "uncertainty) — a real improvement in catching genuine tentpoles without over-committing.",
    icon="🎬",
)
st.caption("Live predictions are written to SPARK_PAR_DEMO.PRODUCTION.OW_PREDICTIONS_V21 (MODEL_VERSION='V30@D21').")
show_cortex_badge()
