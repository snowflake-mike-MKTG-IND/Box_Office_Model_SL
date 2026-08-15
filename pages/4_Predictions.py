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
    {"title": "The End of Oak Street", "release": "2026-08-14", "point": 24.2, "bayes": 23.6,
     "lo": 15.8, "upside": 40.6, "p_large": 0.0, "tier": "MID",
     "note": "V31 V3 @ D-3 (final pre-release score). Warner Bros./Bad Robot original sci-fi thriller (David Robert Mitchell), retitled from Flowervale Street, starring Anne Hathaway and Ewan McGregor; PG-13. Best-estimate MID $24.2M, risk-adjusted $23.6M, 50% HDR $15.8M-$40.6M. Google Trends velocity/acceleration in the 97th-98th pctile and Wikipedia peaking ~51K/day let the star/studio pedigree re-enter via the model's gated interactions. Still MID against an $85M production budget."},
    {"title": "Fall 2: Deadpoint", "release": "2026-09-02", "point": 15.5, "bayes": 15.5,
     "lo": 8.0, "upside": 39.1, "p_large": 0.0, "tier": "MID",
     "note": "V31 V3 @ D-21 (early forecast). Lionsgate survival-thriller sequel to Fall (2022, $2.5M OW). Corrected this cycle: the cast was fixed to the real leads (Harriet Slater, Arsema Thomas, Tom Brittney — a wrong cast had inflated star power), genre set to Thriller, and the franchise/predecessor wired in. Best-estimate MID $15.5M, risk-adjusted $15.5M, 50% HDR $8.0M-$39.1M. Two trailers (Lionsgate), 4,635 comments."},
    {"title": "By Any Means", "release": "2026-09-04", "point": 13.8, "bayes": 13.8,
     "lo": 6.7, "upside": 41.9, "p_large": 0.0, "tier": "SMALL",
     "note": "V31 V3 @ D-21 (early forecast). Paramount / Thunder Road action-thriller (Mark Wahlberg, Yahya Abdul-Mateen II). Best-estimate SMALL $13.8M, risk-adjusted $13.8M, 50% HDR $6.7M-$41.9M. Eased from an earlier MID read once YouTube viewing-intent (net-negative) was folded into the feature set."},
    {"title": "Insidious: Out of the Further", "release": "2026-08-21", "point": 12.0, "bayes": 12.4,
     "lo": 7.1, "upside": 22.8, "p_large": 0.0, "tier": "SMALL",
     "note": "V31 V3 @ D-7. Sony/Screen Gems Insidious sequel (predecessor The Red Door opened $33.0M). Franchise-level reach — comment volume, Wikipedia at parity — but a cooler qualitative reaction: trailer sentiment is net-neutral-to-negative and 'pass' intent runs higher than the prior film. Studio/predecessor pedigree corrected this cycle. Best-estimate SMALL $12.0M, 50% HDR $7.1M-$22.8M."},
    {"title": "Mutiny", "release": "2026-08-21", "point": 11.4, "bayes": 11.4,
     "lo": 7.7, "upside": 19.2, "p_large": 0.0, "tier": "SMALL",
     "note": "V31 V3 @ D-7. Lionsgate action (Jason Statham). Best-estimate SMALL $11.4M, risk-adjusted $11.4M, 50% HDR $7.7M-$19.2M. Up from $9.2M after correcting the studio-tier flag (Lionsgate is a major, not a small studio) — the erroneous flop-damper was suppressing the point."},
    {"title": "The Dog Stars", "release": "2026-08-28", "point": 10.8, "bayes": 10.8,
     "lo": 6.9, "upside": 18.6, "p_large": 0.0, "tier": "SMALL",
     "note": "V31 V3 @ D-14. 20th Century sci-fi survival drama (Josh Brolin, Jacob Elordi), adapted from Peter Heller's novel. Best-estimate SMALL $10.8M, 50% HDR $6.9M-$18.6M. Stepped down from an earlier MID read once YouTube viewing-intent was aggregated in — the true net-negative signal replaced a previously-missing (optimistic) zero."},
    {"title": "Onslaught", "release": "2026-09-04", "point": 10.0, "bayes": 10.4,
     "lo": 5.6, "upside": 20.7, "p_large": 0.0, "tier": "SMALL",
     "note": "V31 V3 @ D-21 (early forecast). A24 action-horror (Adria Arjona, Dan Stevens). Heavy 'pass' intent in trailer comments (net −22) → SMALL $10.0M, 50% HDR $5.6M-$20.7M."},
    {"title": "The Magic Faraway Tree", "release": "2026-08-21", "point": 5.1, "bayes": 5.1,
     "lo": 3.3, "upside": 8.8, "p_large": 0.0, "tier": "SMALL",
     "note": "V31 V3 @ D-7. StudioCanal family fantasy from Enid Blyton's book series (recognized this cycle as tier-2 IP). Thin trailer engagement plus net-negative viewing-intent → SMALL $5.1M, 50% HDR $3.3M-$8.8M."},
    {"title": "Finding Emily", "release": "2026-08-28", "point": 4.8, "bayes": 4.6,
     "lo": 3.1, "upside": 7.9, "p_large": 0.0, "tier": "SMALL",
     "note": "V31 V3 @ D-14. Focus Features / Working Title romance. Bottom-decile demand → SMALL $4.8M, 50% HDR $3.1M-$7.9M."},
    {"title": "Hot Spot", "release": "2026-08-21", "point": 3.2, "bayes": 3.2,
     "lo": 1.6, "upside": 8.0, "p_large": 0.0, "tier": "SMALL",
     "note": "V31 V3 @ D-7. Focus Features European sci-fi thriller (Noomi Rapace). Near-zero pre-release Google Trends demand → SMALL $3.2M, 50% HDR $1.6M-$8.0M."},
]

# Recently released — prediction of record (verbatim, no hindsight) vs actual domestic OW
RELEASED = [
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
