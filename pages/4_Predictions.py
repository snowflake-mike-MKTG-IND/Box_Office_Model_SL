"""Predictions — the live predictions of record, read from data/films.json.

data/films.json is published by Lomis (the production pipeline agent) after Gruney's review, the same file the
public Opening Weekend Intelligence app reads. This page never edits numbers by hand: the film notes below are
documentation, shown only while the published point they describe is still the prediction of record.
"""
import json
import re
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from theme import SF_BLUE, VIOLET, ORANGE, apply_page_config, page_header, section, show_cortex_badge

apply_page_config("Predictions", icon="🎯")
page_header(
    "Predictions",
    "Every film gets a distribution, two points, and a confidence flag — not a single guess",
)

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "films.json"
TIER_COLOR = {"LARGE+": VIOLET, "MID": ORANGE, "SMALL": SF_BLUE}


def norm(title):
    return re.sub(r"[^a-z0-9]", "", (title or "").lower())


# Hand-written notes, keyed by film: (the point they were written about in $M, the note).
UPCOMING_NOTES = {
    norm("Resident Evil"): (28.5, "V31 @ D-3. Sony/Constantin reboot of the video game franchise (tier-3 IP, $80M budget). 36,156 trailer comments with net-negative viewing-intent (-7.9%). GT demand rose into release week. Best-estimate MID $28.5M, risk-adjusted $28.4M, 50% HDR $19.4M-$48.0M."),
    norm("Digger"): (18.5, "V31 @ D-21. Warner Bros./Legendary black comedy ($125M budget, R). Early-horizon call — strong Wikipedia peak (31,797) but moderate GT. 6,724 trailer comments with positive net intent (+1.2%). Best-estimate MID $18.5M, risk-adjusted $17.7M, 50% HDR $12.8M-$30.9M."),
    norm("Other Mommy"): (14.9, "V31 @ D-21. Blumhouse/Universal horror (Jessica Chastain, $20M budget est., R). Novel adaptation (Josh Malerman). 5,704 trailer comments. Best-estimate SMALL $14.9M, risk-adjusted $15.1M, 50% HDR $8.6M-$29.5M."),
    norm("Verity"): (14.2, "V31 @ D-21. Amazon MGM Studios thriller (Colleen Hoover novel, $40M budget, R). 1,983 trailer comments. GT demand at 100th percentile (entity-crush likely inflating raw). Best-estimate SMALL $14.2M, risk-adjusted $14.4M, 50% HDR $8.8M-$25.4M."),
    norm("The Social Reckoning"): (10.1, "V31 @ D-21. Sony drama-thriller (Aaron Sorkin, sequel to The Social Network, $40M budget, R). 2,307 trailer comments. Predecessor OW $22.4M. Best-estimate SMALL $10.1M, risk-adjusted $9.5M, 50% HDR $7.1M-$16.6M."),
    norm("Heart of the Beast"): (9.6, "V31 @ D-14. Paramount survival thriller (Brad Pitt, $140M budget, PG-13). Original IP. Demand softened from D-21 ($10.5M) to D-14 ($9.6M) — GT and Wikipedia both below 30th percentile. Best-estimate SMALL $9.6M, risk-adjusted $9.9M, 50% HDR $5.9M-$17.1M."),
}
RELEASED_NOTES = {
    norm("Practical Magic 2"): (26.7, "Predicted $26.7M (MID) at D-3; opened to $30.0M (MID) on 4,146 screens. Tier called correctly, abs error $3.3M."),
    norm("By Any Means"): (11.8, "Predicted $11.8M (SMALL) at D-3; opened to $7.5M (SMALL) on 2,903 screens. Tier called correctly, abs error $4.2M."),
    norm("Onslaught"): (10.1, "Predicted $10.1M (SMALL) at D-3; opened to $1.9M (SMALL) on 1,918 screens. Tier called correctly, but over-predicted substantially (abs error $8.2M)."),
    norm("Fall 2: Deadpoint"): (15.6, "Predicted $15.6M (MID) at D-3; opened to $2.3M (SMALL) on 1,704 screens. Tier miss — opened on fewer screens than expected for a wide release."),
    norm("Insidious: Out of the Further"): (12.9, "Predicted $12.9M (SMALL) at D-3; opened to $25.1M (MID) on 3,303 screens. Tier miss — the Insidious franchise's brand loyalty outran the demand signals. Abs error $12.2M."),
    norm("The Dog Stars"): (13.8, "Predicted $13.8M (SMALL) at D-3; opened to $7.7M (SMALL) on 3,330 screens. Tier called correctly, abs error $6.1M. Against a $110M budget this was a steep miss for the studio."),
    norm("Mutiny"): (11.8, "Predicted $11.8M (SMALL) at D-3; opened to $7.5M (SMALL) on 2,703 screens. Tier called correctly."),
    norm("The End of Oak Street"): (24.2, "Predicted $24.2M (MID) at D-3; opened to $21.0M (MID). Tier called correctly, abs error $3.2M."),
    norm("Spider-Man: Brand New Day"): (177.1, "Predicted $177.1M (LARGE+) at D-3; opened to a record $360.1M. Tier called correctly; magnitude a large-film miss consistent with V31's known ceiling on demand-quiet giants."),
    norm("One Night Only"): (7.5, "Predicted $7.5M (SMALL) at D-3; opened to $5.5M. Tier called correctly."),
    norm("Super Troopers 3"): (7.2, "Predicted $7.2M (SMALL) at D-3; opened to $4.0M. Tier called correctly."),
}


def note_for(notes, d):
    """The hand-written note while it still describes this prediction; otherwise the published summary.
    Dollar signs are escaped: Streamlit markdown would read the text between two of them as math."""
    point, text = notes.get(norm(d["MOVIE_TITLE"]), (None, None))
    if not (text and abs(point - (d.get("PRED_OW_M") or 0)) < 0.06):
        text = d.get("SUMMARY_TEXT") or ""
    return text.replace("$", "\\$")


def live_row(details, mid, kind):
    return next((d for d in details if d["MOVIE_ID"] == mid and d["PREDICTION_TYPE"] == kind), None)


if not DATA_FILE.exists():
    st.warning("No published predictions yet (data/films.json is published by Lomis).")
    st.stop()
DATA = json.loads(DATA_FILE.read_text())
META = DATA.get("meta") or {}
if META.get("updated"):
    st.caption(f"Predictions updated {META['updated']} · published by Lomis after an independent review by Gruney"
               + (f" ({META['review_id']})" if META.get("review_id", "").startswith("R-") else ""))

section("Upcoming releases (predictions of record)")
upcoming = [d for u in DATA["upcoming"] for d in [live_row(DATA["details"], u["MOVIE_ID"], "UPCOMING")] if d]
if not upcoming:
    st.info("No upcoming films are scored right now.")
for d in upcoming:
    with st.container(border=True):
        tier = d["PRED_TIER"]
        st.markdown(f"### {d['MOVIE_TITLE']}  ·  <span style='color:{TIER_COLOR.get(tier, SF_BLUE)}'>{tier}</span>",
                    unsafe_allow_html=True)
        st.caption(f"Opens {d['RELEASE_DATE']} · {d['MODEL_VERSION']}")
        c1, c2, c3, c4 = st.columns(4)
        bayes = d.get("BAYES_OW_M")
        c1.metric("Best-estimate point", f"${d['PRED_OW_M']:.1f}M")
        c2.metric("Risk-adjusted (Bayes)", f"${bayes:.1f}M" if bayes is not None else "—")
        c3.metric("Range (floor–upside)", f"${d['BEAR_OW_M']:.0f}–{d['BULL_OW_M']:.0f}M")
        c4.metric("Demand-forward P(≥$50M)", f"{d['P_LARGE_PCT']:.0f}%")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[d["BEAR_OW_M"], d["BULL_OW_M"]], y=[0, 0], mode="lines",
                                 line=dict(color="rgba(150,150,150,0.6)", width=10), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[d["PRED_OW_M"]], y=[0], mode="markers+text", text=["point"], textposition="top center",
                                 marker=dict(size=16, color=VIOLET), showlegend=False))
        if bayes is not None:
            fig.add_trace(go.Scatter(x=[bayes], y=[0], mode="markers+text", text=["Bayes"], textposition="bottom center",
                                     marker=dict(size=12, symbol="x", color="#555"), showlegend=False))
        fig.update_layout(height=120, margin=dict(l=10, r=10, t=10, b=10),
                          yaxis=dict(visible=False, range=[-1, 1]), xaxis=dict(title="Opening weekend ($M)"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(note_for(UPCOMING_NOTES, d))

section("Recently released — prediction of record vs. actual")
released = sorted((d for d in DATA["details"] if d["PREDICTION_TYPE"] == "RELEASED" and d.get("ACTUAL_OW_M") is not None),
                  key=lambda d: d["RELEASE_DATE"], reverse=True)[:12]
for d in released:
    with st.container(border=True):
        atier = d.get("ACTUAL_TIER") or ""
        st.markdown(f"### {d['MOVIE_TITLE']}  ·  <span style='color:{TIER_COLOR.get(atier, SF_BLUE)}'>{atier}</span>",
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted (point)", f"${d['PRED_OW_M']:.1f}M")
        c2.metric("Actual opening", f"${d['ACTUAL_OW_M']:.1f}M", f"{d['ACTUAL_OW_M'] - d['PRED_OW_M']:+.1f}M vs pred")
        c3.metric("Tier call", "Hit" if d.get("TIER_HIT") else "Miss")
        st.caption(note_for(RELEASED_NOTES, d))

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
st.caption("These are the predictions of record: each film's closest-horizon V31 call made before release, from "
           "SPARK_PAR_DEMO.PRODUCTION.OW_PREDICTIONS_V21, never edited after the fact.")
show_cortex_badge()
