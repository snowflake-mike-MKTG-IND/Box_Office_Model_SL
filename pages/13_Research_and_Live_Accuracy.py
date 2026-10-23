"""Brown's research log and live accuracy. Every number comes from data/brown/*.json, written by `lomis research docs build`."""
import json
import os

import pandas as pd
import streamlit as st

from theme import apply_page_config, page_header, section, kpi_row, freshness_caption


@st.cache_data
def _load(name):
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "brown", name)) as f:
        return json.load(f)


apply_page_config("Research & Live Accuracy", icon="🧪")
page_header(
    "Research & Live Accuracy",
    "The production model's live record, its backtest, and Brown's pre-registered studies and challengers",
)

CARD = _load("model_card.json")
LOG = _load("research_log.json")

# -- Live accuracy -----------------------------------------------------------
live = CARD["live"]["last_180_days"]
section(
    f"Live accuracy · production {CARD['model']} {CARD['production_version']}",
    f"Predictions of record graded against actual opening weekends, {live['SINCE']} to {live['THROUGH']}.",
)
kpi_row([
    ("Graded films", str(live["FILMS"]), None),
    ("Median error", f"{live['MEDIAN_APE_PCT']:.1f}%", None),
    ("Actual inside published range", f"{live['IN_RANGE_PCT']:.1f}%", None),
    ("Size tier correct", f"{live['TIER_HIT_PCT']:.1f}%", None),
])
st.caption(
    "Most graded films were scored by earlier versions (see the Record column). With this few films, "
    "live numbers carry wide uncertainty; study S-1019-b7b4ca below examines the gap to the backtest."
)
recent = pd.DataFrame(CARD["live"]["recent"])
if not recent.empty:
    st.dataframe(recent, hide_index=True, use_container_width=True)

section("Backtest of the production version", CARD["backtest"]["note"])
st.dataframe(pd.DataFrame(CARD["backtest"]["by_horizon"]), hide_index=True, use_container_width=True)

# -- Studies -----------------------------------------------------------------
section("Studies", "Each study is pre-registered (hypothesis, metric, decision rule) before it runs.")
studies = LOG["studies"]
if not studies:
    st.write("No studies yet.")
for s in studies:
    state = s["OUTCOME"] if s["STATUS"] == "CLOSED" else "open, not yet concluded"
    with st.expander(f"{s['STUDY_ID']} · {s['TITLE']} · {state}"):
        st.markdown(f"**Hypothesis.** {s['HYPOTHESIS']}")
        st.markdown(f"**Primary metric.** {s['PRIMARY_METRIC']}")
        st.markdown(f"**Decision rule.** {s['DECISION_RULE']}")
        if s["STATUS"] == "CLOSED":
            st.markdown(f"**Result** (closed {s['CLOSED_AT'][:10]}). {s['RESULT']}")
        else:
            st.markdown(f"Opened {s['OPENED_AT'][:10]}; results will appear when it closes.")

# -- Challengers and shadow ---------------------------------------------------
section("Challenger models and shadow forecasts",
        "A challenger is frozen from a closed study and forecasts each wide release before it opens.")
kpi_row([
    ("Challengers", str(len(LOG["challengers"])), None),
    ("Shadow forecasts recorded", str(len(LOG["shadow"])), None),
])
if LOG["challengers"]:
    st.dataframe(pd.DataFrame(LOG["challengers"]), hide_index=True, use_container_width=True)
else:
    st.write("No challenger is registered yet: none of the closed studies recommended a model change.")

freshness_caption("lomis research docs build", LOG["generated_at"][:10])
