"""V30 Errors — the demand-quiet-giant ceiling and how V30 manages it."""
import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from theme import SF_BLUE, VIOLET, apply_page_config, page_header, section, show_cortex_badge

apply_page_config("V30 · Errors", icon="🔎")
page_header(
    "Where V30 Misses — and Why That's a Signal Limit, Not a Model Limit",
)


@st.cache_data
def _preds():
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "cv_predictions_v30.json")) as f:
        return json.load(f)


P = _preds()
df = pd.DataFrame(P)
df["signed_err_m"] = df["predicted_ow_m"] - df["actual_ow_m"]
df["ape"] = (df["signed_err_m"].abs() / df["actual_ow_m"].clip(lower=1e-6) * 100)

section("The error profile is deliberately asymmetric")
c1, c2, c3 = st.columns(3)
over = df[df["predicted_ow_m"] > 1.5 * df["actual_ow_m"]]
low = df[df["actual_ow_m"] < 60]
c1.metric("Median APE", f"{df['ape'].median():.0f}%")
c2.metric("Flop over-predictions (<$60M, >1.5×)", f"{(len(over[over['actual_ow_m']<60]) / max(len(low),1) * 100):.0f}%")
c3.metric("Under vs over (all films)", f"{(df['signed_err_m']<0).mean()*100:.0f}% under")
st.caption(
    "V30 is tuned so that when it errs, it errs **low** — under-predicting is cheap, over-predicting a flop is the "
    "cardinal sin. The distribution of signed errors leans negative by design."
)

section("The misses are demand-quiet giants")
big_miss = df[(df["actual_ow_m"] >= 60) & (df["predicted_ow_m"] < df["actual_ow_m"] * 0.7)].sort_values("actual_ow_m", ascending=False)
st.markdown(
    "The films V30 under-shoots most are large openers that had **quiet pre-release demand** — modest Google "
    "search, Wikipedia, and YouTube signal that simply did not foreshadow the eventual opening. These are the "
    "hardest films in the dataset for *any* model that refuses tracking/ticketing data."
)
if len(big_miss):
    show = big_miss.head(10)[["movie_title", "actual_ow_m", "predicted_ow_m", "bayes_ow_m", "p_large"]].rename(
        columns={"movie_title": "Film", "actual_ow_m": "Actual $M", "predicted_ow_m": "HDR50 $M",
                 "bayes_ow_m": "Bayes $M", "p_large": "RF P(large)"})
    st.dataframe(show, use_container_width=True, hide_index=True)

section("Why we don't chase them")
st.markdown(
    "- We tested this exhaustively: a dedicated large-film **classifier + specialist regressor** (routing) was "
    "built and rejected — the classifier's large-film recall caps around **45–48%** and, crucially, the films it "
    "misses are exactly these demand-quiet giants.\n"
    "- We swapped the classifier across **five algorithms** (CatBoost, RandomForest, ExtraTrees, HistGBM, "
    "Logistic Regression). Recall clustered in the same band for all of them, and every one missed the same films "
    "— proving the ceiling is a **signal limit**, not a model-architecture limit.\n"
    "- Chasing them means over-predicting the modest films that look similar pre-release — reintroducing the "
    "cardinal-sin flop over-predictions V30 was built to suppress.\n"
    "- Instead, V30 **prices the uncertainty honestly** (wide HDR bands on ambiguous tentpoles) and leans "
    "conservative via the Bayes point. Given the stated value function, that is the correct trade."
)
show_cortex_badge()
