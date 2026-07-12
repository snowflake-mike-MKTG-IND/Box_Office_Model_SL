"""V30 Features — feature stack-ranking (pedigree enters only via demand interactions)."""
import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from theme import SF_BLUE, DK2, ORANGE, VIOLET, apply_page_config, page_header, section, show_cortex_badge

apply_page_config("V30 · Features", icon="📊")
page_header(
    "V30 Feature Importance",
    "37 regressor features · CatBoost gain importance · pedigree survives only through demand-gated interactions",
)


@st.cache_data
def load_importance():
    p = os.path.join(os.path.dirname(__file__), "..", "data", "feature_importance_v30.json")
    with open(p) as f:
        return json.load(f)


base = load_importance()["base"]
df = pd.DataFrame(base).sort_values("importance", ascending=False)

# Classify each feature for coloring
INTERACTIONS = {"PCTILE_X_STAR", "PCTILE_X_ACTION", "PCTILE_X_IP_HIGH", "PCTILE_X_PREDOW",
                "PCTILE_X_HORROR", "SENT_X_PCTILE", "THEA_X_PCTILE", "WIKIPK_X_STAR"}
DEMAND = {"ROLLING_3D_PCTILE", "ROLLING_7D_PCTILE", "ROLLING_14D_PCTILE", "TRENDS_PEAK_PCTILE",
          "VEL_3V7_PCTILE", "LOG_SLOPE_PCTILE", "WIKI_R7D_PCTILE", "WIKI_PEAK_PCTILE",
          "WIKI_CUM_PCTILE", "WIKI_VEL_PCTILE"}
INTENT = {"LOG_YT", "NET_INTENT_PCT", "AVG_SENT", "PCT_THEA", "PCT_POS", "PCT_NEG", "PCT_PASS"}


def cat(f):
    if f in INTERACTIONS:
        return "Demand × pedigree interaction"
    if f in DEMAND:
        return "Percentile demand (Trends/Wiki)"
    if f in INTENT:
        return "YouTube / intent / sentiment"
    return "Movie attributes / genre / rating"


df["Category"] = df["feature"].map(cat)
CMAP = {
    "Demand × pedigree interaction": VIOLET,
    "Percentile demand (Trends/Wiki)": SF_BLUE,
    "YouTube / intent / sentiment": ORANGE,
    "Movie attributes / genre / rating": "#9CA3AF",
}

section("Top 20 features by CatBoost gain")
top = df.head(20).iloc[::-1]
fig = px.bar(top, x="importance", y="feature", color="Category", orientation="h",
             color_discrete_map=CMAP, height=560,
             labels={"importance": "Gain importance", "feature": ""})
fig.update_layout(legend=dict(orientation="h", y=-0.12), margin=dict(l=10, r=10, t=10, b=10),
                  yaxis=dict(categoryorder="total ascending"))
st.plotly_chart(fig, use_container_width=True)

c1, c2, c3 = st.columns(3)
c1.metric("Top feature", df.iloc[0]["feature"], f"{df.iloc[0]['importance']:.1f} gain")
c2.metric("Interaction terms in top 10", str(sum(1 for f in df.head(10)["feature"] if f in INTERACTIONS)))
c3.metric("Standalone pedigree features", "0", "removed — gated only")

section("Why this ranking matters")
st.markdown(
    "- **`LOG_YT` (YouTube trailer engagement) leads** — the single strongest pre-release demand signal we can "
    "observe without tracking or ticketing data.\n"
    "- **`PCTILE_X_STAR`, `PCTILE_X_ACTION`, `PCTILE_X_IP_HIGH` rank next** — these are *interactions*, not raw "
    "pedigree. Star power, franchise/action, and high-profile IP only move the prediction **when Google-Trends "
    "demand confirms them**. A big-budget star vehicle with quiet search interest is not treated as large.\n"
    "- **`NET_INTENT_PCT` and sentiment/intent aggregates** (from scored YouTube comments) contribute genuine "
    "signal that univariate correlation misses — they matter through nonlinear interactions in the tree.\n"
    "- **No standalone `BUDGET_LOG` / `MAX_STAR_POWER` / `IP_*` / `PREDECESSOR_OW`** appear — they were removed "
    "from the regressor. This gating is the mechanism that fixed hype-flop over-prediction."
)

with st.expander("Full 37-feature table"):
    st.dataframe(df[["feature", "importance", "Category"]].reset_index(drop=True), use_container_width=True)

st.caption("Source: CatBoost gain importance from the registered V30 regressor (cm_v7_pg3.pkl).")
show_cortex_badge()
