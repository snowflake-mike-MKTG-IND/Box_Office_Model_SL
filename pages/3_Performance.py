"""V30 Performance — temporal holdout, generational comparison, and the OOF scatter."""
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from theme import SF_BLUE, DK2, ORANGE, VIOLET, apply_page_config, page_header, section, freshness_caption, show_cortex_badge

apply_page_config("V30 · Performance", icon="📈")
page_header(
    "V30 Performance",
    "Validated on a true future holdout — flop-safety is the value function that matters",
)


@st.cache_data
def _load(name):
    with open(os.path.join(os.path.dirname(__file__), "..", "data", name)) as f:
        return json.load(f)


PERF = _load("performance_v30.json")
PREDS = _load("cv_predictions_v30.json")
HO = PERF["holdout_2026"]
OOF = PERF["temporal_oof"]

# -- Holdout head-to-head ----------------------------------------------------
section("True future holdout — train ≤2025 → predict 2026", HO["note"])
hdf = pd.DataFrame([
    {"Model": "V30", "MAPE %": HO["v30"]["mape_pct"], "Asymmetric loss (r=2)": HO["v30"]["aloss_r2"], "Flop over-pred %": HO["v30"]["lowband_over_pct"]},
    {"Model": "v28b (no TMDB)", "MAPE %": HO["v28b_notmdb"]["mape_pct"], "Asymmetric loss (r=2)": HO["v28b_notmdb"]["aloss_r2"], "Flop over-pred %": HO["v28b_notmdb"]["lowband_over_pct"]},
    {"Model": "v28b (as-authored)", "MAPE %": HO["v28b_asauthored"]["mape_pct"], "Asymmetric loss (r=2)": HO["v28b_asauthored"]["aloss_r2"], "Flop over-pred %": HO["v28b_asauthored"]["lowband_over_pct"]},
])
c1, c2 = st.columns(2)
f1 = px.bar(hdf, x="Model", y="Asymmetric loss (r=2)", color="Model",
            color_discrete_map={"V30": VIOLET, "v28b (no TMDB)": ORANGE, "v28b (as-authored)": "#9CA3AF"},
            height=340, title="Asymmetric loss (lower = better; over-prediction penalized 2×)")
f1.update_layout(showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
c1.plotly_chart(f1, use_container_width=True)
f2 = px.bar(hdf, x="Model", y="Flop over-pred %", color="Model",
            color_discrete_map={"V30": VIOLET, "v28b (no TMDB)": ORANGE, "v28b (as-authored)": "#9CA3AF"},
            height=340, title="Flop over-prediction rate (the cardinal sin; lower = better)")
f2.update_layout(showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
c2.plotly_chart(f2, use_container_width=True)
st.dataframe(hdf, use_container_width=True, hide_index=True)
st.caption(
    "26-film holdout (small sample → wide confidence intervals). The as-authored v28b includes TMDB popularity; "
    "the holdout confirmed TMDB is time-inconsistent leakage — stripping it *helped* v28b, but V30 still wins on every metric."
)

# -- OOF scatter -------------------------------------------------------------
section("Out-of-fold predictions (8-block quarterly temporal CV)",
        f"{OOF['n_films']} films · each predicted by a model that never saw it · point = HDR50_MEAN, colored by abs % error")
sdf = pd.DataFrame({
    "Actual ($M)": [r["actual_ow_m"] for r in PREDS],
    "Predicted ($M)": [r["predicted_ow_m"] for r in PREDS],
    "Movie": [r["movie_title"] for r in PREDS],
    "APE %": [round(abs(r["predicted_ow_m"] - r["actual_ow_m"]) / max(r["actual_ow_m"], 1e-6) * 100, 1) for r in PREDS],
})
mx = st.select_slider("Zoom", options=[20, 60, 220], value=220, format_func=lambda x: f"${x}M")
fig = px.scatter(sdf, x="Actual ($M)", y="Predicted ($M)", color="APE %", hover_name="Movie",
                 color_continuous_scale="RdYlGn_r", range_color=[0, 100], height=520)
fig.add_shape(type="line", x0=0, y0=0, x1=mx, y1=mx, line=dict(dash="dash", color="#444"))
fig.add_shape(type="line", x0=0, y0=0, x1=mx, y1=mx * 1.5, line=dict(dash="dot", color="rgba(220,40,40,0.4)"))
fig.update_xaxes(range=[0, mx]); fig.update_yaxes(range=[0, mx])
fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)
k1, k2, k3, k4 = st.columns(4)
k1.metric("MAPE", f"{OOF['mape_pct']:.0f}%")
k2.metric("Median APE", f"{OOF['median_ape_pct']:.0f}%")
k3.metric("Flop over-pred", f"{OOF['lowband_over_pct']:.0f}%")
k4.metric("50% HDR coverage", f"{OOF['hdr50_coverage_pct']:.0f}%", "target 50%")
st.caption("Points below the dashed diagonal = conservative (under-predicted). The dotted red line marks the 1.5× over-prediction boundary — note how few points cross it.")

# -- Guardrail: the two segment-honest error metrics side by side ------------
section("Guardrail metrics — flop-safety vs large-film coverage",
        "The forecaster is tuned to trade large-film under-prediction for low-end flop-safety, by design.")
g1, g2 = st.columns(2)
g1.metric("Flop over-prediction (<$60M, >1.5×)", f"{OOF['lowband_over_pct']:.0f}%", "cardinal sin — kept low")
g2.metric("Large-film under-prediction (≥$60M, mean signed log-err)", f"{OOF.get('large_under_logerr', float('nan')):+.2f}", "negative = under (accepted trade)")
st.caption(
    "We tested a suite of output-layer lifts (P(large)-weighted mixture, r-decay Bayes point, full dollar-space "
    "mean, soft two-part mixture) and one tail feature (absolute-scale demand). They improve aggregate loss and "
    "hold flop-safety, but do **not** move large-film under-prediction on the true 2026 holdout: the giants that "
    "are under-predicted are *demand-quiet* (low P(large)), so any lift conditioned on observed demand cannot "
    "reach them. This is the same signal ceiling — a precise statement of where industry-critical tracking data "
    "would still add value."
)

# -- Generational --------------------------------------------------------------
section("Generational evolution", "Prior versions were classifiers (tier accuracy); V30 is distributional (holdout aLoss / flop-safety).")
gdf = pd.DataFrame(PERF["generations"])
gdf_show = gdf[["model", "date", "type", "tier_acc_pct", "note"]].rename(
    columns={"model": "Model", "date": "Date", "type": "Architecture", "tier_acc_pct": "Tier acc %", "note": "Notes"})
st.dataframe(gdf_show, use_container_width=True, hide_index=True)
st.caption(
    "Tier-classification accuracy peaked around V22c (~78%). V30 changes the objective: instead of a tier label it "
    "emits a calibrated distribution, and is judged on the asymmetric (flop-safety) loss on a true future holdout — "
    "where it decisively beats the prior generation."
)

freshness_caption("cm_v7_pg3.pkl temporal OOF + 2025→2026 holdout · OW_PREDICTION_V30", "2026-07-11")
show_cortex_badge()
