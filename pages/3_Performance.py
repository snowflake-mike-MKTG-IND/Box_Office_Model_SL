"""Page 3: Performance — V18 CV results and error analysis."""
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from theme import (
    TIER_COLORS,
    apply_page_config,
    freshness_caption,
    kpi_row,
    page_header,
    section,
    show_cortex_badge,
)

apply_page_config("Performance", icon="📈")

page_header(
    "Performance",
    "V18 5-fold GroupKFold CV · 276 training films · 72 features.",
)

kpi_row([
    ("CV Accuracy (-7d)", "77.2%", "+5.5pp vs V17.2"),
    ("CV MAE (-7d)",      "$10.96M", "-$0.71M vs V17.2"),
    ("CV Accuracy (-14d)", "75.7%", None),
    ("CV Accuracy (-3d)",  "74.6%", None),
])
freshness_caption("5-fold GroupKFold CV (grouped by MOVIE_ID)", "2026-04-21")

tab_scatter, tab_horizon, tab_tier, tab_versions = st.tabs(
    ["Predicted vs actual", "By horizon", "By tier", "Version comparison"]
)

with tab_horizon:
    horizon_data = pd.DataFrame({
        "Days Out": ["-14 days", "-7 days", "-3 days"],
        "Classification": [75.7, 77.2, 74.6],
        "MAE ($M)": [11.22, 10.96, 11.21],
        "Median AE ($M)": [5.4, 4.7, 5.0],
    })
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(horizon_data, x="Days Out", y="Classification",
                     title="Classification accuracy", text="Classification",
                     color="Classification", color_continuous_scale="Greens")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(yaxis_range=[60, 85], height=360)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(horizon_data, x="Days Out", y="MAE ($M)",
                     title="MAE by horizon", text="MAE ($M)",
                     color="MAE ($M)", color_continuous_scale="Reds_r")
        fig.update_traces(texttemplate="$%{text:.1f}M", textposition="outside")
        fig.update_layout(yaxis_range=[0, 15], height=360)
        st.plotly_chart(fig, use_container_width=True)

with tab_tier:
    tier_data = pd.DataFrame({
        "Tier": ["SMALL", "MID", "LARGE+"],
        "Accuracy %": [87.3, 59.5, 72.5],
        "MAE ($M)": [3.78, 12.35, 31.29],
        "Median OW ($M)": [7, 28, 85],
        "Training films": [142, 84, 50],
    })
    tier_data["MAE % of Median"] = (tier_data["MAE ($M)"] / tier_data["Median OW ($M)"] * 100).round(0)

    c1, c2 = st.columns([1, 1])
    with c1:
        fig = px.bar(tier_data, x="Tier", y="MAE ($M)", color="Tier",
                     color_discrete_map=TIER_COLORS, text="MAE ($M)")
        fig.update_traces(texttemplate="$%{text:.1f}M", textposition="outside")
        fig.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.dataframe(tier_data, use_container_width=True, hide_index=True)
        st.caption(
            "LARGE+ has the largest absolute MAE but the smallest **relative** error (~37% of median). "
            "MID is the hardest tier because it sits across both classifier boundaries."
        )

with tab_scatter:
    view_mode = st.radio(
        "View",
        ["CV (out-of-sample)", "Training fit"],
        horizontal=True,
        help="CV = each film predicted by a model that never saw it. Training fit = model scored on films it was trained on (shows overfit gap).",
    )
    if view_mode == "CV (out-of-sample)":
        path = os.path.join(os.path.dirname(__file__), "..", "data", "cv_predictions_v18.json")
        st.caption("V18 out-of-fold predictions from 5-fold GroupKFold CV at -7d. Each film is predicted by a model that never saw it during training.")
    else:
        path = os.path.join(os.path.dirname(__file__), "..", "data", "training_predictions_v17_1.json")
        st.caption("⚠️ V17.1 training-fit shown (V18 training-fit predictions are not cached locally). Illustrates the *overfit gap*: training fit is tighter than CV because the model has seen every point.")
    with open(path) as f:
        preds = json.load(f)

    df_s = pd.DataFrame({
        "Actual ($M)":    [r["actual_ow_m"] for r in preds],
        "Predicted ($M)": [r["predicted_ow_m"] for r in preds],
        "Tier":           [r.get("actual_tier") for r in preds],
        "Movie":          [r["movie_title"] for r in preds],
    })
    max_val = st.select_slider("Zoom", options=[20, 50, 255], value=255, format_func=lambda x: f"${x}M")
    fig = px.scatter(df_s, x="Actual ($M)", y="Predicted ($M)", color="Tier",
                     color_discrete_map=TIER_COLORS, hover_data=["Movie"])
    fig.update_layout(height=480, xaxis=dict(range=[0, max_val]), yaxis=dict(range=[0, max_val]),
                      shapes=[dict(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                                   line=dict(color="gray", dash="dash", width=1), layer="below")])
    st.plotly_chart(fig, use_container_width=True)

    errors = df_s["Predicted ($M)"].values - df_s["Actual ($M)"].values
    c1, c2 = st.columns(2)
    with c1:
        fig_h = px.histogram(errors, nbins=40, title="Error distribution",
                             labels={"value": "Error ($M)", "count": "Frequency"})
        fig_h.add_vline(x=0, line_dash="dash", line_color="red")
        fig_h.update_layout(showlegend=False, height=340)
        st.plotly_chart(fig_h, use_container_width=True)
    with c2:
        st.markdown(
            f"""
            | Metric | Value |
            |---|---|
            | Mean error | ${np.mean(errors):.2f}M |
            | Median error | ${np.median(errors):.2f}M |
            | Std dev | ${np.std(errors):.2f}M |
            | 90th %ile \\|err\\| | ${np.percentile(np.abs(errors), 90):.2f}M |
            """
        )
        st.caption(
            f"Error distribution for {'out-of-fold CV' if view_mode == 'CV (out-of-sample)' else 'training fit'} predictions. "
            "Mean error near zero → model is approximately unbiased. The right tail is driven by "
            "LARGE+ variance (blockbuster breakouts)."
        )

with tab_versions:
    comparison = pd.DataFrame({
        "Metric": ["Training films", "Features", "CV acc -14d", "CV acc -7d", "CV acc -3d",
                   "CV MAE -14d", "CV MAE -7d", "CV MAE -3d"],
        "V17":    ["277", "59", "73.2%", "73.2%", "73.6%", "$11.56M", "$11.74M", "$11.65M"],
        "V17.1":  ["277", "59", "72.9%", "76.2%", "73.6%", "$11.64M", "$11.44M", "$11.31M"],
        "V17.2":  ["276", "59", "72.5%", "71.7%", "72.1%", "$11.68M", "$11.67M", "$11.43M"],
        "V18":    ["276", "72", "**75.7%**", "**77.2%**", "**74.6%**", "**$11.22M**", "**$10.96M**", "**$11.21M**"],
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    section("What changed in V18")
    st.markdown(
        "- **+13 Wikipedia pageview features** (see Features page and Development Story).\n"
        "- **Re-tuned classifier**: Stage 1 (i=200, d=7, lr=0.02), Stage 2 (i=400, d=5, lr=0.03).\n"
        "- **Full data-integrity pass** on all 276 films against The-Numbers.com "
        "(4 fabricated OW values corrected, 13 release dates fixed).\n"
        "- **Key finding**: the 74.6% ceiling from V17.2 was a *data* issue, not an *architecture* "
        "issue — cleaning + new signal unlocked +5.5pp."
    )

show_cortex_badge()
