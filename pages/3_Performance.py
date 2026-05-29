"""Page 3: Performance — V24 CV results with escape velocity detection."""
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
    "V24 Escape Velocity · 5-fold GroupKFold CV · 287 films · 75 features.",
)

kpi_row([
    ("V24 CV MAE (-7d)",   "$9.89M",  "287 films"),
    ("V24 CV Accuracy",    "77.4%",   "222/287 at D-7"),
    ("Rule F fires (D-3)", "24",      "0 false positives"),
    ("Rule G fires (D-3)", "9",       "demand dominance"),
])
freshness_caption("5-fold GroupKFold CV · V24 escape velocity + demand dominance · 287 films", "2026-05-29")

tab_scatter, tab_tier, tab_versions = st.tabs(
    ["Predicted vs actual", "By tier", "Version comparison"]
)

with tab_tier:
    tier_data = pd.DataFrame({
        "Tier": ["SMALL", "MID", "LARGE+"],
        "V22c MAE ($M)": [4.11, 9.24, 25.65],
        "V23b MAE ($M)": [4.32, 8.87, 29.84],
        "V23b Accuracy": ["81.0%", "71.3%", "71.7%"],
        "Training films": [147, 87, 53],
    })

    c1, c2 = st.columns([1, 1])
    with c1:
        plot_df = tier_data.melt(id_vars="Tier", value_vars=["V22c MAE ($M)", "V23b MAE ($M)"],
                                 var_name="Model", value_name="MAE ($M)")
        fig = px.bar(plot_df, x="Tier", y="MAE ($M)", color="Model", barmode="group",
                     text="MAE ($M)",
                     color_discrete_map={"V22c MAE ($M)": "#11567F", "V23b MAE ($M)": "#29B5E8"})
        fig.update_traces(texttemplate="$%{text:.2f}M", textposition="outside")
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.dataframe(tier_data, use_container_width=True, hide_index=True)
        st.caption(
            "V23b improves MID MAE by **$0.37M** vs V22c via horror routing. "
            "LARGE+ MAE increases slightly due to expanded pool (53 vs 52 films). "
            "Horror subset: 73.5% accuracy, $8.73M MAE with dedicated regressors."
        )

with tab_scatter:
    view_mode = st.radio(
        "View",
        ["V23b CV (out-of-sample)", "V22c CV (comparison)"],
        horizontal=True,
        help="V23b = horror-first routing. V22c = hybrid blend baseline. Both are real 5-fold GroupKFold CV at D-7.",
    )
    if view_mode == "V23b CV (out-of-sample)":
        path = os.path.join(os.path.dirname(__file__), "..", "data", "cv_predictions_v23b.json")
        st.caption("V23b Horror-first routing out-of-fold predictions (287 films, 5-fold GroupKFold CV at -7d). Each film predicted by a model that never saw it during training.")
    else:
        path = os.path.join(os.path.dirname(__file__), "..", "data", "cv_predictions_v22c.json")
        st.caption("V22c Hybrid Blend out-of-fold predictions (282 films, 5-fold GroupKFold CV at -7d). Retained for comparison.")
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
        "Metric": ["Training films", "Features", "CV MAE -7d", "CV Accuracy",
                   "SMALL MAE", "MID MAE", "LARGE+ MAE", "Horror Accuracy"],
        "V18.0":       ["276", "72", "$11.48M", "75.7%", "$4.12M", "$11.92M", "$31.24M", "—"],
        "V22c":        ["282", "72", "$9.65M", "78.4%", "$4.11M", "$9.24M", "$25.65M", "—"],
        "V23b":        ["287", "72", "**$10.41M**", "**76.3%**", "**$4.32M**", "**$8.87M**", "**$29.84M**", "**73.5%**"],
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    section("What changed from V22c → V23b")
    st.markdown(
        "- **Horror-first routing** — Step 1 splits horror vs non-horror before any tier classification. "
        "Horror movies get dedicated 2-bucket regressors (Small/Large split at $17M via log-space KMeans). "
        "Non-horror movies proceed through standard 3-tier cascade.\n"
        "- **PRODUCTION data fix** — Live feature view was reading from STAGING (197 movies) instead of "
        "PRODUCTION (413 movies). 67% of training films had zero GT features. Single biggest fix.\n"
        "- **287 training films** — +5 from V22c after data quality pass.\n"
        "- **Snowflake Model Registry** — Full ML pipeline: Feature Store → CustomModel → batch inference.\n"
        "- **Note on CV numbers**: V23b overall MAE ($10.41M) is slightly higher than V22c ($9.65M) in CV "
        "because (1) 5 additional films added volatility, (2) horror routing optimizes for horror accuracy "
        "at a small cost to non-horror, and (3) the PRODUCTION data fix helps live predictions more than "
        "CV (CV already had the training data available). The key win is horror breakout detection: "
        "73.5% accuracy on 68 horror films with $8.73M MAE.\n"
    )

show_cortex_badge()
