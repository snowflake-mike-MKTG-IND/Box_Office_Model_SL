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
    "V28-B horizon-normalized demand classification · nested 5-fold GroupKFold CV · calibrated breakout odds.",
)

kpi_row([
    ("V28-B CV accuracy",  "76.6%",  "D-14: 75.2% · D-7: 77.0% · D-3: 77.7%"),
    ("Leak-safe backtest",  "$10.96M", "75.3% · 288 incl. recent breakouts"),
    ("Breakout recall",     "67%",     "of breakouts flagged early"),
    ("Calibration",         ">50%→87%", "flagged → actual LARGE+"),
])
freshness_caption("Nested 5-fold GroupKFold CV · V28-B horizon-normalized demand classification · deployed to Snowflake Model Registry (OW_PREDICTION_V28)", "2026-06-08")

tab_scatter, tab_tier, tab_versions = st.tabs(
    ["Predicted vs actual", "By tier", "Version comparison"]
)

with tab_tier:
    tier_data = pd.DataFrame({
        "Tier": ["SMALL", "MID", "LARGE+"],
        "V23b MAE ($M)": [4.32, 8.87, 29.84],
        "V28-B MAE ($M)": [3.85, 8.60, 35.16],
        "V28-B Accuracy": ["81.1%", "71.6%", "65.4%"],
        "Films (backtest)": [148, 88, 52],
    })

    c1, c2 = st.columns([1, 1])
    with c1:
        plot_df = tier_data.melt(id_vars="Tier", value_vars=["V23b MAE ($M)", "V28-B MAE ($M)"],
                                 var_name="Model", value_name="MAE ($M)")
        fig = px.bar(plot_df, x="Tier", y="MAE ($M)", color="Model", barmode="group",
                     text="MAE ($M)",
                     color_discrete_map={"V23b MAE ($M)": "#11567F", "V28-B MAE ($M)": "#29B5E8"})
        fig.update_traces(texttemplate="$%{text:.2f}M", textposition="outside")
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.dataframe(tier_data, use_container_width=True, hide_index=True)
        st.caption(
            "V28-B (rule-free) tightens SMALL and MID MAE vs V23b. LARGE+ MAE rises because the "
            "310-film leak-safe backtest now includes the 4 hardest recent breakouts — and LARGE+ "
            "sits at the **measured noise floor**, where two films that look identical pre-release can "
            "open $50M apart. So V28-B reports a calibrated **breakout probability** instead of chasing "
            "a point estimate the data can't support."
        )
        st.markdown(
            "**Breakout-odds calibration** (P(LARGE+) bucket → actual LARGE+ rate): "
            "<15% → 1% · 15-30% → 17% · 30-50% → 39% · >50% → **87%**."
        )

with tab_scatter:
    view_mode = st.radio(
        "View",
        ["V28-B backtest (out-of-sample)", "V23b CV (comparison)"],
        horizontal=True,
        help="V28-B = horizon-normalized demand classification, 310-film leak-safe backtest. V23b = horror-first routing. Both are out-of-fold at D-7.",
    )
    if view_mode == "V28-B backtest (out-of-sample)":
        path = os.path.join(os.path.dirname(__file__), "..", "data", "cv_predictions_v28.json")
        st.caption("V28-B horizon-normalized demand classification out-of-fold predictions (310 films × 3 horizons, leak-safe backtest at -7d incl. the 4 hardest recent breakouts). Each film predicted by a model that never saw it during training.")
    else:
        path = os.path.join(os.path.dirname(__file__), "..", "data", "cv_predictions_v23b.json")
        st.caption("V23b Horror-first routing out-of-fold predictions (310 films × 3 horizons, 5-fold GroupKFold CV at -7d). Retained for comparison.")
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
        "Metric": ["Films", "CV MAE -7d", "CV Accuracy",
                   "SMALL MAE", "MID MAE", "LARGE+ MAE", "Rule-free?"],
        "V18.0":       ["276", "$11.48M", "75.7%", "$4.12M", "$11.92M", "$31.24M", "No"],
        "V23b":        ["287", "$10.41M", "76.3%", "$4.32M", "$8.87M", "$29.84M", "No (rules)"],
        "V25":         ["287", "$9.88M", "77.4%", "—", "—", "—", "No (rules)"],
        "V28-B":       ["288", "**$10.96M**", "**75.3%**", "**$3.85M**", "**$8.60M**", "**$35.16M**", "**Yes**"],
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.caption(
        "V28-B's backtest (310 films × 3 horizons) includes the 4 hardest recent breakouts that earlier versions "
        "were never measured against; on the **same-basis** nested CV it is **77.7% / $9.99M** (310 films × 3 horizons), "
        "comparable to V25/V27. The headline change isn't raw accuracy — it's being fully rule-free and "
        "adding calibrated breakout odds."
    )

    section("What changed: V25 → V27 → V28-B")
    st.markdown(
        "- **V25 — demand-driven classifier**: Google Trends features moved into tier assignment so budget "
        "no longer dominates. D-7 77.4% / $9.88M.\n"
        "- **V27 — modern ensemble**: tuned CatBoost (trees) + TabPFN (transformer) soft-vote classifier, "
        "no hand rules. Diverse members lift borderline tier accuracy (agreement 87.5%, corr 0.95).\n"
        "- **V28-B — horizon-normalized demand classification**: a small model learns how to combine the base "
        "classifier + per-tier regressors (FINAL = 0.7·g + 0.3·mixture) instead of hand-coded rules. "
        "Adds bear/base/bull ranges + calibrated P(LARGE+). LARGE+ now sits at the measured noise floor, "
        "so V28-B reports probability rather than chasing point accuracy the data can't support.\n"
    )

show_cortex_badge()
