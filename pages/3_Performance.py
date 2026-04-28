"""Page 3: Performance — V20 CV results; V18 scatter retained as the base-cascade view."""
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
    "V20-Clip + Rule C · 5-fold GroupKFold CV · 277 films · 72 features.",
)

kpi_row([
    ("V20 CV MAE (-7d)",   "$9.58M",  "-16.5% vs V18.0"),
    ("V20 CV R²",          "0.814",   "+0.090 vs V18.0"),
    ("V20 LARGE+ MAE",     "$25.55M", "-$5.69M vs V18.0"),
    ("V20 MID MAE",        "$9.10M",  "-$2.82M vs V18.0"),
])
freshness_caption("5-fold GroupKFold CV · V20-Clip + Rule C · 277 films · fresh OOF run", "2026-04-27")

tab_scatter, tab_horizon, tab_tier, tab_versions = st.tabs(
    ["Predicted vs actual", "By horizon", "By tier", "Version comparison"]
)

with tab_horizon:
    horizon_data = pd.DataFrame({
        "Days Out": ["-14 days", "-7 days", "-3 days"],
        "Classification": [75.7, 77.2, 74.6],
        "V18 MAE ($M)": [11.22, 10.96, 11.21],
        "V20 MAE ($M)": [None, 9.58, None],
    })
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(horizon_data, x="Days Out", y="Classification",
                     title="Classification accuracy (V18.7 classifier, reused by V20)", text="Classification",
                     color="Classification", color_continuous_scale="Greens")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(yaxis_range=[60, 85], height=360)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        plot_df = horizon_data.melt(id_vars="Days Out",
                                    value_vars=["V18 MAE ($M)", "V20 MAE ($M)"],
                                    var_name="Model", value_name="MAE ($M)").dropna()
        fig = px.bar(plot_df, x="Days Out", y="MAE ($M)", color="Model",
                     title="MAE by horizon", text="MAE ($M)", barmode="group",
                     color_discrete_map={"V18 MAE ($M)": "#11567F", "V20 MAE ($M)": "#29B5E8"})
        fig.update_traces(texttemplate="$%{text:.2f}M", textposition="outside")
        fig.update_layout(yaxis_range=[0, 15], height=360)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("V20-Clip + Rule C was re-run at D-7 this session; V18 numbers shown for the other horizons.")

with tab_tier:
    tier_data = pd.DataFrame({
        "Tier": ["SMALL", "MID", "LARGE+"],
        "V18 MAE ($M)": [4.12, 11.92, 31.24],
        "V20 MAE ($M)": [4.12, 9.10, 25.55],
        "Median OW ($M)": [7, 28, 85],
        "Training films": [142, 84, 51],
    })
    tier_data["V20 MAE % of Median"] = (tier_data["V20 MAE ($M)"] / tier_data["Median OW ($M)"] * 100).round(0)

    c1, c2 = st.columns([1, 1])
    with c1:
        plot_df = tier_data.melt(id_vars="Tier", value_vars=["V18 MAE ($M)", "V20 MAE ($M)"],
                                 var_name="Model", value_name="MAE ($M)")
        fig = px.bar(plot_df, x="Tier", y="MAE ($M)", color="Model", barmode="group",
                     text="MAE ($M)",
                     color_discrete_map={"V18 MAE ($M)": "#11567F", "V20 MAE ($M)": "#29B5E8"})
        fig.update_traces(texttemplate="$%{text:.2f}M", textposition="outside")
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.dataframe(tier_data, use_container_width=True, hide_index=True)
        st.caption(
            "V20-Clip + guarded Rule C cuts LARGE+ MAE by **$5.69M** and MID MAE by **$2.82M** "
            "vs V18.0 — the two tiers V20 was specifically designed to attack."
        )

with tab_scatter:
    view_mode = st.radio(
        "View",
        ["CV (out-of-sample)", "Training fit"],
        horizontal=True,
        help="CV = each film predicted by a model that never saw it. Training fit = model scored on films it was trained on (shows overfit gap).",
    )
    if view_mode == "CV (out-of-sample)":
        path = os.path.join(os.path.dirname(__file__), "..", "data", "cv_predictions_v20.json")
        st.caption("V20-Clip + Rule C out-of-fold predictions (277 films, 5-fold GroupKFold CV at -7d). Each film is predicted by a V20 model that never saw it during training.")
    else:
        path = os.path.join(os.path.dirname(__file__), "..", "data", "training_predictions_v20.json")
        st.caption("V20-Clip + Rule C training-fit predictions (trained + predicted on the same 277 films). Illustrates the *overfit gap*: training fit is tight (MAE $1.55M, R² 0.996) because the model has seen every point — compare to the out-of-sample CV view (MAE $9.58M) for the true generalization picture.")
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
        "Metric": ["Training films", "Features", "CV MAE -7d", "CV R²",
                   "SMALL MAE", "MID MAE", "LARGE+ MAE"],
        "V17.1":       ["277", "59", "$11.44M", "0.71",  "—",       "—",        "—"],
        "V18.0":       ["276", "72", "$11.48M", "0.724", "$4.12M", "$11.92M", "$31.24M"],
        "V18.7":       ["277", "72", "$10.42M", "0.757", "$4.24M", "$9.40M",  "$29.30M"],
        "V20-Clip":    ["277", "72", "$10.29M", "0.756", "$4.18M", "$9.34M",  "$28.86M"],
        "V20-Clip+RC": ["277", "72", "**$9.58M**", "**0.814**", "**$4.12M**", "**$9.10M**", "**$25.55M**"],
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    section("What changed from V18 → V20")
    st.markdown(
        "- **V18.7 soft mixture** — probability-weighted blend across the 3 tier regressors instead "
        "of argmax hard-routing. First $1M+ MAE win.\n"
        "- **V20 quantile window** — 6 expanded-pool quantile regressors (Q10/Q90 × 3 tiers) trained "
        "on tier + 30% of each neighbor, so the window stretches when the classifier is uncertain.\n"
        "- **V20-Clip** — clip the soft-mixture point into the adaptive window.\n"
        "- **Guarded Rule C** — V20 keeps Rule C's TMDB override but only fires when V20-Clip < $60M, "
        "avoiding double-lift on already-large predictions.\n"
        "- See the [V20 Model Story](./V20_Model_Story) for the full iteration arc."
    )

show_cortex_badge()
