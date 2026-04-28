"""Page 5: Error analysis — V20 base cascade biggest misses and what V20-Clip + RC fixes."""
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from theme import TIER_COLORS, apply_page_config, page_header, section, show_cortex_badge

apply_page_config("Errors", icon="⚠️")

page_header(
    "Error Analysis",
    "Biggest misses on the V20 base cascade (V18.7 soft-mixture OOF), plus what V20-Clip + RC fixes.",
)


@st.cache_data
def load_preds():
    p = os.path.join(os.path.dirname(__file__), "..", "data", "cv_predictions_v18.json")
    with open(p) as f:
        return json.load(f)


preds = load_preds()
df = pd.DataFrame(preds)
df["error_m"] = df["predicted_ow_m"] - df["actual_ow_m"]
df["abs_error_m"] = df["error_m"].abs()

tab_worst, tab_patterns, tab_limits = st.tabs(
    ["Biggest misses", "Under vs over", "Known limitations"]
)

with tab_worst:
    section(
        "Top 15 largest OOF errors (V20 base cascade)",
        "Out-of-fold predictions from the V18.7 soft-mixture stage that V20 inherits; "
        "each film was predicted by a model that never saw it in training.",
    )
    worst = df.nlargest(15, "abs_error_m")[[
        "movie_title", "actual_tier", "predicted_tier", "actual_ow_m", "predicted_ow_m", "error_m"
    ]].copy()
    worst.columns = ["Movie", "Actual Tier", "Predicted Tier",
                     "Actual ($M)", "Predicted ($M)", "Error ($M)"]
    worst["Actual ($M)"] = worst["Actual ($M)"].round(2)
    worst["Predicted ($M)"] = worst["Predicted ($M)"].round(2)
    worst["Error ($M)"] = worst["Error ($M)"].round(2)

    fig = px.bar(worst, x="Movie", y="Error ($M)", color="Error ($M)",
                 color_continuous_scale="RdBu_r",
                 hover_data=["Actual ($M)", "Predicted ($M)", "Actual Tier", "Predicted Tier"])
    fig.update_layout(height=420, xaxis_tickangle=35)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(worst, use_container_width=True, hide_index=True)

with tab_patterns:
    under = df[df["error_m"] < -10].sort_values("error_m").head(10)
    over = df[df["error_m"] > 10].sort_values("error_m", ascending=False).head(10)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Under-predictions**")
        st.caption("Model said too low — usually cultural breakouts or under-represented IP.")
        cols = ["movie_title", "actual_ow_m", "predicted_ow_m", "error_m"]
        u = under[cols].copy()
        u.columns = ["Movie", "Actual ($M)", "Pred ($M)", "Error ($M)"]
        st.dataframe(u.round(2), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Over-predictions**")
        st.caption("Model said too high — usually franchise fatigue or crowded release windows.")
        cols = ["movie_title", "actual_ow_m", "predicted_ow_m", "error_m"]
        o = over[cols].copy()
        o.columns = ["Movie", "Actual ($M)", "Pred ($M)", "Error ($M)"]
        st.dataframe(o.round(2), use_container_width=True, hide_index=True)

    section("Error distribution")
    fig = px.histogram(df, x="error_m", nbins=40,
                       labels={"error_m": "Error ($M)"}, color_discrete_sequence=["#29B5E8"])
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.update_layout(height=340, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean error", f"${df['error_m'].mean():.2f}M")
    c2.metric("Median error", f"${df['error_m'].median():.2f}M")
    c3.metric("|error| ≤ $5M", f"{(df['abs_error_m'] <= 5).mean() * 100:.0f}%")
    c4.metric("|error| ≤ $15M", f"{(df['abs_error_m'] <= 15).mean() * 100:.0f}%")

with tab_limits:
    st.info(
        "**What V20 fixes vs V18.0:** V20-Clip trims MID uncertainty (MID MAE $11.92M → $9.27M) by "
        "clipping to the adaptive [Q10, Q90] window, and guarded Rule C attacks the LARGE+ tail "
        "(LARGE+ MAE $31.24M → $25.17M) by lifting tentpoles flagged by TMDB D14 when V20-Clip < $60M."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Where the model still struggles**")
        st.markdown(
            "- **Event films** (Taylor Swift, BTS) — not theatrical releases in the training sense.\n"
            "- **Cultural phenomena** (Barbenheimer-style viral moments) — social-media-driven.\n"
            "- **Franchise fatigue** — model uses prior-film OW; doesn't yet detect declining interest.\n"
            "- **Anime** (Demon Slayer, Dragon Ball) — excluded from training, different fanbase."
        )
    with c2:
        st.markdown("**Where V20 excels**")
        st.markdown(
            "- Standard studio releases with typical marketing patterns.\n"
            "- Established franchises that aren't fatigued.\n"
            "- Mid-range films — the largest training bucket.\n"
            "- Genre films with clear comparables (horror, animation, action franchise)."
        )

    section("Roadmap")
    st.markdown(
        "- **Social media features** (TikTok, X) — close the event-film gap.\n"
        "- **Franchise-fatigue detector** — derived from sentiment delta vs prior film.\n"
        "- **D-3 override extension** — within-tier regressor correction at the closest horizon.\n"
        "- **Continuous data refresh** — TMDB + Wikipedia auto-pull windowed around each release."
    )

show_cortex_badge()
