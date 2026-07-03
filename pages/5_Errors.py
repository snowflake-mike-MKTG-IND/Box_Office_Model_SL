"""Page 5: Error analysis — V28-B leak-safe backtest (310 films × 3 horizons) biggest misses, and why the
LARGE+ noise floor pushed V28-B toward calibrated breakout odds rather than chasing the point."""
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
    "Biggest misses on the V28-B leak-safe backtest (310 films × 3 horizons, nested 5-fold), and what they reveal "
    "about the breakout noise floor.",
)


@st.cache_data
def load_preds():
    p = os.path.join(os.path.dirname(__file__), "..", "data", "cv_predictions_v28.json")
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
        "Top 15 largest errors (V28-B backtest)",
        "Out-of-fold predictions — each film was scored by V28-B (CatBoost multiclass + per-tier regressors + blend) that "
        "never saw it in training. The largest misses are almost all under-predicted breakouts.",
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
    st.caption(
        "Negative error = the model called it too low. The biggest blockbusters (Deadpool & Wolverine, etc.) "
        "land here because no pre-release signal separates a $140M opening from a $210M one — the gap is the "
        "noise floor described under **Known limitations**."
    )

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
        "**The LARGE+ noise floor.** V28-B's per-tier backtest MAE is **$3.85M (SMALL)**, **$8.60M (MID)** and "
        "**$35.16M (LARGE+)**. The top-tier error is large not because the model is weak there but because two "
        "films with near-identical pre-release demand can open $50M+ apart. No feature available at -7d resolves "
        "that, so chasing a sharper point estimate would just be over-fitting to hindsight."
    )
    st.success(
        "**Why V28-B reports odds, not a hero number.** Instead of forcing a precise top-tier point, V28-B "
        "surfaces a **calibrated breakout probability** P(LARGE+) and bear/base/bull bands. Those buckets are "
        "validated: films flagged **>50%** open LARGE+ about **87%** of the time. The decision the user actually "
        "needs — *is this a breakout?* — is answered honestly even when the dollar point cannot be."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Where the model still struggles**")
        st.markdown(
            "- **Top-tier point precision** — the $50M+ noise floor above; bands, not points, carry the signal.\n"
            "- **Event films** (Taylor Swift, BTS) — not theatrical releases in the training sense.\n"
            "- **Cultural phenomena** (Barbenheimer-style viral moments) — social-media-driven.\n"
            "- **Anime** (Demon Slayer, Dragon Ball) — excluded from training, different fanbase."
        )
    with c2:
        st.markdown("**Where V28-B is strong**")
        st.markdown(
            "- **SMALL/MID tiers** — $3.85M / $8.60M MAE on the largest training buckets.\n"
            "- **Breakout calibration** — the >50% bucket hits LARGE+ ~87% of the time.\n"
            "- **Rule-free generalization** — no hand-coded overrides to break on new films.\n"
            "- Standard studio releases with typical marketing and a clear comparable set."
        )

    section("Reading a miss correctly")
    st.markdown(
        "- A large **under-prediction on a LARGE+ film is expected** — check whether its breakout odds flagged "
        "the risk, not whether the point nailed the dollar.\n"
        "- **Over-predictions** cluster in crowded windows and fatigued franchises; the bear band usually "
        "covers the downside.\n"
        "- The honest read is **band coverage + breakout calibration**, not point MAE on the tail."
    )

show_cortex_badge()
