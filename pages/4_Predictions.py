"""Page 4: Interactive cascade simulator.

Note: the live inference backend uses the last locally-bundled joblib
(V17.1). The V18 model requires Wikipedia features that are produced upstream
in Snowflake (`PRODUCTION.WIKIPEDIA_FEATURES_V`) and is not packaged with this
app. This page exposes the same 2-stage cascade + Rule C logic so users can
explore feature sensitivity; numeric predictions should be read as
V17.1-equivalent estimates.
"""
import gzip
import os

import joblib
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from theme import TIER_COLORS, apply_page_config, page_header, section, show_cortex_badge

apply_page_config("Cascade simulator", icon="🔮")

page_header(
    "Cascade Simulator",
    "Explore the base cascade logic — the live V28-B model runs in Snowflake.",
)

st.info(
    "The current production model is **V28-B** — a horizon-normalized demand classification over a tuned "
    "CatBoost multiclass base, deployed to the Snowflake Model Registry "
    "(`SPARK_PAR_DEMO.ML_PIPELINE.OW_PREDICTION_V28`). This page runs the locally-bundled V17.1 joblib "
    "to let you explore the 2-stage cascade + Rule C logic that the base inherits from V18.7. The "
    "**tier decision** illustrates the cascade; the **dollar point** is a V17.1-equivalent estimate and "
    "does not include V28-B's full horizon-normalized pipeline, breakout odds, or bear/base/bull range."
)
st.caption(
    "For actual V28-B predictions see the Recent Predictions page (live weekend tracking)."
)


@st.cache_resource
def load_model():
    p = os.path.join(os.path.dirname(__file__), "..", "models", "ow_pipeline_v17_1_production.joblib.gz")
    with gzip.open(p, "rb") as f:
        return joblib.load(f)


try:
    model = load_model()
    model_loaded = True
except Exception as exc:
    model_loaded = False
    st.error(f"Could not load model: {exc}")


POPULARITY = {
    "Indie release (Moonlight)": 25, "Limited (Lady Bird)": 75,
    "Mid-tier (A Quiet Place)": 110, "Moderate (Knives Out)": 150,
    "High anticipation (Dune)": 300, "Blockbuster (Avengers)": 450,
}
STARS = {
    "Unknown cast": (5.0, 3.0), "Rising (Chalamet)": (25.0, 12.0),
    "Recognizable (Gosling)": (40.0, 18.0), "A-list (Cruise)": (55.0, 25.0),
    "Ensemble (Barbie)": (70.0, 45.0), "Mega (Avengers)": (95.0, 60.0),
}
TRENDS_EX = {
    "Minimal (indie)": (15, -5), "Steady (mid drama)": (35, 5),
    "Growing (Oppenheimer)": (60, 15), "Viral (Barbie)": (85, 35),
    "Peak hype (Endgame)": (100, 50),
}


def apply_override(pop_d14, pop_d7, model_tier):
    if pop_d14 is None or np.isnan(pop_d14):
        return model_tier
    if pop_d14 >= 25:
        return max(model_tier, 2)
    if pop_d14 >= 15 and pop_d7 is not None and not np.isnan(pop_d7) and pop_d14 > 0:
        if pop_d7 / pop_d14 >= 1.3:
            return max(model_tier, 1)
    return model_tier


inputs_col, results_col = st.columns([1, 2])

with inputs_col:
    st.subheader("Inputs")
    tab_movie, tab_hype, tab_cast = st.tabs(["Movie", "Hype", "Cast & Trends"])

    with tab_movie:
        budget = st.slider("Budget ($M)", 1, 300, 80)
        popularity_choice = st.selectbox("Buzz level", list(POPULARITY), index=3)
        mdb_pop = POPULARITY[popularity_choice]
        runtime = st.slider("Runtime (min)", 80, 200, 120)
        is_major = st.checkbox("Major studio", value=True)
        release_month = st.selectbox("Release month", list(range(1, 13)), index=5)
        is_peak = st.checkbox("Peak season", value=True)
        genre = st.selectbox("Genre",
                             ["Action Franchise", "Animation/Family", "Horror", "Prestige", "Drama/Comedy/Other"])
        rating = st.selectbox("MPAA rating", ["G", "PG", "PG-13", "R"], index=2)
        ip_tier = st.selectbox("IP tier",
                               ["Original", "Niche IP", "Moderate IP", "High-Profile IP"])
        horizon = st.selectbox("Horizon", ["-14 days", "-7 days", "-3 days"], index=1)

    with tab_hype:
        tmdb_d14 = st.number_input("TMDB D-14 popularity", 0.0, 100.0, 15.0)
        tmdb_d7 = st.number_input("TMDB D-7 popularity", 0.0, 200.0, 20.0)
        momentum = tmdb_d7 / tmdb_d14 if tmdb_d14 > 0 else 1.0
        st.caption(f"D7/D14 momentum: **{momentum:.2f}**")
        yt_comments = st.number_input("YouTube comments", 0, 500_000, 50_000)
        sentiment = st.slider("Sentiment", -1.0, 1.0, 0.3)

    with tab_cast:
        star_choice = st.selectbox("Cast level", list(STARS), index=3)
        max_star, avg_star = STARS[star_choice]
        trends_choice = st.selectbox("Trends level", list(TRENDS_EX), index=2)
        rolling_7d, velocity = TRENDS_EX[trends_choice]
        predecessor_ow = st.number_input("Predecessor OW ($M)", 0.0, 500.0, 0.0)

with results_col:
    st.subheader("Prediction")
    if not model_loaded:
        st.warning("Model not loaded — showing rule-of-thumb fallback.")
        st.stop()

    genre_flags = {g: 1 if genre == label else 0 for g, label in [
        ("GENRE_ACTION_FRANCHISE", "Action Franchise"),
        ("GENRE_ANIMATION_FAMILY", "Animation/Family"),
        ("GENRE_HORROR", "Horror"), ("GENRE_PRESTIGE", "Prestige"),
        ("GENRE_ORIGINAL", "Drama/Comedy/Other"),
    ]}
    rating_flags = {f"RATING_{r.replace('-', '')}": 1 if rating == r else 0 for r in ["G", "PG", "PG-13", "R"]}
    ip_order = ["Original", "Niche IP", "Moderate IP", "High-Profile IP"]
    ip_flags = {
        "IP_ORIGINAL": int(ip_tier == "Original"),
        "IP_NICHE": int(ip_tier == "Niche IP"),
        "IP_MODERATE": int(ip_tier == "Moderate IP"),
        "IP_HIGH_PROFILE": int(ip_tier == "High-Profile IP"),
        "KNOWN_IP_TIER": ip_order.index(ip_tier),
    }

    days_out = int(horizon.split()[0])
    horizon_model = model["models"][days_out]

    static_features = [
        yt_comments, sentiment / 10, sentiment,
        0.4, 0.3, 0.2, 0.1,
        release_month, int(is_peak),
        genre_flags["GENRE_ACTION_FRANCHISE"], genre_flags["GENRE_ANIMATION_FAMILY"],
        genre_flags["GENRE_HORROR"], genre_flags["GENRE_PRESTIGE"], genre_flags["GENRE_ORIGINAL"],
        rating_flags["RATING_G"], rating_flags["RATING_PG"],
        rating_flags["RATING_PG13"], rating_flags["RATING_R"],
        ip_flags["KNOWN_IP_TIER"], ip_flags["IP_HIGH_PROFILE"],
        ip_flags["IP_MODERATE"], ip_flags["IP_NICHE"], ip_flags["IP_ORIGINAL"],
        max_star, (max_star + avg_star) / 2, avg_star, 3,
        budget * 1e6, runtime, mdb_pop, np.log1p(budget * 1e6),
        np.log1p(predecessor_ow * 1e6),
        tmdb_d14, tmdb_d7, momentum,
        int(is_major),
    ]
    trends_features = [
        rolling_7d * 0.8, rolling_7d * 0.9, rolling_7d,
        rolling_7d * 0.7, rolling_7d * 0.8, rolling_7d * 0.85,
        velocity * 0.7, velocity * 0.85, velocity,
        rolling_7d * 30, 15, rolling_7d * 1.2, 20,
        rolling_7d * ip_flags["IP_HIGH_PROFILE"],
        rolling_7d * genre_flags["GENRE_ACTION_FRANCHISE"],
        rolling_7d * genre_flags["GENRE_HORROR"],
        sentiment * rolling_7d, max_star * rolling_7d / 100,
        max_star * ip_flags["IP_HIGH_PROFILE"], 0.1 * rolling_7d,
        rolling_7d * 0.95, rolling_7d * 0.92, rolling_7d * 0.6,
    ]
    X_static = np.array(static_features, dtype=float)
    X_full = np.array(static_features + trends_features, dtype=float)

    is_small = int(horizon_model["stage1_classifier"].predict([X_static])[0])
    stage1_proba = horizon_model["stage1_classifier"].predict_proba([X_static])[0]

    if is_small == 1:
        model_tier = 0
        stage2_proba = [0.5, 0.5]
    else:
        upper = int(horizon_model["stage2_classifier"].predict([X_static])[0])
        stage2_proba = horizon_model["stage2_classifier"].predict_proba([X_static])[0]
        model_tier = upper + 1

    final_tier = apply_override(tmdb_d14, tmdb_d7, model_tier)
    overridden = final_tier != model_tier

    ow_pred = float(np.exp(horizon_model["regressors"][final_tier].predict([X_full])[0]))
    tier_name = model["tier_names"][final_tier]
    model_tier_name = model["tier_names"][model_tier]

    tier_ranges = {"SMALL": "$0–15M", "MID": "$15–50M", "LARGE+": "$50M+"}
    mae_by_tier = {"SMALL": 4.11, "MID": 12.76, "LARGE+": 31.75}

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted tier", tier_name)
    c2.metric("Predicted OW", f"${ow_pred:.1f}M")
    c3.metric("Tier range", tier_ranges[tier_name])

    if overridden:
        reason = (
            f"TMDB D14={tmdb_d14:.1f} ≥ 25 → forced LARGE+" if tmdb_d14 >= 25
            else f"TMDB D14={tmdb_d14:.1f} ≥ 15, momentum={momentum:.2f} ≥ 1.3 → forced MID"
        )
        st.warning(f"**Rule C active** — model said {model_tier_name}, override → **{tier_name}** ({reason}).")

    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="gauge+number", value=stage1_proba[0] * 100,
        title={"text": "Stage 1: NON-SMALL probability"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2ca02c"},
               "steps": [{"range": [0, 50], "color": TIER_COLORS["SMALL"]},
                         {"range": [50, 100], "color": "#ff7f0e"}],
               "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 50}},
        domain={"x": [0, 0.45], "y": [0.1, 1]},
    ))
    if is_small == 0:
        fig.add_trace(go.Indicator(
            mode="gauge+number", value=stage2_proba[1] * 100,
            title={"text": "Stage 2: LARGE+ probability"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#ff7f0e"},
                   "steps": [{"range": [0, 50], "color": TIER_COLORS["MID"]},
                             {"range": [50, 100], "color": TIER_COLORS["LARGE+"]}],
                   "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 50}},
            domain={"x": [0.55, 1], "y": [0.1, 1]},
        ))
    fig.update_layout(height=280, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    section("Confidence range")
    mae = mae_by_tier[tier_name]
    fig_r = go.Figure()
    fig_r.add_trace(go.Bar(
        x=[ow_pred], y=["Prediction"], orientation="h",
        marker_color=TIER_COLORS[tier_name],
        error_x=dict(type="data", array=[mae], arrayminus=[min(mae, ow_pred)]),
        text=[f"${ow_pred:.1f}M"], textposition="outside",
    ))
    fig_r.update_layout(height=140, xaxis_title="Opening weekend ($M)",
                        showlegend=False, xaxis_range=[0, max((ow_pred + mae) * 1.2, 50)])
    st.plotly_chart(fig_r, use_container_width=True)
    st.caption(f"Range uses {tier_name} tier MAE = ${mae:.1f}M.")

show_cortex_badge()
