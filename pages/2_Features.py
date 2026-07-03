"""Page 2: Feature importance — V28-B horizon-normalized classifier
(static + GT percentiles + Wiki percentiles + interactions + DAYS_OUT)."""
import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from theme import apply_page_config, page_header, section, show_cortex_badge

apply_page_config("Features", icon="📊")

page_header(
    "Feature Importance",
    "V28-B classifier importances — now using **horizon-relative percentiles** for demand features. "
    "A film's Google Trends and Wikipedia signals are compared to other films at the *same stage* (D-14 vs D-14, D-7 vs D-7).",
)


@st.cache_data
def load_importance():
    p = os.path.join(os.path.dirname(__file__), "..", "data", "feature_importance_v28.json")
    with open(p) as f:
        return json.load(f)


_imp = load_importance()
# Base features: list of {feature, importance} dicts → convert to dict
_base_list = _imp["base"]
FEATURE_IMPORTANCE = {item["feature"]: item["importance"] for item in _base_list}
# Percentile features tab
META_IMPORTANCE = _imp["meta"]
N_FILMS = 310

# Categories matching the V28-B 52-feature classifier
FEATURE_CATEGORIES = {
    "Star Power": ["MAX_STAR_POWER", "TOP2_STAR_POWER", "AVG_STAR_POWER", "NUM_STARS_WITH_HISTORY"],
    "Movie Attributes": [
        "BUDGET", "BUDGET_LOG", "RUNTIME", "TMDB_POPULARITY", "RELEASE_MONTH",
        "IS_PEAK_SEASON", "PREDECESSOR_OW_LOG", "IS_MAJOR_STUDIO",
    ],
    "YouTube/Intent": [
        "YT_COMMENTS", "ENGAGEMENT_RATIO", "SENTIMENT",
        "THEATRICAL_INTENT_PCT", "STREAMING_INTENT_PCT", "PASS_INTENT_PCT", "NET_INTENT_PCT",
    ],
    "Genre": ["GENRE_ACTION_FRANCHISE", "GENRE_ANIMATION_FAMILY", "GENRE_HORROR", "GENRE_PRESTIGE", "GENRE_ORIGINAL"],
    "Rating": ["RATING_G", "RATING_PG", "RATING_PG13", "RATING_R"],
    "IP/Franchise": ["KNOWN_IP_TIER", "IP_HIGH_PROFILE", "IP_MODERATE", "IP_NICHE", "IP_ORIGINAL"],
    "TMDB Snapshots": ["TMDB_POPULARITY_D14", "TMDB_POPULARITY_D7", "TMDB_POP_MOMENTUM"],
    "GT Demand Percentiles": [
        "GT_3Day_Demand_Percentile", "GT_7Day_Demand_Percentile", "GT_14Day_Demand_Percentile",
        "GT_Peak_Demand_Percentile", "GT_Acceleration_Percentile", "GT_Trend_Slope_Percentile",
    ],
    "Wiki Percentiles": [
        "Wiki_Total_Views_Percentile", "Wiki_7Day_Views_Percentile", "Wiki_3Day_Views_Percentile",
        "Wiki_14Day_Views_Percentile", "Wiki_Peak_Day_Percentile", "Wiki_Velocity_Percentile",
    ],
    "Percentile Interactions": [
        "GT_Demand_x_Franchise_IP", "GT_Demand_x_Star_Power", "GT_Demand_x_Budget",
    ],
}

tab_top, tab_category, tab_drill, tab_meta, tab_availability = st.tabs(
    ["Top 20 features", "By category", "Drill-down", "Percentile features", "Availability by horizon"]
)

with tab_top:
    top20 = dict(sorted(FEATURE_IMPORTANCE.items(), key=lambda x: x[1], reverse=True)[:20])
    df_top = pd.DataFrame({"Feature": list(top20), "Importance": list(top20.values())})
    fig = px.bar(df_top, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="Blues")
    fig.update_layout(height=560, yaxis={"categoryorder": "total ascending"})
    fig.update_traces(texttemplate="%{x:.1f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"V28-B tier classifier (CatBoost multiclass), importances across {N_FILMS} films × 3 horizons (928 rows). "
        "Top signals: **YouTube comments**, TMDB popularity, **star power**, predecessor OW, and the "
        "**GT Demand × Franchise IP** interaction — demand and pedigree together, not budget alone."
    )
    st.info(
        "**Horizon-normalized demand:** Google Trends and Wikipedia features are converted to percentile ranks "
        "within their scoring horizon (D-14, D-7, or D-3) before the classifier sees them. A film at the 70th "
        "percentile of demand at D-14 is treated the same as one at the 70th percentile at D-7 — even though "
        "the raw numbers differ by ~2×. This fixes systematic underestimation at early horizons."
    )

with tab_category:
    totals = {cat: sum(FEATURE_IMPORTANCE.get(f, 0) for f in feats)
              for cat, feats in FEATURE_CATEGORIES.items()}
    df_cat = pd.DataFrame({"Category": list(totals), "Total Importance": list(totals.values())})
    df_cat = df_cat.sort_values("Total Importance", ascending=True)
    fig_cat = px.bar(df_cat, x="Total Importance", y="Category", orientation="h",
                     color="Total Importance", color_continuous_scale="Viridis")
    fig_cat.update_layout(height=420)
    st.plotly_chart(fig_cat, use_container_width=True)
    st.caption(
        "GT Demand Percentiles, Wiki Percentiles, and their interactions now form distinct category groups. "
        "YouTube/Intent and Star Power dominate individual feature importance, but the combined percentile "
        "categories contribute significant signal for tier classification."
    )

with tab_drill:
    selected = st.selectbox("Select a category", list(FEATURE_CATEGORIES))
    feats = FEATURE_CATEGORIES[selected]
    df_drill = pd.DataFrame({
        "Feature": feats,
        "Importance": [FEATURE_IMPORTANCE.get(f, 0) for f in feats],
    }).sort_values("Importance", ascending=True)
    fig = px.bar(df_drill, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="Oranges")
    fig.update_layout(height=max(260, len(feats) * 32))
    st.plotly_chart(fig, use_container_width=True)
    total = df_drill["Importance"].sum()
    share = total / sum(FEATURE_IMPORTANCE.values()) * 100
    st.caption(f"{len(feats)} features · total importance {total:.1f} · {share:.1f}% of classifier total")

with tab_meta:
    section("Horizon-normalized percentile features")
    st.markdown(
        "V28-B converts demand features (Google Trends + Wikipedia) to **horizon-relative percentiles** before "
        "the classifier sees them. A film's R7D at D-14 is compared to other films at D-14 — so 'strong for this stage' "
        "is recognized regardless of when the prediction runs. These 12 percentile features (6 GT + 6 Wiki) plus "
        "3 percentile interactions replace absolute demand values in the classifier."
    )
    df_meta = pd.DataFrame({"Feature": list(META_IMPORTANCE), "Importance": list(META_IMPORTANCE.values())})
    df_meta = df_meta.sort_values("Importance", ascending=True)
    fig_m = px.bar(df_meta, x="Importance", y="Feature", orientation="h",
                   color="Importance", color_continuous_scale="Teal")
    fig_m.update_layout(height=480)
    fig_m.update_traces(texttemplate="%{x:.1f}", textposition="outside")
    st.plotly_chart(fig_m, use_container_width=True)
    st.caption(
        "GT_Demand_x_Franchise_IP is the strongest percentile feature — it captures 'this is a franchise film "
        "AND its demand signal is strong for this stage.' Wiki_Total_Views_Percentile at #2 shows cumulative "
        "Wikipedia interest matters more than any single rolling window."
    )

with tab_availability:
    st.markdown(
        """
        | Feature family | -14d | -7d | -3d |
        |---|---|---|---|
        | Static (Budget, Genre, Cast, Predecessor OW, Studio) | ✅ | ✅ | ✅ |
        | GT Demand Percentiles (horizon-normalized) | ✅ | ✅ | ✅ |
        | Wiki Percentiles (horizon-normalized) | ✅ | ✅ | ✅ |
        | YouTube engagement + intent | ✅ | ✅ | ✅ |
        | TMDB D14 snapshot | ✅ | ✅ | ✅ |
        | TMDB D7 + momentum | ❌ | ✅ | ✅ |
        | DAYS_OUT (explicit horizon feature) | ✅ | ✅ | ✅ |
        """
    )
    st.caption(
        "V28-B trains on all three horizons simultaneously (928 rows). DAYS_OUT as an explicit feature lets "
        "the model learn residual horizon effects beyond what the percentile transform captures. "
        "Earlier horizons have wider confidence intervals but the same classification accuracy target."
    )

show_cortex_badge()
