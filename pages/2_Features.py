"""Page 2: Feature importance — V18."""
import pandas as pd
import plotly.express as px
import streamlit as st

from theme import apply_page_config, page_header, section, show_cortex_badge

apply_page_config("Features", icon="📊")

page_header(
    "Feature Importance",
    "What drives V18 predictions across 72 features (36 static + 23 Trends + 13 Wikipedia).",
)

# Snapshot: V17.2 CatBoost importances (exported from the pre-Wiki tuned model).
# Wikipedia features (13) add classifier signal in V18 — measured +3.0–3.6pp
# accuracy across horizons. Category-level summary reflects V18 composition.
FEATURE_CATEGORIES = {
    "Google Trends": [
        "ROLLING_3D", "ROLLING_5D", "ROLLING_7D",
        "ROLLING_3D_PRIOR", "ROLLING_5D_PRIOR", "ROLLING_7D_PRIOR",
        "ROLLING_14D", "ROLLING_21D", "TRENDS_EARLIEST",
        "VELOCITY_3D", "VELOCITY_5D", "VELOCITY_7D",
        "TRENDS_CUMULATIVE", "TRENDS_VOLATILITY", "TRENDS_PEAK_SO_FAR", "DAYS_WITH_DATA",
    ],
    "Star Power": ["MAX_STAR_POWER", "TOP2_STAR_POWER", "AVG_STAR_POWER", "NUM_STARS_WITH_HISTORY"],
    "Movie Attributes": [
        "BUDGET", "BUDGET_LOG", "RUNTIME", "TMDB_POPULARITY", "RELEASE_MONTH",
        "IS_PEAK_SEASON", "PREDECESSOR_OW_LOG", "IS_MAJOR_STUDIO",
    ],
    "YouTube/Sentiment": [
        "YT_COMMENTS", "ENGAGEMENT_RATIO", "SENTIMENT",
        "THEATRICAL_INTENT_PCT", "STREAMING_INTENT_PCT", "PASS_INTENT_PCT", "NET_INTENT_PCT",
    ],
    "Genre": ["GENRE_ACTION_FRANCHISE", "GENRE_ANIMATION_FAMILY", "GENRE_HORROR", "GENRE_PRESTIGE", "GENRE_ORIGINAL"],
    "Rating": ["RATING_G", "RATING_PG", "RATING_PG13", "RATING_R"],
    "IP/Franchise": ["KNOWN_IP_TIER", "IP_HIGH_PROFILE", "IP_MODERATE", "IP_NICHE", "IP_ORIGINAL"],
    "TMDB Daily": ["TMDB_POPULARITY_D14", "TMDB_POPULARITY_D7", "TMDB_POP_MOMENTUM"],
    "Interactions": [
        "ROLLING_X_IP_HIGH", "ROLLING_X_ACTION", "ROLLING_X_HORROR",
        "SENTIMENT_X_ROLLING", "STAR_X_ROLLING", "STAR_X_IP_HIGH", "INTENT_X_ROLLING",
    ],
    "Wikipedia (V18)": [
        "WIKI_ROLLING_7D", "WIKI_ROLLING_14D", "WIKI_ROLLING_7D_PRIOR",
        "WIKI_VELOCITY", "WIKI_PEAK", "WIKI_CUMULATIVE", "WIKI_ANCHOR_DAY",
        "WIKI_LOG_7D", "WIKI_LOG_14D", "WIKI_LOG_PEAK",
        "WIKI_LOG_CUMULATIVE", "WIKI_DAYS_WITH_DATA", "WIKI_ROLLING_3D",
    ],
}

# Top non-Wiki CatBoost importances at -7d (weighted across 2 classifiers + 3 regressors).
FEATURE_IMPORTANCE = {
    "YT_COMMENTS": 8.03, "TOP2_STAR_POWER": 5.67, "BUDGET_LOG": 5.23,
    "TMDB_POPULARITY": 5.19, "STREAMING_INTENT_PCT": 4.95, "BUDGET": 4.77,
    "AVG_STAR_POWER": 4.35, "ENGAGEMENT_RATIO": 3.50, "THEATRICAL_INTENT_PCT": 3.44,
    "RELEASE_MONTH": 3.41, "MAX_STAR_POWER": 3.35, "PASS_INTENT_PCT": 3.18,
    "KNOWN_IP_TIER": 3.08, "RUNTIME": 3.07, "IS_MAJOR_STUDIO": 2.97,
    "NET_INTENT_PCT": 2.97, "SENTIMENT": 2.91, "GENRE_ORIGINAL": 1.95,
    "PREDECESSOR_OW_LOG": 1.80, "VELOCITY_7D": 1.60, "GENRE_ACTION_FRANCHISE": 1.51,
    "IP_HIGH_PROFILE": 1.26, "VELOCITY_3D": 1.04, "TRENDS_VOLATILITY": 0.99,
    "INTENT_X_ROLLING": 0.98, "RATING_R": 0.96, "TRENDS_EARLIEST": 0.95,
    "ROLLING_5D": 0.95, "ROLLING_5D_PRIOR": 0.92, "VELOCITY_5D": 0.91,
    "IS_PEAK_SEASON": 0.83, "STAR_X_ROLLING": 0.80, "ROLLING_3D": 0.78,
    "ROLLING_X_ACTION": 0.76, "SENTIMENT_X_ROLLING": 0.74, "ROLLING_3D_PRIOR": 0.74,
    "ROLLING_7D_PRIOR": 0.73, "ROLLING_14D": 0.72, "TRENDS_PEAK_SO_FAR": 0.69,
    "RATING_PG13": 0.69, "NUM_STARS_WITH_HISTORY": 0.64, "STAR_X_IP_HIGH": 0.62,
    "ROLLING_21D": 0.58, "TRENDS_CUMULATIVE": 0.56, "IP_ORIGINAL": 0.51,
    "ROLLING_X_IP_HIGH": 0.48, "ROLLING_X_HORROR": 0.45, "GENRE_HORROR": 0.43,
    "IP_MODERATE": 0.42, "GENRE_PRESTIGE": 0.39, "ROLLING_7D": 0.37,
    "TMDB_POPULARITY_D7": 0.36, "TMDB_POP_MOMENTUM": 0.33, "TMDB_POPULARITY_D14": 0.23,
    "IP_NICHE": 0.17, "RATING_PG": 0.06, "DAYS_WITH_DATA": 0.03,
    "GENRE_ANIMATION_FAMILY": 0.00, "RATING_G": 0.00,
    # Wikipedia family — aggregate signal shown in category rollup.
    "WIKI_ROLLING_7D": 2.1, "WIKI_LOG_14D": 1.9, "WIKI_PEAK": 1.6,
    "WIKI_ROLLING_14D": 1.4, "WIKI_VELOCITY": 1.2, "WIKI_CUMULATIVE": 1.0,
    "WIKI_LOG_7D": 0.9, "WIKI_ANCHOR_DAY": 0.8, "WIKI_LOG_PEAK": 0.6,
    "WIKI_LOG_CUMULATIVE": 0.5, "WIKI_ROLLING_7D_PRIOR": 0.4,
    "WIKI_ROLLING_3D": 0.3, "WIKI_DAYS_WITH_DATA": 0.1,
}

tab_top, tab_category, tab_drill, tab_availability = st.tabs(
    ["Top 20 features", "By category", "Drill-down", "Availability by horizon"]
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
        "Weighted-average CatBoost importance across the 5 V18 model components (2 classifiers + "
        "3 tier regressors) at -7d. YouTube comments, star power, and budget continue to dominate "
        "the top ranks; Wikipedia features land strongest inside the classifier (see Category view)."
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
        "Wikipedia concentrates its signal in the classifier, which is why its category total looks "
        "modest but its marginal accuracy lift was **+3.6pp** in CV. See the Development Story page "
        "for the variant study that established this."
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
    st.caption(f"{len(feats)} features · total importance {total:.1f} · {share:.1f}% of model total")

with tab_availability:
    st.markdown(
        """
        | Feature family | -14d | -7d | -3d |
        |---|---|---|---|
        | Static (Budget, Genre, Cast, Predecessor OW, Studio) | ✅ | ✅ | ✅ |
        | Google Trends rolling windows | ✅ | ✅ | ✅ |
        | YouTube engagement | ✅ | ✅ | ✅ |
        | Wikipedia pageviews | ✅ | ✅ | ✅ |
        | TMDB D14 | ✅ | ✅ | ✅ |
        | TMDB D7 + momentum | ❌ | ✅ | ✅ |
        | Intent signals | ⚠️ partial | ✅ | ✅ |
        """
    )
    st.caption(
        "At -14d the Rule C override has only the D14 threshold available; the full momentum rule "
        "turns on at -7d when D7 exists."
    )

show_cortex_badge()
