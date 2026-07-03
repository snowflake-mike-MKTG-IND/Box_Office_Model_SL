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
FEATURE_IMPORTANCE = _imp["base"]          # base CatBoost classifier over STATIC_WIKI + OOF_STACK
META_IMPORTANCE = _imp["meta"]             # learned combiner g over its 7 meta-features
N_FILMS = _imp.get("n_films", 291)

# Categories rebuilt to match the actual V28-A STATIC_WIKI feature set + the stacked point.
FEATURE_CATEGORIES = {
    "Stacked signal": ["OOF_STACK"],
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
    "Google Trends": ["ROLLING_3D", "ROLLING_7D", "ROLLING_14D", "TRENDS_PEAK_SO_FAR", "VEL_3V7", "LOG_SLOPE_14_TO_3"],
    "Wikipedia": [
        "WIKI_7D_LOG", "WIKI_14D_LOG", "WIKI_ROLLING_3D", "WIKI_ROLLING_7D", "WIKI_ROLLING_14D",
        "WIKI_ROLLING_7D_PRIOR", "WIKI_VELOCITY_7D", "WIKI_PEAK", "WIKI_PEAK_LOG",
        "WIKI_CUMULATIVE", "WIKI_CUMULATIVE_LOG", "WIKI_DAYS_WITH_DATA",
    ],
}

tab_top, tab_category, tab_drill, tab_meta, tab_availability = st.tabs(
    ["Top 20 features", "By category", "Drill-down", "Meta-combiner inputs", "Availability by horizon"]
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
        f"V28-A base tier classifier (CatBoost), importances normalized to 100 over {N_FILMS} films at **-14d**. "
        "The **stacked OOF point** is the single strongest input, followed by star power, **YouTube comments**, "
        "predecessor OW and TMDB popularity — demand and pedigree, not budget alone."
    )
    st.info(
        "**Why -14d?** Two weeks out is where an early call is most valuable — and hardest. The live Google-Trends "
        "spike hasn't formed yet, so the model leans more on signals that are already meaningful that early: the "
        "stacked OOF point, star power, **YouTube comment volume** and predecessor OW. The short-window Trends "
        "features (ROLLING_3D/14D) carry more weight closer to release. These are the **classifier's** inputs; the "
        "three per-tier $ regressors additionally use the full Trends windows, and a separate quantile pool "
        "produces the conformal bear/base/bull bands."
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
        "Wikipedia and Google-Trends families concentrate demand signal across many small features, so their "
        "category totals are larger than any single feature suggests. Genre and rating contribute little to the "
        "classifier directly — the model reads demand, not labels."
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
    section("What the rule-free combiner leans on")
    st.markdown(
        "V28-A replaced the hand-coded demand rules with a learned meta-combiner **g** — a small CatBoost "
        "regressor over **7 meta-features** built from the base layer: the log of each tier's $ point estimate, "
        "the three class probabilities, and the log of the soft mixture (Σ prob·point). Its importances show "
        "the model trusting the **mixture and the class-probability distribution** above any single tier point."
    )
    df_meta = pd.DataFrame({"Meta-feature": list(META_IMPORTANCE), "Importance": list(META_IMPORTANCE.values())})
    df_meta = df_meta.sort_values("Importance", ascending=True)
    fig_m = px.bar(df_meta, x="Importance", y="Meta-feature", orientation="h",
                   color="Importance", color_continuous_scale="Teal")
    fig_m.update_layout(height=360)
    fig_m.update_traces(texttemplate="%{x:.1f}", textposition="outside")
    st.plotly_chart(fig_m, use_container_width=True)
    st.caption(
        "Final OW = 0.7·exp(g) + 0.3·mixture. The combiner is fit on inner-OOF base predictions inside each "
        "outer fold, so it never sees a film whose base predictions trained it (leakage-safe nested stacking)."
    )

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
        "Earlier horizons carry less daily demand signal, so predictions at -14d are wider and the model "
        "leans more on static pedigree; by -7d the full TMDB momentum and intent signals are present."
    )

show_cortex_badge()
