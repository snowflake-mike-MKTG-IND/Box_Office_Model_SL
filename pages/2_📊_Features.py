"""
Page 2: Feature Importance Analysis
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from cortex_badge import show_cortex_badge

st.set_page_config(page_title="Features", page_icon="📊", layout="wide")

st.title("Feature Importance Analysis")
st.subheader("Understanding What Drives V17 Predictions")

FEATURE_CATEGORIES = {
    'Google Trends': ['ROLLING_3D', 'ROLLING_5D', 'ROLLING_7D', 'ROLLING_3D_PRIOR', 'ROLLING_5D_PRIOR',
                      'ROLLING_7D_PRIOR', 'ROLLING_14D', 'ROLLING_21D', 'TRENDS_EARLIEST',
                      'VELOCITY_3D', 'VELOCITY_5D', 'VELOCITY_7D',
                      'TRENDS_CUMULATIVE', 'TRENDS_VOLATILITY', 'TRENDS_PEAK_SO_FAR', 'DAYS_WITH_DATA'],
    'Star Power': ['MAX_STAR_POWER', 'TOP2_STAR_POWER', 'AVG_STAR_POWER', 'NUM_STARS_WITH_HISTORY'],
    'Movie Attributes': ['BUDGET', 'BUDGET_LOG', 'RUNTIME', 'TMDB_POPULARITY', 'RELEASE_MONTH', 'IS_PEAK_SEASON',
                         'PREDECESSOR_OW_LOG', 'IS_MAJOR_STUDIO'],
    'YouTube/Sentiment': ['YT_COMMENTS', 'ENGAGEMENT_RATIO', 'SENTIMENT',
                          'THEATRICAL_INTENT_PCT', 'STREAMING_INTENT_PCT', 'PASS_INTENT_PCT', 'NET_INTENT_PCT'],
    'Genre': ['GENRE_ACTION_FRANCHISE', 'GENRE_ANIMATION_FAMILY', 'GENRE_HORROR', 'GENRE_PRESTIGE', 'GENRE_ORIGINAL'],
    'Rating': ['RATING_G', 'RATING_PG', 'RATING_PG13', 'RATING_R'],
    'IP/Franchise': ['KNOWN_IP_TIER', 'IP_HIGH_PROFILE', 'IP_MODERATE', 'IP_NICHE', 'IP_ORIGINAL'],
    'TMDB Daily Signals': ['TMDB_POPULARITY_D14', 'TMDB_POPULARITY_D7', 'TMDB_POP_MOMENTUM'],
    'Interactions': ['ROLLING_X_IP_HIGH', 'ROLLING_X_ACTION', 'ROLLING_X_HORROR',
                     'SENTIMENT_X_ROLLING', 'STAR_X_ROLLING', 'STAR_X_IP_HIGH', 'INTENT_X_ROLLING']
}

FEATURE_IMPORTANCE = {
    'YT_COMMENTS': 8.03, 'TOP2_STAR_POWER': 5.67, 'BUDGET_LOG': 5.23,
    'TMDB_POPULARITY': 5.19, 'STREAMING_INTENT_PCT': 4.95, 'BUDGET': 4.77,
    'AVG_STAR_POWER': 4.35, 'ENGAGEMENT_RATIO': 3.50, 'THEATRICAL_INTENT_PCT': 3.44,
    'RELEASE_MONTH': 3.41, 'MAX_STAR_POWER': 3.35, 'PASS_INTENT_PCT': 3.18,
    'KNOWN_IP_TIER': 3.08, 'RUNTIME': 3.07, 'IS_MAJOR_STUDIO': 2.97,
    'NET_INTENT_PCT': 2.97, 'SENTIMENT': 2.91, 'GENRE_ORIGINAL': 1.95,
    'PREDECESSOR_OW_LOG': 1.80, 'VELOCITY_7D': 1.60, 'GENRE_ACTION_FRANCHISE': 1.51,
    'IP_HIGH_PROFILE': 1.26, 'VELOCITY_3D': 1.04, 'TRENDS_VOLATILITY': 0.99,
    'INTENT_X_ROLLING': 0.98, 'RATING_R': 0.96, 'TRENDS_EARLIEST': 0.95,
    'ROLLING_5D': 0.95, 'ROLLING_5D_PRIOR': 0.92, 'VELOCITY_5D': 0.91,
    'IS_PEAK_SEASON': 0.83, 'STAR_X_ROLLING': 0.80, 'ROLLING_3D': 0.78,
    'ROLLING_X_ACTION': 0.76, 'SENTIMENT_X_ROLLING': 0.74, 'ROLLING_3D_PRIOR': 0.74,
    'ROLLING_7D_PRIOR': 0.73, 'ROLLING_14D': 0.72, 'TRENDS_PEAK_SO_FAR': 0.69,
    'RATING_PG13': 0.69, 'NUM_STARS_WITH_HISTORY': 0.64, 'STAR_X_IP_HIGH': 0.62,
    'ROLLING_21D': 0.58, 'TRENDS_CUMULATIVE': 0.56, 'IP_ORIGINAL': 0.51,
    'ROLLING_X_IP_HIGH': 0.48, 'ROLLING_X_HORROR': 0.45, 'GENRE_HORROR': 0.43,
    'IP_MODERATE': 0.42, 'GENRE_PRESTIGE': 0.39, 'ROLLING_7D': 0.37,
    'TMDB_POPULARITY_D7': 0.36, 'TMDB_POP_MOMENTUM': 0.33, 'TMDB_POPULARITY_D14': 0.23,
    'IP_NICHE': 0.17, 'RATING_PG': 0.06, 'DAYS_WITH_DATA': 0.03,
    'GENRE_ANIMATION_FAMILY': 0.00, 'RATING_G': 0.00,
}

st.divider()

st.header("Top 20 Features")
st.caption("Weighted-average CatBoost importance across all 5 model components (2 classifiers + 3 regressors) at -7d horizon")

top_20 = dict(sorted(FEATURE_IMPORTANCE.items(), key=lambda x: x[1], reverse=True)[:20])
df_top = pd.DataFrame({'Feature': list(top_20.keys()), 'Importance': list(top_20.values())})

fig = px.bar(df_top, x='Importance', y='Feature', orientation='h',
             color='Importance', color_continuous_scale='Blues')
fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
fig.update_traces(texttemplate='%{x:.1f}', textposition='outside')
st.plotly_chart(fig, use_container_width=True)

st.info(
    "**V17 New Features**: `TRENDS_EARLIEST` (avg trends days -21 to -15) ranks #27 overall (0.95) and "
    "#3 in the LARGE+ regressor (4.86). `ROLLING_14D` (0.72) and `ROLLING_21D` (0.58) provide longer "
    "lookback windows that improve early-horizon predictions. ROLLING_21D ranks #4 in the SMALL regressor "
    "at -14d (importance=3.93), confirming the model uses these new features meaningfully."
)

st.divider()

st.header("Feature Importance by Category")

category_totals = {}
for cat, features in FEATURE_CATEGORIES.items():
    total = sum(FEATURE_IMPORTANCE.get(f, 0) for f in features)
    category_totals[cat] = total

df_cat = pd.DataFrame({'Category': list(category_totals.keys()),
                       'Total Importance': list(category_totals.values())})
df_cat = df_cat.sort_values('Total Importance', ascending=True)

fig_cat = px.bar(df_cat, x='Total Importance', y='Category', orientation='h',
                 color='Total Importance', color_continuous_scale='Viridis')
fig_cat.update_layout(height=400)
st.plotly_chart(fig_cat, use_container_width=True)

st.divider()

st.header("Drill-Down by Category")

selected_cat = st.selectbox("Select Category to Explore", list(FEATURE_CATEGORIES.keys()))

cat_features = FEATURE_CATEGORIES[selected_cat]
cat_importance = {f: FEATURE_IMPORTANCE.get(f, 0) for f in cat_features}
df_drill = pd.DataFrame({'Feature': list(cat_importance.keys()),
                         'Importance': list(cat_importance.values())})
df_drill = df_drill.sort_values('Importance', ascending=True)

col1, col2 = st.columns([2, 1])

with col1:
    fig_drill = px.bar(df_drill, x='Importance', y='Feature', orientation='h',
                       color='Importance', color_continuous_scale='Oranges')
    fig_drill.update_layout(height=max(300, len(cat_features) * 40))
    st.plotly_chart(fig_drill, use_container_width=True)

with col2:
    st.subheader(f"{selected_cat} Features")
    st.markdown(f"**Count**: {len(cat_features)} features")
    st.markdown(f"**Total Importance**: {sum(cat_importance.values()):.1f}")
    st.markdown(f"**% of Total**: {sum(cat_importance.values()) / sum(FEATURE_IMPORTANCE.values()) * 100:.1f}%")

    st.divider()
    st.markdown("**Feature Descriptions:**")

    descriptions = {
        'Google Trends': "Search interest over time, rolling averages (3D-21D), velocities, and cumulative signals. V17 added ROLLING_14D, ROLLING_21D, TRENDS_EARLIEST",
        'Star Power': "Historical box office performance of cast members",
        'Movie Attributes': "Budget, runtime, popularity score, release timing, predecessor OW, and studio tier",
        'YouTube/Sentiment': "Trailer engagement, sentiment analysis, and audience intent signals (all derived from YouTube comments)",
        'Genre': "Primary genre classification flags",
        'Rating': "MPAA rating one-hot encoding",
        'IP/Franchise': "Intellectual property tier (original vs sequel vs high-profile)",
        'TMDB Daily Signals': "TMDB popularity at day -14 and -7 before release, plus momentum (D7/D14 ratio). Primarily powers the Rule C override",
        'Interactions': "Cross-feature interactions (e.g., Trends x IP tier)"
    }
    st.info(descriptions.get(selected_cat, ""))

st.divider()

st.header("Key Insights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Most Predictive Categories")
    st.markdown("""
    1. **YouTube/Sentiment** (25.9) - YT_COMMENTS dominates at 8.03; intent signals strong
    2. **Movie Attributes** (24.8) - Budget, TMDB popularity, runtime, studio tier
    3. **Star Power** (14.0) - TOP2_STAR_POWER #2 overall at 5.67
    4. **Google Trends** (12.8) - Velocity, rolling averages (now including 14D/21D windows)
    """)

with col2:
    st.subheader("V17 Additions")
    st.markdown("""
    - **ROLLING_14D** (#38, 0.72) — 14-day lookback average; improves early-horizon predictions
    - **ROLLING_21D** (#43, 0.58) — 21-day lookback average; ranks #4 in SMALL regressor at -14d
    - **TRENDS_EARLIEST** (#27, 0.95) — Average trends from days -21 to -15; strong in LARGE+ regressor (4.86)
    - V16 features retained: IS_MAJOR_STUDIO (#15, 2.97), TMDB D14/D7/Momentum
    """)

st.divider()

st.header("Feature Availability by Time Horizon")

st.markdown("""
| Feature Type | -14 days | -7 days | -3 days |
|--------------|----------|---------|---------|
| Static (Budget, Genre, Cast, Predecessor OW, Studio) | ✅ | ✅ | ✅ |
| Google Trends (Rolling) | ✅ | ✅ | ✅ |
| YouTube Engagement | ✅ | ✅ | ✅ |
| TMDB Popularity D14 | ✅ | ✅ | ✅ |
| TMDB Popularity D7 | ❌ | ✅ | ✅ |
| TMDB Momentum (D7/D14) | ❌ | ✅ | ✅ |
| Intent Signals | ⚠️ Partial | ✅ | ✅ |

**Note**: At the -14d horizon, only TMDB D14 is available for the override system. 
D7 becomes available at -7d, enabling the full momentum calculation.
""")

st.divider()

st.info(
    "**AI-Assisted Feature Discovery**: Cortex Code helped identify and engineer all 59 features — "
    "from building the Google Trends normalization pipeline (scaled against Movie Showtimes baseline) "
    "to creating star power metrics from actor box office history to scoring YouTube trailer sentiment. "
    "The V17 rolling window features (14D, 21D, EARLIEST) were designed to improve early-horizon predictions."
)

show_cortex_badge()
