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
st.subheader("Understanding What Drives V16 Predictions")

FEATURE_CATEGORIES = {
    'Google Trends': ['ROLLING_3D', 'ROLLING_5D', 'ROLLING_7D', 'ROLLING_3D_PRIOR', 'ROLLING_5D_PRIOR',
                      'ROLLING_7D_PRIOR', 'VELOCITY_3D', 'VELOCITY_5D', 'VELOCITY_7D',
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
    'YT_COMMENTS': 8.36, 'TOP2_STAR_POWER': 6.05, 'TMDB_POPULARITY': 5.89,
    'BUDGET_LOG': 4.96, 'STREAMING_INTENT_PCT': 4.83, 'BUDGET': 4.34,
    'THEATRICAL_INTENT_PCT': 4.14, 'AVG_STAR_POWER': 4.13, 'PASS_INTENT_PCT': 3.65,
    'NET_INTENT_PCT': 3.32, 'IS_MAJOR_STUDIO': 3.23, 'ENGAGEMENT_RATIO': 3.07,
    'RUNTIME': 3.06, 'MAX_STAR_POWER': 2.95, 'RELEASE_MONTH': 2.76,
    'KNOWN_IP_TIER': 2.67, 'SENTIMENT': 2.57, 'GENRE_ORIGINAL': 1.65,
    'PREDECESSOR_OW_LOG': 1.65, 'RATING_R': 1.41, 'IP_HIGH_PROFILE': 1.40,
    'VELOCITY_7D': 1.38, 'GENRE_ACTION_FRANCHISE': 1.37, 'VELOCITY_3D': 1.31,
    'ROLLING_X_ACTION': 1.17, 'STAR_X_ROLLING': 1.13, 'ROLLING_7D_PRIOR': 1.11,
    'TRENDS_VOLATILITY': 0.96, 'VELOCITY_5D': 0.93, 'NUM_STARS_WITH_HISTORY': 0.92,
    'STAR_X_IP_HIGH': 0.90, 'SENTIMENT_X_ROLLING': 0.87, 'ROLLING_3D_PRIOR': 0.86,
    'INTENT_X_ROLLING': 0.82, 'ROLLING_7D': 0.79, 'TMDB_POPULARITY_D14': 0.78,
    'TRENDS_PEAK_SO_FAR': 0.69, 'ROLLING_5D_PRIOR': 0.68, 'RATING_PG13': 0.68,
    'ROLLING_5D': 0.68, 'TMDB_POPULARITY_D7': 0.65, 'TMDB_POP_MOMENTUM': 0.64,
    'GENRE_HORROR': 0.64, 'IS_PEAK_SEASON': 0.62, 'ROLLING_3D': 0.59,
    'IP_ORIGINAL': 0.54, 'ROLLING_X_IP_HIGH': 0.45, 'TRENDS_CUMULATIVE': 0.43,
    'IP_MODERATE': 0.40, 'ROLLING_X_HORROR': 0.38, 'GENRE_PRESTIGE': 0.26,
    'IP_NICHE': 0.19, 'GENRE_ANIMATION_FAMILY': 0.04, 'RATING_PG': 0.04,
    'DAYS_WITH_DATA': 0.02, 'RATING_G': 0.00,
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
    "**V16 New Features**: `IS_MAJOR_STUDIO` ranks #11 overall (3.23), strong in the MID regressor (7.28). "
    "TMDB daily popularity features (`D14`=0.78, `D7`=0.65, `MOMENTUM`=0.64) show low CatBoost importance "
    "because only ~30 training films have this data — too sparse for tree-based learning. But the raw "
    "signal is extremely strong (r=0.817), which is why it powers the Rule C override instead."
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
        'Google Trends': "Search interest over time, rolling averages and velocities",
        'Star Power': "Historical box office performance of cast members",
        'Movie Attributes': "Budget, runtime, popularity score, release timing, predecessor OW, and studio tier (V16)",
        'YouTube/Sentiment': "Trailer engagement, sentiment analysis, and audience intent signals (all derived from YouTube comments)",
        'Genre': "Primary genre classification flags",
        'Rating': "MPAA rating one-hot encoding",
        'IP/Franchise': "Intellectual property tier (original vs sequel vs high-profile)",
        'TMDB Daily Signals': "TMDB popularity at day -14 and -7 before release, plus momentum (D7/D14 ratio). New in V16 — primarily powers the Rule C override",
        'Interactions': "Cross-feature interactions (e.g., Trends x IP tier)"
    }
    st.info(descriptions.get(selected_cat, ""))

st.divider()

st.header("Key Insights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Most Predictive Categories")
    st.markdown("""
    1. **YouTube/Sentiment** (26.3) - YT_COMMENTS dominates at 8.36; intent signals strong
    2. **Movie Attributes** (24.6) - Budget, TMDB popularity, runtime, studio tier
    3. **Star Power** (14.1) - TOP2_STAR_POWER #2 overall at 6.05
    4. **Google Trends** (9.4) - Velocity and rolling averages across all horizons
    """)

with col2:
    st.subheader("V16 Additions")
    st.markdown("""
    - **IS_MAJOR_STUDIO** (#11, 3.23) — Disney, Universal, Warner Bros, Paramount, Sony, 20th Century
    - **TMDB_POPULARITY_D14** — Day -14 popularity; powers Rule C override (D14>=25 -> LARGE+)
    - **TMDB_POPULARITY_D7** — Day -7 popularity; used in momentum calculation
    - **TMDB_POP_MOMENTUM** — D7/D14 ratio; momentum gate prevents false positives
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
    "**AI-Assisted Feature Discovery**: Cortex Code helped identify and engineer all 56 features — "
    "from building the Google Trends normalization pipeline (scaled against Movie Showtimes baseline) "
    "to creating star power metrics from actor box office history to scoring YouTube trailer sentiment. "
    "The V16 TMDB features and IS_MAJOR_STUDIO were added and validated in a single Cortex Code session."
)

show_cortex_badge()
