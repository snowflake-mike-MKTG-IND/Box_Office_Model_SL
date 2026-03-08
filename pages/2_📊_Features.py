"""
Page 2: Feature Importance Analysis
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.set_page_config(page_title="Features", page_icon="📊", layout="wide")

st.title("Feature Importance Analysis")
st.subheader("Understanding What Drives V15 Predictions")

FEATURE_CATEGORIES = {
    'Google Trends': ['ROLLING_3D', 'ROLLING_5D', 'ROLLING_7D', 'ROLLING_3D_PRIOR', 'ROLLING_5D_PRIOR', 
                      'ROLLING_7D_PRIOR', 'VELOCITY_3D', 'VELOCITY_5D', 'VELOCITY_7D',
                      'TRENDS_CUMULATIVE', 'TRENDS_VOLATILITY', 'TRENDS_PEAK_SO_FAR', 'DAYS_WITH_DATA'],
    'Star Power': ['MAX_STAR_POWER', 'TOP2_STAR_POWER', 'AVG_STAR_POWER', 'NUM_STARS_WITH_HISTORY'],
    'Movie Attributes': ['BUDGET', 'BUDGET_LOG', 'RUNTIME', 'TMDB_POPULARITY', 'RELEASE_MONTH', 'IS_PEAK_SEASON', 'PREDECESSOR_OW_LOG'],
    'YouTube/Sentiment': ['YT_COMMENTS', 'ENGAGEMENT_RATIO', 'SENTIMENT',
                          'THEATRICAL_INTENT_PCT', 'STREAMING_INTENT_PCT', 'PASS_INTENT_PCT', 'NET_INTENT_PCT'],
    'Genre': ['GENRE_ACTION_FRANCHISE', 'GENRE_ANIMATION_FAMILY', 'GENRE_HORROR', 'GENRE_PRESTIGE', 'GENRE_ORIGINAL'],
    'Rating': ['RATING_G', 'RATING_PG', 'RATING_PG13', 'RATING_R'],
    'IP/Franchise': ['KNOWN_IP_TIER', 'IP_HIGH_PROFILE', 'IP_MODERATE', 'IP_NICHE', 'IP_ORIGINAL'],
    'Interactions': ['ROLLING_X_IP_HIGH', 'ROLLING_X_ACTION', 'ROLLING_X_HORROR',
                     'SENTIMENT_X_ROLLING', 'STAR_X_ROLLING', 'STAR_X_IP_HIGH', 'INTENT_X_ROLLING']
}

FEATURE_IMPORTANCE = {
    'YT_COMMENTS': 47.8, 'BUDGET': 40.2, 'BUDGET_LOG': 36.5, 'TMDB_POPULARITY': 33.8,
    'ROLLING_7D': 30.1, 'AVG_STAR_POWER': 27.9, 'TOP2_STAR_POWER': 26.3, 'PREDECESSOR_OW_LOG': 24.1,
    'MAX_STAR_POWER': 23.5, 'ROLLING_5D': 22.7, 'ROLLING_3D': 21.2, 'TRENDS_CUMULATIVE': 19.5,
    'SENTIMENT': 17.8, 'THEATRICAL_INTENT_PCT': 16.4, 'ENGAGEMENT_RATIO': 15.1,
    'GENRE_ACTION_FRANCHISE': 14.5, 'RELEASE_MONTH': 13.6, 'RUNTIME': 12.2,
    'NET_INTENT_PCT': 11.5, 'VELOCITY_7D': 10.9, 'IS_PEAK_SEASON': 10.3,
    'IP_HIGH_PROFILE': 9.7, 'KNOWN_IP_TIER': 9.2, 'ROLLING_7D_PRIOR': 8.8,
    'TRENDS_PEAK_SO_FAR': 8.4, 'STAR_X_ROLLING': 8.0, 'VELOCITY_5D': 7.6,
    'ROLLING_X_IP_HIGH': 7.2, 'STREAMING_INTENT_PCT': 6.8, 'NUM_STARS_WITH_HISTORY': 6.5,
    'GENRE_HORROR': 6.1, 'SENTIMENT_X_ROLLING': 5.7, 'TRENDS_VOLATILITY': 5.4,
    'VELOCITY_3D': 5.1, 'ROLLING_5D_PRIOR': 4.8, 'ROLLING_3D_PRIOR': 4.5,
    'GENRE_ANIMATION_FAMILY': 4.2, 'PASS_INTENT_PCT': 3.9, 'INTENT_X_ROLLING': 3.6,
    'IP_MODERATE': 3.3, 'DAYS_WITH_DATA': 3.0, 'RATING_PG13': 2.7, 'RATING_R': 2.4,
    'ROLLING_X_ACTION': 2.1, 'STAR_X_IP_HIGH': 1.9, 'GENRE_PRESTIGE': 1.7,
    'IP_NICHE': 1.5, 'RATING_PG': 1.3, 'GENRE_ORIGINAL': 1.1,
    'ROLLING_X_HORROR': 0.9, 'IP_ORIGINAL': 0.7, 'RATING_G': 0.4
}

st.divider()

st.header("Top 20 Features")
st.caption("By CatBoost importance score (V15)")

top_20 = dict(sorted(FEATURE_IMPORTANCE.items(), key=lambda x: x[1], reverse=True)[:20])
df_top = pd.DataFrame({'Feature': list(top_20.keys()), 'Importance': list(top_20.values())})

fig = px.bar(df_top, x='Importance', y='Feature', orientation='h',
             color='Importance', color_continuous_scale='Blues')
fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
fig.update_traces(texttemplate='%{x:.1f}', textposition='outside')
st.plotly_chart(fig, use_container_width=True)

st.info("**🆕 V15**: `PREDECESSOR_OW_LOG` enters at #8, providing sequel/franchise performance signal. This helps the model differentiate between original IPs and sequels with known box office history.")

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
        'Movie Attributes': "Budget, runtime, popularity score, release timing, and predecessor OW (new in V15)",
        'YouTube/Sentiment': "Trailer engagement, sentiment analysis, and audience intent signals (all derived from YouTube comments)",
        'Genre': "Primary genre classification flags",
        'Rating': "MPAA rating one-hot encoding",
        'IP/Franchise': "Intellectual property tier (original vs sequel vs high-profile)",
        'Interactions': "Cross-feature interactions (e.g., Trends × IP tier)"
    }
    st.info(descriptions.get(selected_cat, ""))

st.divider()

st.header("Key Insights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Most Predictive Categories")
    st.markdown("""
    1. **Movie Attributes** (170.7) - Budget + predecessor OW drive expectations
    2. **Google Trends** (126.1) - Real-time interest signals
    3. **YouTube/Sentiment** (104.3) - Trailer reception + intent
    4. **Star Power** (84.2) - Cast matters for draw
    """)

with col2:
    st.subheader("Surprising Findings")
    st.markdown("""
    - **YT_COMMENTS** (#1) beats all other features
    - **PREDECESSOR_OW_LOG** (#8) — new V15 feature immediately ranks high
    - **BUDGET** more important than STAR_POWER
    - **GENRE** features have low individual importance
    - **Interactions** add marginal value (~7% of total)
    """)

st.divider()

st.header("Feature Availability by Time Horizon")

st.markdown("""
| Feature Type | -14 days | -7 days | -3 days |
|--------------|----------|---------|---------|
| Static (Budget, Genre, Cast, Predecessor OW) | ✅ | ✅ | ✅ |
| Google Trends (Rolling) | ✅ | ✅ | ✅ |
| YouTube Engagement | ✅ | ✅ | ✅ |
| Intent Signals | ⚠️ Partial | ✅ | ✅ |

**Note**: All 52 features are available at all prediction horizons, but 
Google Trends rolling averages become more predictive closer to release.
""")
