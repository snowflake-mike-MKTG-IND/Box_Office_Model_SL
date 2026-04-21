"""
Page 8: Wikipedia Integration — V18 Selling Point
Iteration velocity story: from idea to production in under an hour.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from cortex_badge import show_cortex_badge

st.set_page_config(page_title="Wikipedia Integration", page_icon="📚", layout="wide")

st.title("Wikipedia Integration — V18 Data Ingestion Velocity")
st.subheader("From user question to production model in 49 minutes")

st.divider()

st.info(
    "**Why this matters**: Demonstrates how fast a data science team can add a brand new "
    "signal source to a production model on Snowflake — ingestion, feature engineering, "
    "evaluation, and hyperparameter tuning all in under an hour, with measurable accuracy lift."
)

st.header("The Ask")
st.markdown("""
> *"I still want to improve our classification accuracy. NRG Tracking, Ticket Sales, and Quorum Tracking
> are inaccessible for a demo. Does Wikipedia provide enough historical data to have full coverage over
> all our movies?"*

**Answer**: Yes. And in 49 minutes we integrated it, engineered features, tested 4 variants at 3 horizons,
retuned hyperparameters across 108 configs, and confirmed +3.6pp accuracy + -$0.39M MAE improvement.
""")

st.divider()

st.header("End-to-End Timeline (2026-04-20)")

phases = pd.DataFrame([
    {"Phase": "1. Title mapping (OpenSearch API)", "Duration (min)": 3.1, "Output": "275/277 articles mapped"},
    {"Phase": "2a. Fetch daily pageviews (Wikimedia REST API)", "Duration (min)": 1.2, "Output": "6,975 rows"},
    {"Phase": "2b. Canonical resolution (summary API redirects)", "Duration (min)": 0.55, "Output": "+28 films via redirects"},
    {"Phase": "2c-f. URL encoding + year-suffix audit + manual fixes", "Duration (min)": 11.0, "Output": "1,400+ more rows, 277/277 coverage"},
    {"Phase": "3. Feature engineering (SQL DDL)", "Duration (min)": 0.02, "Output": "13 features × 3 horizons"},
    {"Phase": "4. V18 baseline training (3 variants)", "Duration (min)": 3.3, "Output": "Variant placement confirmed"},
    {"Phase": "5. Deep evaluation across horizons + HPT (108 configs)", "Duration (min)": 11.4, "Output": "Best S1/S2 HPs identified"},
])

col1, col2 = st.columns([3, 2])

with col1:
    fig = px.bar(phases, x="Duration (min)", y="Phase", orientation='h',
                 text='Duration (min)', color='Duration (min)', color_continuous_scale='Blues')
    fig.update_traces(texttemplate='%{text:.1f} min', textposition='outside')
    fig.update_layout(height=430, title="Time per phase (cumulative: 30.6 min compute + 19 min audit = 49 min)",
                      xaxis_range=[0, 14], yaxis=dict(categoryorder='total ascending'))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.metric("Total wall-clock time", "~49 min", "from idea → tuned model")
    st.metric("Movies ingested", "277", "100% coverage")
    st.metric("Pageview rows", "8,455", "daily granularity")
    st.metric("Total pageviews loaded", "193.3M", "D-30 to D+0 per film")

st.divider()

st.header("Data Pipeline Architecture")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 1. Ingest")
    st.markdown("""
    - **Source**: Wikimedia REST API + MediaWiki Action API
    - **No third-party vendor**
    - **Rate**: ~50 req/s (polite)
    - **Window per film**: D-30 to D+0 (31 days daily)
    - **Snowflake table**: `PRODUCTION.WIKIPEDIA_PAGEVIEWS`
    """)

with col2:
    st.markdown("### 2. Resolve")
    st.markdown("""
    - **OpenSearch API** for initial matching (275/277)
    - **Summary API** for canonical redirects (+28)
    - **Year-suffix audit** for disambiguation (13 big films fixed)
    - **Manual overrides** for edge cases (`MaXXXine`, `Thunderbolts*`, `The_Bride!`)
    - **Unicode handling** (em dashes, apostrophes, ampersands, `?`, `!`, `*`)
    """)

with col3:
    st.markdown("### 3. Engineer")
    st.markdown("""
    - **13 features** per horizon (-14d, -7d, -3d)
    - Rolling windows (3d, 7d, 14d) + prior windows
    - Velocity (7d/prior_7d ratio)
    - Peak, cumulative, anchor day
    - Log-transformed versions for tree-based models
    - **View**: `PRODUCTION.WIKIPEDIA_FEATURES_V`
    """)

st.divider()

st.header("Signal Strength — Why Wikipedia Works")

col1, col2 = st.columns(2)

with col1:
    corr_data = pd.DataFrame({
        'Signal': ['Wikipedia 14d views', 'Google Trends 14d (typical)', 'YT Comments'],
        'Correlation with OW': [0.749, 0.50, 0.55],
    })
    fig = px.bar(corr_data, x='Signal', y='Correlation with OW',
                 text='Correlation with OW', color='Correlation with OW',
                 color_continuous_scale='Blues')
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(yaxis_range=[0, 1], height=350)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.success("**Wikipedia views are a leading indicator of awareness**")
    st.markdown("""
    - Research-intent behavior: users actively seeking info about upcoming releases
    - **Less noise** than Google Trends (no search query ambiguity)
    - **Global consistency**: same article across countries
    - **Daily granularity** for 31 days pre-release captures marketing campaign lift
    - **Free, public, documented API** with no vendor lock-in
    """)

st.divider()

st.header("Where Wiki Features Help Most")
st.caption("Fair 5-fold GroupKFold CV, 276 dedup films, at V17.2 hyperparameters")

variant_data = pd.DataFrame({
    'Variant': ['A: V17.2 baseline', 'B: Wiki in regressor only', 'C: Wiki in classifier only', 'D: Wiki in both'],
    'D-14 Acc': [72.5, 72.5, 75.7, 75.7],
    'D-7 Acc': [72.8, 72.8, 75.4, 75.4],
    'D-3 Acc': [72.1, 72.1, 74.6, 74.6],
    'D-7 MAE ($M)': [11.67, 11.82, 11.02, 11.22],
})

st.dataframe(variant_data, use_container_width=True, hide_index=True)

st.markdown("""
**Insight**: Wikipedia features primarily help the **classifier** (tier boundary decisions).
The regressor's benefit is limited because Wiki signals are largely redundant with existing
log-transformed features (budget, star power, trends). Adding to the classifier unlocks +3pp
across every horizon.
""")

st.divider()

st.header("V18 Hyperparameter Tuning")
st.caption("108 configurations evaluated (27 Stage 1 + 27 Stage 2 × multiple reruns)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Stage 1 (Small vs Non-Small)")
    st.code("""
{
    'iterations': 200,    # (was 300 in V17.2)
    'depth': 7,           # same
    'learning_rate': 0.02 # (was 0.03)
}
""", language="python")

with col2:
    st.subheader("Stage 2 (Mid vs Large+)")
    st.code("""
{
    'iterations': 400,    # (was 300)
    'depth': 5,           # same
    'learning_rate': 0.03,# (was 0.02)
    'l2_leaf_reg': 5,     # preserved from V17.2
    'subsample': 0.8,     # preserved
    'colsample_bylevel': 0.9,
    'border_count': 64,
}
""", language="python")

st.divider()

st.header("Results Summary")

col1, col2, col3, col4 = st.columns(4)
col1.metric("V17.2 baseline (dedup)", "72.5%", "$11.46M MAE")
col2.metric("V18 final", "76.1%", "$11.07M MAE")
col3.metric("Accuracy delta", "+3.6pp", "10 additional films correct")
col4.metric("MAE delta", "-$0.39M", "-3.4%")

st.divider()

st.header("Bottom Line")
st.success("""
**The Snowflake Data Cloud enables data-science teams to iterate at the speed of curiosity.**

A customer asked "would Wikipedia help?" at 7:00 PM. By 7:52 PM the question was answered
with a quantified, production-ready model improvement — no procurement, no vendor negotiation,
no weeks of data engineering. Just Python + Snowflake + the open web.
""")

show_cortex_badge()
