import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from plotly.subplots import make_subplots

st.set_page_config(page_title="Model Timeline", page_icon="📅", layout="wide")

st.title("📅 Model Development Timeline")
st.markdown("Evolution of the OW prediction model from V2 to V14, with actual timestamps from Snowflake.")

TIMELINE_DATA = [
    {
        "version": "Base",
        "date": "2026-01-30 13:28",
        "category": "Infrastructure",
        "description": "ML Infrastructure Setup",
        "features": [
            "TRAINING_DATA_V - Core training data view",
            "BOX_OFFICE_FEATURES_V - Box office feature engineering",
            "QUORUM_TRACKING_V - Data quality tracking"
        ]
    },
    {
        "version": "IP",
        "date": "2026-02-15 13:16",
        "category": "Features",
        "description": "IP Classification",
        "features": [
            "TRAINING_FEATURES_IP_V - IP tier classification",
            "IP_HIGH_PROFILE, IP_MODERATE, IP_NICHE, IP_ORIGINAL flags"
        ]
    },
    {
        "version": "V2",
        "date": "2026-02-15 14:09",
        "category": "Features",
        "description": "Rolling Windows",
        "features": [
            "ROLLING_5D - 5-day rolling average trends",
            "ROLLING_5D_PRIOR - Prior period comparison",
            "VELOCITY - Momentum ratio (current/prior)",
            "MOMENTUM_DELTA - Absolute trend change",
            "Theater interaction features"
        ]
    },
    {
        "version": "V3",
        "date": "2026-02-15 14:24",
        "category": "Features",
        "description": "Star Power",
        "features": [
            "MAX_STAR_POWER - Top actor box office history",
            "TOP2_STAR_POWER - Lead duo combined power",
            "AVG_STAR_POWER - Cast average",
            "STAR_X_ROLLING - Star × Trends interaction",
            "STAR_X_IP_HIGH - Star × IP interaction"
        ]
    },
    {
        "version": "V4",
        "date": "2026-02-15 14:27",
        "category": "Features",
        "description": "3-Day Windows",
        "features": [
            "ROLLING_3D - Shorter-term trend signal",
            "VELOCITY_3D - 3-day momentum",
            "MOMENTUM_3D - Short-term change detection"
        ]
    },
    {
        "version": "V5",
        "date": "2026-02-15 14:34",
        "category": "Features",
        "description": "Intent & Extended Windows",
        "features": [
            "THEATRICAL_INTENT_PCT - YouTube intent signals",
            "STREAMING_INTENT_PCT - Streaming preference",
            "PASS_INTENT_PCT - Audience pass rate",
            "KNOWN_IP_TIER - Numeric IP tier",
            "IP_SOURCE_TYPE - IP origin category",
            "ROLLING_7D - 7-day trend windows"
        ]
    },
    {
        "version": "V10",
        "date": "2026-02-15 20:17",
        "category": "Architecture",
        "description": "Tier Classification",
        "features": [
            "CLASSIFY_TIER_V10 procedure",
            "MOVIE_TIER_CLASSIFICATIONS table",
            "4-tier system: SMALL/MID/LARGE/BLOCKBUSTER",
            "CatBoost classifier for tier prediction"
        ]
    },
    {
        "version": "V11",
        "date": "2026-02-16 00:51",
        "category": "Architecture",
        "description": "Budget Features",
        "features": [
            "BUDGET_TIER categorical feature",
            "BUDGET_MEGA, BUDGET_HIGH, BUDGET_MID flags",
            "Budget × Trends interactions"
        ]
    },
    {
        "version": "V13",
        "date": "2026-02-16 14:18",
        "category": "Architecture",
        "description": "Production Pipeline",
        "features": [
            "RUN_V13_PREDICTIONS procedure",
            "ML_PREDICTIONS_V production view",
            "OW_PREDICTION_FEATURES_V aligned schema"
        ]
    },
    {
        "version": "V14",
        "date": "2026-02-27 19:07",
        "category": "Production",
        "description": "3-Tier Cascade",
        "features": [
            "3-tier system: SMALL (<\\$15M) / MID (\\$15-50M) / LARGE+ (≥\\$50M)",
            "2-stage cascade classifier",
            "Tier-specific CatBoost regressors",
            "Quantile loss for LARGE+ (high variance)",
            "Multi-horizon models (3d, 7d, 14d)",
            "Hybrid star power integration"
        ]
    },
    {
        "version": "V14 Prod",
        "date": "2026-02-28 07:42",
        "category": "Production",
        "description": "Production Deployment",
        "features": [
            "ow_pipeline_v14_production.joblib.gz",
            "ow_pipeline_v14_3tier_3d.joblib.gz",
            "ow_pipeline_v14_3tier_7d.joblib.gz", 
            "ow_pipeline_v14_3tier_14d.joblib.gz"
        ]
    }
]

df = pd.DataFrame(TIMELINE_DATA)
df['datetime'] = pd.to_datetime(df['date'])
df['y_pos'] = range(len(df))

category_colors = {
    "Infrastructure": "#6c757d",
    "Features": "#0d6efd",
    "Architecture": "#198754",
    "Production": "#dc3545"
}
df['color'] = df['category'].map(category_colors)

st.header("Development Journey")

fig = go.Figure()

for cat in ["Infrastructure", "Features", "Architecture", "Production"]:
    cat_df = df[df['category'] == cat]
    fig.add_trace(go.Scatter(
        x=cat_df['datetime'],
        y=cat_df['y_pos'],
        mode='markers+text',
        name=cat,
        marker=dict(size=20, color=category_colors[cat], symbol='circle'),
        text=cat_df['version'],
        textposition='top center',
        textfont=dict(size=12, color='white'),
        hovertemplate='<b>%{customdata[0]}</b><br>' +
                      '%{customdata[1]}<br>' +
                      '<i>%{customdata[2]}</i><extra></extra>',
        customdata=list(zip(cat_df['version'], cat_df['date'], cat_df['description']))
    ))

for i in range(len(df) - 1):
    fig.add_shape(
        type="line",
        x0=df.iloc[i]['datetime'],
        y0=df.iloc[i]['y_pos'],
        x1=df.iloc[i+1]['datetime'],
        y1=df.iloc[i+1]['y_pos'],
        line=dict(color="rgba(255,255,255,0.3)", width=2, dash="dot")
    )

fig.update_layout(
    height=500,
    xaxis_title="Date",
    yaxis_title="",
    yaxis=dict(showticklabels=False, showgrid=False),
    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    margin=dict(l=20, r=20, t=60, b=40),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

st.header("Version Details")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔧 Feature Engineering (Feb 15)")
    
    with st.expander("V2 - Rolling Windows", expanded=False):
        st.markdown("""
        **Time**: 14:09  
        First introduction of time-series features:
        - `ROLLING_5D` - 5-day rolling average Google Trends
        - `VELOCITY` - Momentum ratio (current period / prior period)
        - `MOMENTUM_DELTA` - Absolute trend change
        """)
    
    with st.expander("V3 - Star Power", expanded=False):
        st.markdown("""
        **Time**: 14:24 (+15 min)  
        Added actor box office history:
        - `MAX_STAR_POWER` - Highest-grossing lead actor
        - `TOP2_STAR_POWER` - Combined top 2 actors
        - `STAR_X_ROLLING` - Star × Trends interaction
        """)
    
    with st.expander("V4 - 3-Day Windows", expanded=False):
        st.markdown("""
        **Time**: 14:27 (+3 min)  
        Shorter-term trend detection:
        - `ROLLING_3D` - More responsive to recent changes
        - `VELOCITY_3D` - Short-term momentum
        """)
    
    with st.expander("V5 - Intent & 7-Day", expanded=False):
        st.markdown("""
        **Time**: 14:34 (+7 min)  
        YouTube intent signals and extended windows:
        - `THEATRICAL_INTENT_PCT` - "Want to see in theaters"
        - `STREAMING_INTENT_PCT` - "Wait for streaming"
        - `ROLLING_7D` - Longer-term trend stability
        """)

with col2:
    st.subheader("🏗️ Architecture Evolution")
    
    with st.expander("V10 - Tier Classification (Feb 15)", expanded=False):
        st.markdown("""
        **Time**: 20:17  
        Introduced segmented modeling:
        - **4 tiers**: SMALL / MID / LARGE / BLOCKBUSTER
        - `CLASSIFY_TIER_V10` - CatBoost classifier
        - `MOVIE_TIER_CLASSIFICATIONS` - Prediction storage
        """)
    
    with st.expander("V11 - Budget Features (Feb 16)", expanded=False):
        st.markdown("""
        **Time**: 00:51  
        Added production budget as predictor:
        - `BUDGET_TIER` - Categorical budget level
        - Budget × Trends interaction features
        """)
    
    with st.expander("V13 - Production Pipeline (Feb 16)", expanded=False):
        st.markdown("""
        **Time**: 14:18  
        Productionized prediction pipeline:
        - `RUN_V13_PREDICTIONS` stored procedure
        - `ML_PREDICTIONS_V` output view
        """)
    
    with st.expander("V14 - 3-Tier Cascade (Feb 27-28)", expanded=True):
        st.markdown("""
        **Time**: 19:07 → 07:42  
        Final production architecture:
        
        **Tier Consolidation**:
        - SMALL: < \\$15M (115 films, 48%)
        - MID: \\$15M - \\$50M (78 films, 33%)
        - LARGE+: ≥ \\$50M (46 films, 19%)
        
        **2-Stage Cascade**:
        1. Stage 1: SMALL vs NON-SMALL
        2. Stage 2: MID vs LARGE+ (if NON-SMALL)
        
        **Tier-Specific Regressors**:
        - SMALL/MID: RMSE loss
        - LARGE+: Quantile loss (α=0.5) for high variance
        
        **Multi-Horizon**: 3d, 7d, 14d prediction windows
        """)

st.divider()

st.header("📊 Feature Count by Version")

feature_counts = []
cumulative = 0
for item in TIMELINE_DATA:
    cumulative += len(item['features'])
    feature_counts.append({
        'version': item['version'],
        'new_features': len(item['features']),
        'cumulative': cumulative,
        'category': item['category']
    })

fc_df = pd.DataFrame(feature_counts)

fig2 = go.Figure()

fig2.add_trace(go.Bar(
    x=fc_df['version'],
    y=fc_df['new_features'],
    name='New Features',
    marker_color=[category_colors[c] for c in fc_df['category']],
    text=fc_df['new_features'],
    textposition='outside'
))

fig2.add_trace(go.Scatter(
    x=fc_df['version'],
    y=fc_df['cumulative'],
    name='Cumulative',
    mode='lines+markers',
    line=dict(color='#FFD700', width=3),
    marker=dict(size=10, color='#FFD700')
))

fig2.update_layout(
    height=350,
    xaxis_title="Version",
    yaxis_title="Feature Count",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=20, r=20, t=60, b=40),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

st.plotly_chart(fig2, use_container_width=True)

st.divider()

st.header("🎬 API Integration Sprint")
st.markdown("**Feb 3-27, 2026** — Rapid feature engineering from external API data")

TMDB_SPRINT = [
    {"date": "2026-02-03", "time": "09:15", "item": "MOVIE_LEAD_ACTORS table", "type": "Data", "loc": 85, "hours": 0.5, "detail": "2,383 actor-movie mappings from external API"},
    {"date": "2026-02-03", "time": "10:30", "item": "MAX_STAR_POWER feature", "type": "Feature", "loc": 45, "hours": 0.75, "detail": "Highest lifetime box office per lead actor"},
    {"date": "2026-02-03", "time": "10:45", "item": "TOP2_STAR_POWER feature", "type": "Feature", "loc": 30, "hours": 0.5, "detail": "Combined top 2 lead actors"},
    {"date": "2026-02-03", "time": "11:00", "item": "AVG_STAR_POWER feature", "type": "Feature", "loc": 25, "hours": 0.75, "detail": "Cast average box office"},
    {"date": "2026-02-15", "time": "14:24", "item": "V3 model training", "type": "Model", "loc": 120, "hours": 0.5, "detail": "Star power features integrated"},
    {"date": "2026-02-15", "time": "14:30", "item": "STAR_X_ROLLING interaction", "type": "Feature", "loc": 15, "hours": 0.25, "detail": "Star power × Google Trends"},
    {"date": "2026-02-15", "time": "14:35", "item": "STAR_X_IP_HIGH interaction", "type": "Feature", "loc": 15, "hours": 0.25, "detail": "Star power × High-profile IP"},
    {"date": "2026-02-27", "time": "15:00", "item": "ACTOR_LIFETIME_BOX_OFFICE", "type": "Data", "loc": 95, "hours": 1.5, "detail": "2,205 actor career aggregations"},
    {"date": "2026-02-27", "time": "16:30", "item": "Hybrid star power", "type": "Feature", "loc": 60, "hours": 1.5, "detail": "Combined API + historical lookups"},
    {"date": "2026-02-27", "time": "19:07", "item": "V14 production model", "type": "Model", "loc": 180, "hours": 2.0, "detail": "Full star power integration"},
]

tmdb_df = pd.DataFrame(TMDB_SPRINT)
tmdb_df['datetime'] = pd.to_datetime(tmdb_df['date'] + ' ' + tmdb_df['time'])
tmdb_df = tmdb_df.sort_values('datetime')
tmdb_df['cumulative_loc'] = tmdb_df['loc'].cumsum()
tmdb_df['cumulative_hours'] = tmdb_df['hours'].cumsum()

type_colors = {"Data": "#01B4E4", "Feature": "#E91E63", "Model": "#4CAF50"}

tmdb_active_days = tmdb_df['date'].nunique()
tmdb_total_hours = tmdb_df['hours'].sum()
tmdb_total_loc = tmdb_df['loc'].sum()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Active Work", f"~{tmdb_total_hours:.0f}h", f"{tmdb_active_days} sessions over 24 days")
with col2:
    st.metric("Lines of Code", f"~{tmdb_total_loc}", "SQL + Python")
with col3:
    st.metric("Actors Indexed", "4,588", "Unique actor records")

fig_tmdb = go.Figure()

fig_tmdb.add_trace(go.Scatter(
    x=tmdb_df['cumulative_hours'],
    y=tmdb_df['cumulative_loc'],
    mode='lines+markers',
    name='Cumulative LOC',
    line=dict(color='#FFD700', width=3),
    marker=dict(
        size=14,
        color=[type_colors[t] for t in tmdb_df['type']],
        line=dict(color='white', width=2)
    ),
    hovertemplate='<b>%{customdata[0]}</b><br>Hour %{x:.1f}<br>+%{customdata[1]} lines → %{y} total<br><i>%{customdata[2]}</i><extra></extra>',
    customdata=list(zip(tmdb_df['item'], tmdb_df['loc'], tmdb_df['detail']))
))

for i, row in tmdb_df.iterrows():
    fig_tmdb.add_annotation(
        x=row['cumulative_hours'],
        y=row['cumulative_loc'],
        text=row['item'].replace(' ', '<br>') if len(row['item']) > 15 else row['item'],
        showarrow=True,
        arrowhead=0,
        arrowcolor='rgba(255,255,255,0.3)',
        ax=0,
        ay=-35,
        font=dict(size=9, color='white'),
        align='center'
    )

for t, color in type_colors.items():
    fig_tmdb.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        name=t,
        marker=dict(size=10, color=color),
        showlegend=True
    ))

fig_tmdb.update_layout(
    height=350,
    title="Cumulative Lines of Code",
    xaxis_title="Hours Worked",
    yaxis_title="Lines of Code",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=20, r=20, t=60, b=40),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

st.plotly_chart(fig_tmdb, use_container_width=True)

with st.expander("📋 API Sprint Details", expanded=False):
    for _, row in tmdb_df.iterrows():
        icon = "💾" if row['type'] == "Data" else "⚙️" if row['type'] == "Feature" else "🤖"
        st.markdown(f"{icon} **Hour {row['cumulative_hours']:.1f}** — {row['item']} (+{row['loc']} LOC): {row['detail']}")

st.divider()

st.header("⏱️ Development Velocity")
st.markdown("**Jan 28 → Feb 28, 2026** — Part-time/spare-time development")

ACTIVE_DAYS = [
    {"date": "2026-01-28", "hours": 3.0, "work": "Project kickoff, data exploration"},
    {"date": "2026-01-30", "hours": 4.0, "work": "ML infrastructure setup, base views"},
    {"date": "2026-02-03", "hours": 2.5, "work": "External API integration, actor tables"},
    {"date": "2026-02-15", "hours": 8.0, "work": "Feature engineering marathon (V2-V10)"},
    {"date": "2026-02-16", "hours": 5.0, "work": "Budget features, V11-V13 pipeline"},
    {"date": "2026-02-27", "hours": 6.0, "work": "V14 3-tier cascade, hybrid star power"},
    {"date": "2026-02-28", "hours": 4.0, "work": "Production deployment, visualization app"},
]

total_hours = sum(d['hours'] for d in ACTIVE_DAYS)
active_days = len(ACTIVE_DAYS)
calendar_span = 32  # Jan 28 to Feb 28

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Calendar Span", "32 days", "Jan 28 → Feb 28")
with col2:
    st.metric("Active Coding Days", f"{active_days} days", f"{(active_days/calendar_span*100):.0f}% of calendar")
with col3:
    st.metric("Estimated Hours", f"~{total_hours:.0f}h", f"~{total_hours/active_days:.1f}h per session")
with col4:
    st.metric("Avg Days Between", f"{calendar_span/active_days:.1f} days", "Spare-time cadence")

st.subheader("📦 Artifacts Produced")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Tables", "441", "Base tables")
with col2:
    st.metric("Views", "152", "Feature views")
with col3:
    st.metric("Procedures", "70", "Stored procs")
with col4:
    st.metric("ML Models", "4", "Production models")
with col5:
    st.metric("App Code", "~1,760", "Lines of Python")

st.subheader("📅 Active Development Sessions")

velocity_df = pd.DataFrame(ACTIVE_DAYS)
velocity_df['datetime'] = pd.to_datetime(velocity_df['date'])

fig_velocity = go.Figure()

fig_velocity.add_trace(go.Bar(
    x=velocity_df['datetime'],
    y=velocity_df['hours'],
    name='Hours',
    marker_color='#0d6efd',
    text=velocity_df['hours'].apply(lambda x: f"{x}h"),
    textposition='outside',
    hovertemplate='<b>%{customdata[0]}</b><br>%{customdata[1]}h<br><i>%{customdata[2]}</i><extra></extra>',
    customdata=list(zip(velocity_df['date'], velocity_df['hours'], velocity_df['work']))
))

fig_velocity.update_layout(
    height=300,
    xaxis_title="Date",
    yaxis_title="Hours",
    yaxis=dict(range=[0, 10]),
    margin=dict(l=20, r=20, t=20, b=40),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

st.plotly_chart(fig_velocity, use_container_width=True)

with st.expander("📋 Session Details", expanded=False):
    for row in ACTIVE_DAYS:
        st.markdown(f"**{row['date']}** ({row['hours']}h) — {row['work']}")
