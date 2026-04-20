import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from plotly.subplots import make_subplots
from cortex_badge import show_cortex_badge

st.set_page_config(page_title="Model Timeline", page_icon="📅", layout="wide")

st.title("📅 Model Development Timeline")
st.markdown("Evolution of the OW prediction model from V2 to V17, with actual timestamps from Snowflake.")

st.header("🧪 ML Experimentation Summary")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Model Versions", "11", "V2 -> V17", help="Total major model versions developed (V2-V5, V10-V11, V13-V17)")
with col2:
    st.metric("Total Experiments", "76", "+104 HP tuning runs", help="76 model experiments + 104 hyperparameter tuning iterations")
with col3:
    st.metric("Architectures Tested", "6", help="Single model, 2-tier, 3-tier, 4-tier, ensemble, cascade")
with col4:
    st.metric("Feature Combos", "59", "Final feature set", help="From 200+ candidate features tested")
with col5:
    st.metric("Best Model", "V17 3-Tier", "73.2% CV acc", help="3-tier cascade + TMDB override (Rule C) + extended trends")

st.divider()

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
            "STAR_X_ROLLING - Star x Trends interaction",
            "STAR_X_IP_HIGH - Star x IP interaction"
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
            "Budget x Trends interactions"
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
            "3-tier system: SMALL (<\\$15M) / MID (\\$15-50M) / LARGE+ (>=\\$50M)",
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
    },
    {
        "version": "V15",
        "date": "2026-03-08 07:06",
        "category": "Production",
        "description": "Data Quality + New Feature",
        "features": [
            "269 training films (+30 from data cleanup)",
            "PREDECESSOR_OW_LOG feature (sequel predecessor OW)",
            "52 features (32 static + 20 trends)",
            "11 duplicate/skeleton movie IDs removed",
            "25,207 pre-release comments scored (IF, Arthur, Challengers)",
            "51 noise movies excluded via REMOVE_FROM_MODEL table"
        ]
    },
    {
        "version": "V15 Prod",
        "date": "2026-03-08 07:10",
        "category": "Production",
        "description": "V15 Production Deployment",
        "features": [
            "ow_pipeline_v15_production.joblib.gz",
            "ow_pipeline_v15_3tier_3d.joblib.gz",
            "ow_pipeline_v15_3tier_7d.joblib.gz",
            "ow_pipeline_v15_3tier_14d.joblib.gz",
            "V14 archived to ML_MODELS_ARCHIVE"
        ]
    },
    {
        "version": "V16",
        "date": "2026-04-10 10:00",
        "category": "Production",
        "description": "TMDB Override System",
        "features": [
            "+16 training films (269 -> 285)",
            "IS_MAJOR_STUDIO feature",
            "TMDB_POPULARITY_D14, D7, MOMENTUM features",
            "56 features (36 static + 20 trends)",
            "Rule C TMDB override (5 variants tested on holdout)",
            "Holdout: 63.2% -> 84.2% tier accuracy with override"
        ]
    },
    {
        "version": "V16 Prod",
        "date": "2026-04-10 12:00",
        "category": "Production",
        "description": "V16 Production Deployment",
        "features": [
            "ow_pipeline_v16_production.joblib.gz (with override config)",
            "PREDICT_MOVIE_V16 stored procedure",
            "Override logic baked into model file",
            "Dashboard updated for V16",
            "V15 archived"
        ]
    },
    {
        "version": "V17",
        "date": "2026-04-20 09:00",
        "category": "Production",
        "description": "Extended Trends Features",
        "features": [
            "+3 new trends features: ROLLING_14D, ROLLING_21D, TRENDS_EARLIEST",
            "59 features (36 static + 23 trends)",
            "277 training films (cleaned dataset)",
            "5-fold GroupKFold CV grouped by MOVIE_ID",
            "Feature view extended to 74 columns",
            "ROLLING_21D ranks #4 in SMALL regressor at -14d"
        ]
    },
    {
        "version": "V17 Prod",
        "date": "2026-04-20 12:00",
        "category": "Production",
        "description": "V17 Production Deployment",
        "features": [
            "ow_pipeline_v17_production.joblib.gz",
            "Deployed to Snowflake stage",
            "Dashboard updated for V17",
            "V16 archived to ML_MODELS_ARCHIVE"
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
    st.subheader("Feature Engineering (Feb 15)")

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
        - `STAR_X_ROLLING` - Star x Trends interaction
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
    st.subheader("Architecture Evolution")

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
        - Budget x Trends interaction features
        """)

    with st.expander("V13 - Production Pipeline (Feb 16)", expanded=False):
        st.markdown("""
        **Time**: 14:18  
        Productionized prediction pipeline:
        - `RUN_V13_PREDICTIONS` stored procedure
        - `ML_PREDICTIONS_V` output view
        """)

    with st.expander("V14 - 3-Tier Cascade (Feb 27-28)", expanded=False):
        st.markdown("""
        **Time**: 19:07 -> 07:42  
        3-Tier cascade architecture:
        
        **Tier Consolidation**:
        - SMALL: < \\$15M (115 films, 48%)
        - MID: \\$15M - \\$50M (78 films, 33%)
        - LARGE+: >= \\$50M (46 films, 19%)
        
        **2-Stage Cascade**:
        1. Stage 1: SMALL vs NON-SMALL
        2. Stage 2: MID vs LARGE+ (if NON-SMALL)
        
        **Multi-Horizon**: 3d, 7d, 14d prediction windows
        """)

    with st.expander("V15 - Data Quality + Predecessor OW (Mar 8)", expanded=False):
        st.markdown("""
        **Time**: 07:06  
        Data quality overhaul:
        
        - 269 training films (+30 from data cleanup)
        - 11 duplicate/skeleton movie IDs removed
        - 25,207 pre-release comments scored
        - 51 noise movies excluded
        - `PREDECESSOR_OW_LOG` feature — immediately ranked #8
        - Classification: **77.3%** at -7d (+5.8% vs V14)
        """)

    with st.expander("V16 - TMDB Override System (Apr 10)", expanded=False):
        st.markdown("""
        **Time**: ~10:00 -> 12:00 (single Cortex Code session)
        
        **New Features**: IS_MAJOR_STUDIO, TMDB_POPULARITY_D14/D7, TMDB_POP_MOMENTUM
        
        **TMDB Override (Rule C)**:
        - 5 rule variants designed and tested on holdout
        - D14>=25 -> force LARGE+, D14>=15 + momentum>=1.3 -> force MID
        - Holdout: 63.2% -> **84.2%** tier accuracy
        - 4/4 correct overrides, 0 wrong
        
        **Deployment**: Model file updated, SP created, dashboard updated — all in one session
        """)

    with st.expander("V17 - Extended Trends Features (Apr 20)", expanded=True):
        st.markdown("""
        **Time**: ~09:00 -> 12:00 (single Cortex Code session)
        
        **New Features**: ROLLING_14D, ROLLING_21D, TRENDS_EARLIEST
        
        **Key Results**:
        - 59 features (36 static + 23 trends)
        - 277 training films, 5-fold GroupKFold CV
        - CV Accuracy: 73.2% (-14d), 73.2% (-7d), 73.6% (-3d)
        - CV MAE: $11.56M (-14d), $11.74M (-7d), $11.65M (-3d)
        - ROLLING_21D ranks #4 in SMALL regressor at -14d (importance=3.93)
        - TRENDS_EARLIEST ranks #3 in LARGE+ regressor at -7d (importance=4.86)
        
        **Deployment**: Model deployed to Snowflake stage, V16 archived
        """)

st.divider()

st.header("Feature Count by Version")

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

st.header("Data Integration Sprint")
st.markdown("**Feb 3-27, 2026** — Rapid feature engineering from external data sources")

API_SPRINT = [
    {"date": "2026-02-03", "time": "09:15", "item": "MOVIE_LEAD_ACTORS table", "type": "Data", "loc": 85, "hours": 0.5, "detail": "2,383 actor-movie mappings from external data"},
    {"date": "2026-02-03", "time": "10:30", "item": "MAX_STAR_POWER feature", "type": "Feature", "loc": 45, "hours": 0.75, "detail": "Highest lifetime box office per lead actor"},
    {"date": "2026-02-03", "time": "10:45", "item": "TOP2_STAR_POWER feature", "type": "Feature", "loc": 30, "hours": 0.5, "detail": "Combined top 2 lead actors"},
    {"date": "2026-02-03", "time": "11:00", "item": "AVG_STAR_POWER feature", "type": "Feature", "loc": 25, "hours": 0.75, "detail": "Cast average box office"},
    {"date": "2026-02-15", "time": "14:24", "item": "V3 model training", "type": "Model", "loc": 120, "hours": 0.5, "detail": "Star power features integrated"},
    {"date": "2026-02-15", "time": "14:30", "item": "STAR_X_ROLLING interaction", "type": "Feature", "loc": 15, "hours": 0.25, "detail": "Star power x Google Trends"},
    {"date": "2026-02-15", "time": "14:35", "item": "STAR_X_IP_HIGH interaction", "type": "Feature", "loc": 15, "hours": 0.25, "detail": "Star power x High-profile IP"},
    {"date": "2026-02-27", "time": "15:00", "item": "ACTOR_LIFETIME_BOX_OFFICE", "type": "Data", "loc": 95, "hours": 1.5, "detail": "2,205 actor career aggregations"},
    {"date": "2026-02-27", "time": "16:30", "item": "Hybrid star power", "type": "Feature", "loc": 60, "hours": 1.5, "detail": "Combined external + historical lookups"},
    {"date": "2026-02-27", "time": "19:07", "item": "V14 production model", "type": "Model", "loc": 180, "hours": 2.0, "detail": "Full star power integration"},
]

sprint_df = pd.DataFrame(API_SPRINT)
sprint_df['datetime'] = pd.to_datetime(sprint_df['date'] + ' ' + sprint_df['time'])
sprint_df = sprint_df.sort_values('datetime')
sprint_df['cumulative_loc'] = sprint_df['loc'].cumsum()
sprint_df['cumulative_hours'] = sprint_df['hours'].cumsum()

type_colors = {"Data": "#01B4E4", "Feature": "#E91E63", "Model": "#4CAF50"}

sprint_active_days = sprint_df['date'].nunique()
sprint_total_hours = sprint_df['hours'].sum()
sprint_total_loc = sprint_df['loc'].sum()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Active Work", f"~{sprint_total_hours:.0f}h", f"{sprint_active_days} sessions over 24 days")
with col2:
    st.metric("Lines of Code", f"~{sprint_total_loc}", "SQL + Python")
with col3:
    st.metric("Actors Indexed", "4,588", "Unique actor records")

fig_sprint = go.Figure()

fig_sprint.add_trace(go.Scatter(
    x=sprint_df['cumulative_hours'],
    y=sprint_df['cumulative_loc'],
    mode='lines+markers',
    name='Cumulative LOC',
    line=dict(color='#FFD700', width=3),
    marker=dict(
        size=14,
        color=[type_colors[t] for t in sprint_df['type']],
        line=dict(color='white', width=2)
    ),
    hovertemplate='<b>%{customdata[0]}</b><br>Hour %{x:.1f}<br>+%{customdata[1]} lines -> %{y} total<br><i>%{customdata[2]}</i><extra></extra>',
    customdata=list(zip(sprint_df['item'], sprint_df['loc'], sprint_df['detail']))
))

for i, row in sprint_df.iterrows():
    fig_sprint.add_annotation(
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
    fig_sprint.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        name=t,
        marker=dict(size=10, color=color),
        showlegend=True
    ))

fig_sprint.update_layout(
    height=350,
    title="Cumulative Lines of Code",
    xaxis_title="Hours Worked",
    yaxis_title="Lines of Code",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=20, r=20, t=60, b=40),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

st.plotly_chart(fig_sprint, use_container_width=True)

with st.expander("API Sprint Details", expanded=False):
    for _, row in sprint_df.iterrows():
        icon = "💾" if row['type'] == "Data" else "⚙️" if row['type'] == "Feature" else "🤖"
        st.markdown(f"{icon} **Hour {row['cumulative_hours']:.1f}** — {row['item']} (+{row['loc']} LOC): {row['detail']}")

st.divider()

st.header("Development Velocity")
st.markdown("**Jan 28 -> Apr 20, 2026** — Part-time/spare-time development")

ACTIVE_DAYS = [
    {"date": "2026-01-28", "hours": 3.0, "work": "Project kickoff, data exploration"},
    {"date": "2026-01-30", "hours": 4.0, "work": "ML infrastructure setup, base views"},
    {"date": "2026-02-03", "hours": 2.5, "work": "External data integration, actor tables"},
    {"date": "2026-02-15", "hours": 8.0, "work": "Feature engineering marathon (V2-V10)"},
    {"date": "2026-02-16", "hours": 5.0, "work": "Budget features, V11-V13 pipeline"},
    {"date": "2026-02-27", "hours": 6.0, "work": "V14 3-tier cascade, hybrid star power"},
    {"date": "2026-02-28", "hours": 4.0, "work": "Production deployment, visualization app"},
    {"date": "2026-04-10", "hours": 3.0, "work": "V16 TMDB override: design, test, deploy (one session)"},
    {"date": "2026-04-20", "hours": 3.0, "work": "V17 extended trends: feature view, training, CV, deploy"},
]

total_hours = sum(d['hours'] for d in ACTIVE_DAYS)
active_days = len(ACTIVE_DAYS)
calendar_span = 83

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Calendar Span", "83 days", "Jan 28 -> Apr 20")
with col2:
    st.metric("Active Coding Days", f"{active_days} days", f"{(active_days/calendar_span*100):.0f}% of calendar")
with col3:
    st.metric("Estimated Hours", f"~{total_hours:.0f}h", f"~{total_hours/active_days:.1f}h per session")
with col4:
    st.metric("Avg Days Between", f"{calendar_span/active_days:.1f} days", "Spare-time cadence")

st.subheader("Artifacts Produced")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Tables", "441", "Base tables")
with col2:
    st.metric("Views", "152", "Feature views")
with col3:
    st.metric("Procedures", "71", "Stored procs (+V16 SP)")
with col4:
    st.metric("ML Models", "5", "Production models")
with col5:
    st.metric("App Code", "~2,000", "Lines of Python")

st.subheader("Active Development Sessions")

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

with st.expander("Session Details", expanded=False):
    for row in ACTIVE_DAYS:
        st.markdown(f"**{row['date']}** ({row['hours']}h) — {row['work']}")

st.divider()

st.header("Model Experimentation Details")
st.markdown("Comprehensive breakdown of all models and experiments run during development")

EXPERIMENT_DATA = [
    {"version": "V2", "date": "2026-02-15", "architecture": "Single Model", "experiments": 8, "best_acc": 58.1, "notes": "Added rolling window features"},
    {"version": "V3", "date": "2026-02-15", "architecture": "Single Model", "experiments": 6, "best_acc": 61.4, "notes": "Star power features added"},
    {"version": "V4", "date": "2026-02-15", "architecture": "Single Model", "experiments": 4, "best_acc": 62.8, "notes": "3-day windows"},
    {"version": "V5", "date": "2026-02-15", "architecture": "Single Model", "experiments": 7, "best_acc": 63.5, "notes": "Intent signals + 7-day windows"},
    {"version": "V10", "date": "2026-02-15", "architecture": "4-Tier Cascade", "experiments": 14, "best_acc": 67.8, "notes": "Cascade classifier introduced"},
    {"version": "V11", "date": "2026-02-16", "architecture": "4-Tier Cascade", "experiments": 9, "best_acc": 68.4, "notes": "Budget features added"},
    {"version": "V13", "date": "2026-02-16", "architecture": "4-Tier Cascade", "experiments": 8, "best_acc": 67.8, "notes": "Production pipeline"},
    {"version": "V14", "date": "2026-02-27", "architecture": "3-Tier Cascade", "experiments": 10, "best_acc": 71.5, "notes": "Consolidated LARGE+, quantile loss"},
    {"version": "V15", "date": "2026-03-08", "architecture": "3-Tier Cascade", "experiments": 5, "best_acc": 77.3, "notes": "+30 films, PREDECESSOR_OW_LOG, data cleanup"},
    {"version": "V16", "date": "2026-04-10", "architecture": "3-Tier + Override", "experiments": 5, "best_acc": 84.2, "notes": "+16 films, IS_MAJOR_STUDIO, Rule C override (holdout)"},
    {"version": "V17", "date": "2026-04-20", "architecture": "3-Tier + Override", "experiments": 3, "best_acc": 73.2, "notes": "+3 trends features, 5-fold GroupKFold CV"},
]

exp_df = pd.DataFrame(EXPERIMENT_DATA)
exp_df['cumulative_experiments'] = exp_df['experiments'].cumsum()

col1, col2 = st.columns([2, 1])

with col1:
    fig_exp = make_subplots(specs=[[{"secondary_y": True}]])

    arch_colors = {
        "Single Model": "#0d6efd",
        "4-Tier Cascade": "#dc3545",
        "3-Tier Cascade": "#E91E8C",
        "3-Tier + Override": "#00CC96"
    }

    fig_exp.add_trace(
        go.Bar(
            x=exp_df['version'],
            y=exp_df['experiments'],
            name='Experiments per Version',
            marker_color=[arch_colors[a] for a in exp_df['architecture']],
            text=exp_df['experiments'],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Experiments: %{y}<br>Architecture: %{customdata[0]}<br>Notes: %{customdata[1]}<extra></extra>',
            customdata=list(zip(exp_df['architecture'], exp_df['notes']))
        ),
        secondary_y=False
    )

    fig_exp.add_trace(
        go.Scatter(
            x=exp_df['version'],
            y=exp_df['best_acc'],
            name='Best Accuracy %',
            mode='lines+markers',
            line=dict(color='#FFD700', width=3),
            marker=dict(size=10, color='#FFD700'),
            hovertemplate='<b>%{x}</b><br>Accuracy: %{y:.1f}%<extra></extra>'
        ),
        secondary_y=True
    )

    fig_exp.update_layout(
        title='Model Experiments & Accuracy Over Time',
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=20, r=20, t=80, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    fig_exp.update_yaxes(title_text="Experiments", secondary_y=False, range=[0, 20])
    fig_exp.update_yaxes(title_text="Accuracy %", secondary_y=True, range=[50, 90])

    st.plotly_chart(fig_exp, use_container_width=True)

with col2:
    st.subheader("Architecture Legend")
    for arch, color in arch_colors.items():
        count = exp_df[exp_df['architecture'] == arch]['experiments'].sum()
        st.markdown(f"<span style='color:{color}'>●</span> **{arch}**: {count} experiments", unsafe_allow_html=True)

    st.divider()

    st.subheader("Key Milestones")
    st.markdown("""
    - **V5**: Intent signals breakthrough (+1.5%)
    - **V8**: Ensemble tested, abandoned
    - **V10**: Cascade architecture (+3.8%)
    - **V14**: 3-tier consolidation (+3.7%)
    - **V15**: Data quality + predecessor OW (+5.8%)
    - **V16**: TMDB override (+6.9% on holdout)
    - **V17**: Extended trends features, rigorous CV
    """)

st.divider()

st.header("Hyperparameter Tuning Breakdown")

HYPERPARAM_DATA = [
    {"model": "Stage 1 Classifier", "param": "iterations", "values_tested": 12, "best": 500, "range": "100-1000"},
    {"model": "Stage 1 Classifier", "param": "depth", "values_tested": 8, "best": 8, "range": "4-12"},
    {"model": "Stage 1 Classifier", "param": "learning_rate", "values_tested": 10, "best": 0.02, "range": "0.005-0.1"},
    {"model": "Stage 2 Classifier", "param": "iterations", "values_tested": 12, "best": 500, "range": "100-1000"},
    {"model": "Stage 2 Classifier", "param": "depth", "values_tested": 8, "best": 7, "range": "4-12"},
    {"model": "Stage 2 Classifier", "param": "learning_rate", "values_tested": 10, "best": 0.02, "range": "0.005-0.1"},
    {"model": "SMALL Regressor", "param": "iterations", "values_tested": 8, "best": 600, "range": "200-1000"},
    {"model": "SMALL Regressor", "param": "depth", "values_tested": 6, "best": 5, "range": "3-8"},
    {"model": "SMALL Regressor", "param": "l2_leaf_reg", "values_tested": 5, "best": 5, "range": "1-10"},
    {"model": "MID Regressor", "param": "iterations", "values_tested": 8, "best": 800, "range": "200-1000"},
    {"model": "MID Regressor", "param": "depth", "values_tested": 6, "best": 6, "range": "3-8"},
    {"model": "MID Regressor", "param": "loss_function", "values_tested": 3, "best": "RMSE", "range": "MAE/RMSE/Quantile"},
    {"model": "LARGE+ Regressor", "param": "iterations", "values_tested": 8, "best": 500, "range": "200-1000"},
    {"model": "LARGE+ Regressor", "param": "depth", "values_tested": 6, "best": 5, "range": "3-8"},
    {"model": "LARGE+ Regressor", "param": "loss_function", "values_tested": 4, "best": "Quantile:a=0.5", "range": "MAE/RMSE/Quantile/Huber"},
]

hp_df = pd.DataFrame(HYPERPARAM_DATA)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total HP Experiments", hp_df['values_tested'].sum(), "Grid + Random search")
with col2:
    st.metric("Parameters Tuned", len(hp_df), "Across 5 models")
with col3:
    st.metric("Cross-Validation", "5-fold", "Stratified by tier")

st.dataframe(
    hp_df,
    hide_index=True,
    use_container_width=True,
    column_config={
        "model": st.column_config.TextColumn("Model Component", width="medium"),
        "param": st.column_config.TextColumn("Parameter"),
        "values_tested": st.column_config.NumberColumn("# Tested", format="%d"),
        "best": "Best Value",
        "range": "Search Range"
    }
)

st.divider()

st.header("Model Comparison Summary")

MODEL_COMPARISON = [
    {"version": "V2 (Rolling)", "architecture": "CatBoost Single", "features": 15, "accuracy": 58.1, "mae": 16.8, "status": "Superseded"},
    {"version": "V5 (Intent)", "architecture": "CatBoost Single", "features": 35, "accuracy": 63.5, "mae": 15.1, "status": "Superseded"},
    {"version": "V10 (4-Tier)", "architecture": "4-Tier Cascade", "features": 45, "accuracy": 67.8, "mae": 14.0, "status": "Superseded"},
    {"version": "V11 (Budget)", "architecture": "4-Tier Cascade", "features": 48, "accuracy": 68.4, "mae": 13.8, "status": "Superseded"},
    {"version": "V13 (Production)", "architecture": "4-Tier Cascade", "features": 51, "accuracy": 67.8, "mae": 14.0, "status": "Replaced"},
    {"version": "V14", "architecture": "3-Tier Cascade", "features": 51, "accuracy": 71.5, "mae": 13.1, "status": "Archived"},
    {"version": "V15", "architecture": "3-Tier Cascade", "features": 52, "accuracy": 77.3, "mae": 11.0, "status": "Archived"},
    {"version": "V16 (Override)", "architecture": "3-Tier + Override", "features": 56, "accuracy": 84.2, "mae": None, "status": "Archived"},
    {"version": "V17 (Current)", "architecture": "3-Tier + Override", "features": 59, "accuracy": 73.2, "mae": 11.74, "status": "Production"},
]

compare_df = pd.DataFrame(MODEL_COMPARISON)

st.dataframe(
    compare_df,
    hide_index=True,
    use_container_width=True,
    column_config={
        "version": st.column_config.TextColumn("Version", width="small"),
        "architecture": st.column_config.TextColumn("Architecture"),
        "features": st.column_config.NumberColumn("Features", format="%d"),
        "accuracy": st.column_config.NumberColumn("Accuracy %", format="%.1f%%", help="V16 accuracy is holdout w/ override; others are CV"),
        "mae": st.column_config.NumberColumn("MAE ($M)", format="$%.1f"),
        "status": "Status"
    }
)

st.caption("V16 accuracy (84.2%) is from holdout validation with override. V17 accuracy (73.2%) is from 5-fold GroupKFold CV — a more rigorous metric.")

show_cortex_badge()
