"""
Page 1: Model Architecture Visualization
"""

import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Architecture", page_icon="🏗️", layout="wide")

st.title("Model Architecture")
st.subheader("2-Stage Cascade Classification + Tier-Specific Regression")

st.divider()

st.header("Cascade Flow Diagram")

fig = go.Figure()

box_w = 0.12
box_h = 0.08

boxes = [
    {'x': 0.5, 'y': 0.88, 'color': '#1f77b4', 'label': 'INPUT', 'sublabel': '52 Features', 'hover': '32 Static + 20 Trend Features'},
    {'x': 0.5, 'y': 0.68, 'color': '#2ca02c', 'label': 'STAGE 1', 'sublabel': 'SMALL vs NON-SMALL', 'hover': 'Binary classifier: CatBoost depth=8'},
    {'x': 0.22, 'y': 0.38, 'color': '#17becf', 'label': 'SMALL', 'sublabel': 'Regressor', 'hover': 'MAE loss, 600 iterations, 137 films'},
    {'x': 0.5, 'y': 0.38, 'color': '#ff7f0e', 'label': 'STAGE 2', 'sublabel': 'MID vs LARGE+', 'hover': 'Binary classifier: CatBoost depth=7'},
    {'x': 0.36, 'y': 0.12, 'color': '#9467bd', 'label': 'MID', 'sublabel': 'Regressor', 'hover': 'RMSE loss, 800 iterations, 84 films'},
    {'x': 0.64, 'y': 0.12, 'color': '#d62728', 'label': 'LARGE+', 'sublabel': 'Regressor', 'hover': 'Quantile loss (α=0.5), 500 iterations, 48 films'},
]

for box in boxes:
    fig.add_shape(
        type='rect',
        x0=box['x'] - box_w, y0=box['y'] - box_h,
        x1=box['x'] + box_w, y1=box['y'] + box_h,
        fillcolor=box['color'],
        line=dict(color=box['color'], width=2),
        layer='below'
    )
    fig.add_annotation(
        x=box['x'], y=box['y'] + 0.02,
        text=f"<b>{box['label']}</b>",
        showarrow=False,
        font=dict(color='white', size=13),
    )
    fig.add_annotation(
        x=box['x'], y=box['y'] - 0.03,
        text=box['sublabel'],
        showarrow=False,
        font=dict(color='white', size=10),
    )
    fig.add_trace(go.Scatter(
        x=[box['x']], y=[box['y']],
        mode='markers',
        marker=dict(size=1, color='rgba(0,0,0,0)'),
        hoverinfo='text',
        hovertext=box['hover'],
        showlegend=False
    ))

connections = [
    {'x0': 0.5, 'y0': 0.80, 'x1': 0.5, 'y1': 0.76, 'color': '#666'},
    {'x0': 0.5, 'y0': 0.60, 'x1': 0.5, 'y1': 0.54, 'color': '#666'},
    {'x0': 0.35, 'y0': 0.54, 'x1': 0.22, 'y1': 0.46, 'color': '#17becf'},
    {'x0': 0.5, 'y0': 0.54, 'x1': 0.5, 'y1': 0.46, 'color': '#ff7f0e'},
    {'x0': 0.5, 'y0': 0.30, 'x1': 0.5, 'y1': 0.24, 'color': '#666'},
    {'x0': 0.42, 'y0': 0.24, 'x1': 0.36, 'y1': 0.20, 'color': '#9467bd'},
    {'x0': 0.58, 'y0': 0.24, 'x1': 0.64, 'y1': 0.20, 'color': '#d62728'},
]

for conn in connections:
    fig.add_annotation(
        x=conn['x1'], y=conn['y1'],
        ax=conn['x0'], ay=conn['y0'],
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor=conn['color']
    )

labels = [
    {'x': 0.28, 'y': 0.52, 'text': 'SMALL', 'color': '#17becf'},
    {'x': 0.58, 'y': 0.52, 'text': 'NON-SMALL', 'color': '#ff7f0e'},
    {'x': 0.33, 'y': 0.24, 'text': 'MID', 'color': '#9467bd'},
    {'x': 0.67, 'y': 0.24, 'text': 'LARGE+', 'color': '#d62728'},
]

for lbl in labels:
    fig.add_annotation(
        x=lbl['x'], y=lbl['y'],
        text=f"<b>{lbl['text']}</b>",
        showarrow=False,
        font=dict(color=lbl['color'], size=11),
    )

fig.update_layout(
    showlegend=False,
    xaxis=dict(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
    yaxis=dict(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, scaleanchor='x'),
    height=500,
    margin=dict(l=20, r=20, t=20, b=20),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
)

fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)

st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.header("Tier Boundaries")
    st.markdown("""
    | Tier | Revenue Range | Films |
    |------|---------------|-------|
    | **SMALL** | < \\$15M | 137 (51%) |
    | **MID** | \\$15M – \\$50M | 84 (31%) |
    | **LARGE+** | ≥ \\$50M | 48 (18%) |
    
    **V15 Update**: 269 training films (+30 vs V14's 239) after comprehensive 
    data cleanup — duplicate removal, scoring gap fills, and skeleton movie exclusions.
    """)

with col2:
    st.header("Classifier Configurations")
    st.markdown("""
    **Stage 1: SMALL vs NON-SMALL**
    ```python
    CatBoostClassifier(
        iterations=500, 
        depth=8, 
        learning_rate=0.02
    )
    ```
    
    **Stage 2: MID vs LARGE+**
    ```python
    CatBoostClassifier(
        iterations=500, 
        depth=7, 
        learning_rate=0.02
    )
    ```
    """)

st.divider()

st.header("Tier-Specific Regressor Configurations")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("SMALL Tier")
    st.code("""
{
  'iterations': 600,
  'depth': 5,
  'learning_rate': 0.02,
  'l2_leaf_reg': 5,
  'loss_function': 'MAE'
}
    """, language='python')
    st.caption("Optimized for many small films with low variance")

with col2:
    st.subheader("MID Tier")
    st.code("""
{
  'iterations': 800,
  'depth': 6,
  'learning_rate': 0.015,
  'l2_leaf_reg': 5,
  'loss_function': 'RMSE'
}
    """, language='python')
    st.caption("Balanced approach for mid-range films")

with col3:
    st.subheader("LARGE+ Tier")
    st.code("""
{
  'iterations': 500,
  'depth': 5,
  'learning_rate': 0.02,
  'l2_leaf_reg': 8,
  'loss_function': 'Quantile:alpha=0.5'
}
    """, language='python')
    st.caption("Quantile regression handles high variance blockbusters")

st.divider()

st.header("V15 Feature Set (52 Features)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Static Features (32)")
    st.markdown("""
    **YouTube/Sentiment** (7): YT_COMMENTS, ENGAGEMENT_RATIO, SENTIMENT, 
    THEATRICAL_INTENT_PCT, STREAMING_INTENT_PCT, PASS_INTENT_PCT, NET_INTENT_PCT
    
    **Movie Attributes** (6): BUDGET, BUDGET_LOG, RUNTIME, TMDB_POPULARITY, 
    RELEASE_MONTH, IS_PEAK_SEASON
    
    **Star Power** (4): MAX_STAR_POWER, TOP2_STAR_POWER, AVG_STAR_POWER, 
    NUM_STARS_WITH_HISTORY
    
    **Genre** (5): ACTION_FRANCHISE, ANIMATION_FAMILY, HORROR, PRESTIGE, ORIGINAL
    
    **Rating** (4): G, PG, PG13, R
    
    **IP/Franchise** (5): KNOWN_IP_TIER, IP_HIGH_PROFILE, IP_MODERATE, IP_NICHE, IP_ORIGINAL
    
    **🆕 Sequel Signal** (1): `PREDECESSOR_OW_LOG` — log of predecessor film's OW
    """)

with col2:
    st.subheader("Trends Features (20)")
    st.markdown("""
    **Rolling Averages** (6): ROLLING_3D, ROLLING_5D, ROLLING_7D, 
    ROLLING_3D_PRIOR, ROLLING_5D_PRIOR, ROLLING_7D_PRIOR
    
    **Velocity** (3): VELOCITY_3D, VELOCITY_5D, VELOCITY_7D
    
    **Cumulative** (4): TRENDS_CUMULATIVE, TRENDS_VOLATILITY, 
    TRENDS_PEAK_SO_FAR, DAYS_WITH_DATA
    
    **Interaction Features** (7): ROLLING_X_IP_HIGH, ROLLING_X_ACTION, 
    ROLLING_X_HORROR, SENTIMENT_X_ROLLING, STAR_X_ROLLING, 
    STAR_X_IP_HIGH, INTENT_X_ROLLING
    """)

st.divider()

st.header("Why 3-Tier Instead of 4-Tier?")

col1, col2 = st.columns(2)

with col1:
    st.subheader("V13 (4-Tier) Problems")
    st.markdown("""
    - LARGE tier ($50-100M): Only 27 training films
    - BLOCKBUSTER tier ($100M+): Only 19 training films
    - LARGE tier accuracy: **27%** (terrible)
    - Model couldn't distinguish similar-sized films
    """)

with col2:
    st.subheader("V14/V15 (3-Tier) Solution")
    st.markdown("""
    - Combined into LARGE+ tier: **48 training films** in V15
    - LARGE+ accuracy: **77.1%** in V15 (+50% vs V13!)
    - More robust classification boundary
    - Better generalization with limited data
    """)
