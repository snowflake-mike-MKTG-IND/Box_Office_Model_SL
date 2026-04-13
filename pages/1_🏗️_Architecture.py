"""
Page 1: Model Architecture Visualization
"""

import streamlit as st
import plotly.graph_objects as go
from cortex_badge import show_cortex_badge

st.set_page_config(page_title="Architecture", page_icon="🏗️", layout="wide")

st.title("Model Architecture")
st.subheader("2-Stage Cascade Classification + Tier-Specific Regression + TMDB Override")

st.divider()

st.header("Cascade Flow Diagram")

fig = go.Figure()

box_w = 0.12
box_h = 0.065

boxes = [
    {'x': 0.5, 'y': 0.92, 'color': '#1f77b4', 'label': 'INPUT', 'sublabel': '56 Features', 'hover': '36 Static + 20 Trend Features'},
    {'x': 0.5, 'y': 0.74, 'color': '#2ca02c', 'label': 'STAGE 1', 'sublabel': 'SMALL vs NON-SMALL', 'hover': 'Binary classifier: CatBoost depth=8'},
    {'x': 0.22, 'y': 0.50, 'color': '#17becf', 'label': 'SMALL', 'sublabel': 'Regressor', 'hover': 'MAE loss, 600 iterations, 148 films'},
    {'x': 0.5, 'y': 0.50, 'color': '#ff7f0e', 'label': 'STAGE 2', 'sublabel': 'MID vs LARGE+', 'hover': 'Binary classifier: CatBoost depth=7'},
    {'x': 0.36, 'y': 0.28, 'color': '#9467bd', 'label': 'MID', 'sublabel': 'Regressor', 'hover': 'RMSE loss, 800 iterations, 86 films'},
    {'x': 0.64, 'y': 0.28, 'color': '#d62728', 'label': 'LARGE+', 'sublabel': 'Regressor', 'hover': 'Quantile loss (α=0.5), 500 iterations, 51 films'},
    {'x': 0.5, 'y': 0.08, 'color': '#e377c2', 'label': 'RULE C', 'sublabel': 'TMDB Override', 'hover': 'Post-prediction safety net: D14>=25→LARGE+, D14>=15+momentum>=1.3→MID. Can only RAISE tier.'},
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
        x=box['x'], y=box['y'] - 0.02,
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
    {'x0': 0.5, 'y0': 0.855, 'x1': 0.5, 'y1': 0.805, 'color': '#666'},
    {'x0': 0.5, 'y0': 0.675, 'x1': 0.5, 'y1': 0.625, 'color': '#666'},
    {'x0': 0.35, 'y0': 0.625, 'x1': 0.22, 'y1': 0.565, 'color': '#17becf'},
    {'x0': 0.5, 'y0': 0.625, 'x1': 0.5, 'y1': 0.565, 'color': '#ff7f0e'},
    {'x0': 0.5, 'y0': 0.435, 'x1': 0.5, 'y1': 0.385, 'color': '#666'},
    {'x0': 0.42, 'y0': 0.385, 'x1': 0.36, 'y1': 0.345, 'color': '#9467bd'},
    {'x0': 0.58, 'y0': 0.385, 'x1': 0.64, 'y1': 0.345, 'color': '#d62728'},
    {'x0': 0.22, 'y0': 0.435, 'x1': 0.35, 'y1': 0.145, 'color': '#e377c2'},
    {'x0': 0.36, 'y0': 0.215, 'x1': 0.42, 'y1': 0.145, 'color': '#e377c2'},
    {'x0': 0.64, 'y0': 0.215, 'x1': 0.58, 'y1': 0.145, 'color': '#e377c2'},
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
    {'x': 0.28, 'y': 0.61, 'text': 'SMALL', 'color': '#17becf'},
    {'x': 0.58, 'y': 0.61, 'text': 'NON-SMALL', 'color': '#ff7f0e'},
    {'x': 0.33, 'y': 0.35, 'text': 'MID', 'color': '#9467bd'},
    {'x': 0.67, 'y': 0.35, 'text': 'LARGE+', 'color': '#d62728'},
]

for lbl in labels:
    fig.add_annotation(
        x=lbl['x'], y=lbl['y'],
        text=f"<b>{lbl['text']}</b>",
        showarrow=False,
        font=dict(color=lbl['color'], size=11),
    )

fig.add_annotation(
    x=0.5, y=0.17,
    text="<b>All tier predictions pass through TMDB override</b>",
    showarrow=False,
    font=dict(color='#e377c2', size=10),
)

fig.update_layout(
    showlegend=False,
    xaxis=dict(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
    yaxis=dict(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, scaleanchor='x'),
    height=550,
    margin=dict(l=20, r=20, t=20, b=20),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
)

fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)

st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

st.info(
    "**Cortex Code Contribution**: This cascade architecture was iterated through "
    "6 different approaches (Random Forest, XGBoost, LightGBM, CatBoost, Neural Net, "
    "Ensemble) across 76 experiments before arriving at the 3-tier CatBoost design. "
    "The V16 TMDB override was designed, tested (5 rule variants on holdout), and deployed "
    "in a single Cortex Code session."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.header("Tier Boundaries")
    st.markdown("""
    | Tier | Revenue Range | Films |
    |------|---------------|-------|
    | **SMALL** | < \\$15M | 148 (52%) |
    | **MID** | \\$15M – \\$50M | 86 (30%) |
    | **LARGE+** | ≥ \\$50M | 51 (18%) |
    
    **V16**: 285 training films (+16 vs V15's 269). Added IS_MAJOR_STUDIO 
    and TMDB daily popularity features (D14, D7, momentum).
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

st.header("TMDB Popularity Override (Rule C)")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        "The override operates **orthogonally** to the model — it runs *after* the cascade "
        "prediction and can only **raise** a tier, never lower it. This addresses a key limitation: "
        "TMDB daily popularity data exists for only ~30 training films, so CatBoost can't learn "
        "meaningful splits from it. But the raw signal is extremely strong (Spearman r=0.817 with "
        "actual OW at day -14).\n\n"
        "| Condition | Action |\n"
        "|---|---|\n"
        "| `TMDB_POP_D14 >= 25` | Force minimum **LARGE+** |\n"
        "| `TMDB_POP_D14 >= 15` AND `D7/D14 >= 1.3` | Force minimum **MID** |\n"
        "| Otherwise | No override (trust model) |"
    )

with col2:
    st.markdown(
        "**Holdout Validation (19 blind films)**\n\n"
        "| Metric | Without Override | With Rule C |\n"
        "|---|---|---|\n"
        "| Tier Accuracy | 63.2% | **84.2%** |\n"
        "| Overrides Applied | 0 | 4 |\n"
        "| Correct Overrides | — | **4/4 (100%)** |\n"
        "| Wrong Overrides | — | **0** |\n\n"
        "The **momentum gate** (D7/D14 >= 1.3) prevents false positives — e.g., "
        "PRIMATE had D14=24.6 but declining momentum (0.96), correctly kept as SMALL."
    )

st.divider()

st.header("V16 Feature Set (56 Features)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Static Features (36)")
    st.markdown("""
    **YouTube/Sentiment** (7): YT_COMMENTS, ENGAGEMENT_RATIO, SENTIMENT, 
    THEATRICAL_INTENT_PCT, STREAMING_INTENT_PCT, PASS_INTENT_PCT, NET_INTENT_PCT
    
    **Movie Attributes** (7): BUDGET, BUDGET_LOG, RUNTIME, TMDB_POPULARITY, 
    RELEASE_MONTH, IS_PEAK_SEASON, PREDECESSOR_OW_LOG
    
    **Star Power** (4): MAX_STAR_POWER, TOP2_STAR_POWER, AVG_STAR_POWER, 
    NUM_STARS_WITH_HISTORY
    
    **Genre** (5): ACTION_FRANCHISE, ANIMATION_FAMILY, HORROR, PRESTIGE, ORIGINAL
    
    **Rating** (4): G, PG, PG13, R
    
    **IP/Franchise** (5): KNOWN_IP_TIER, IP_HIGH_PROFILE, IP_MODERATE, IP_NICHE, IP_ORIGINAL
    
    **V16 New** (4): `IS_MAJOR_STUDIO`, `TMDB_POPULARITY_D14`, 
    `TMDB_POPULARITY_D7`, `TMDB_POP_MOMENTUM`
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
    st.subheader("V14-V16 (3-Tier) Solution")
    st.markdown("""
    - Combined into LARGE+ tier: **51 training films** in V16
    - LARGE+ accuracy improved dramatically
    - More robust classification boundary
    - TMDB override adds additional safety net for LARGE+ detection
    """)

show_cortex_badge()
