"""
Page 5: Error Analysis
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import os
from cortex_badge import show_cortex_badge

st.set_page_config(page_title="Error Analysis", page_icon="⚠️", layout="wide")

st.title("Error Analysis")
st.subheader("V17 Model Limitations and Edge Cases")

st.divider()

st.header("Biggest Prediction Misses")

misses_data = pd.DataFrame({
    'Movie': ['Taylor Swift: Eras Tour', 'Super Mario Bros', 'Barbie (early pred)',
              'The Boy and the Heron', 'Five Nights at Freddys', 'Oppenheimer',
              'Ant-Man Quantumania', 'The Flash', 'Indiana Jones 5', 'Haunted Mansion'],
    'Actual ($M)': [92.8, 146.4, 162.0, 12.8, 78.0, 82.5, 106.1, 55.0, 60.4, 24.2],
    'Predicted ($M)': [35.0, 95.0, 110.0, 35.0, 45.0, 55.0, 140.0, 90.0, 95.0, 45.0],
    'Error ($M)': [57.8, 51.4, 52.0, -22.2, 33.0, 27.5, -33.9, -35.0, -34.6, -20.8],
    'Tier': ['LARGE+', 'LARGE+', 'LARGE+', 'SMALL', 'LARGE+', 'LARGE+',
             'LARGE+', 'LARGE+', 'LARGE+', 'MID'],
    'Reason': ['Concert film (event)', 'Nintendo IP underestimated', 'Cultural phenomenon',
               'Anime (different audience)', 'Horror overperformed', 'Nolan event film',
               'Franchise fatigue', 'DC fatigue', 'Legacy sequel underperformed', 'Disney fatigue']
})

misses_data['Abs Error'] = np.abs(misses_data['Error ($M)'])
misses_data = misses_data.sort_values('Abs Error', ascending=False)

fig_misses = px.bar(misses_data, x='Movie', y='Error ($M)',
                    color='Error ($M)', color_continuous_scale='RdBu_r',
                    hover_data=['Actual ($M)', 'Predicted ($M)', 'Reason'])
fig_misses.update_layout(height=400, xaxis_tickangle=45)
st.plotly_chart(fig_misses, use_container_width=True)

st.divider()

st.header("What V17's TMDB Override Addresses")

col1, col2 = st.columns(2)

with col1:
    st.success("**Under-Prediction Pattern (V15's Biggest Weakness)**")
    st.markdown("""
    V15 had 3 major tier misses — all under-predictions where breakout films 
    were classified too low:
    
    | Film | V15 Prediction | Actual | Miss |
    |------|---------------|--------|------|
    | Project Hail Mary | MID $26M | LARGE+ $81M | -$54M |
    | Hoppers | SMALL $10M | MID $45M | -$35M |
    | Reminders of Him | SMALL $7M | MID $18M | -$11M |
    
    These films all had **high TMDB D14 popularity** (18-27) that the base model 
    couldn't leverage (only ~30 training films with TMDB data).
    """)

with col2:
    st.info("**How Rule C Helps**")
    st.markdown("""
    The TMDB override is specifically designed to catch under-predictions:
    
    - **Only raises tiers, never lowers** — can't make over-predictions worse
    - **High D14 popularity** strongly correlates with LARGE+ actual OW (r=0.817)
    - **Momentum gate** (D7/D14 >= 1.3) prevents false positives on declining-hype films
    - **Holdout validation**: 4/4 correct overrides, 0 wrong
    
    **Still not addressed**: Over-prediction from franchise fatigue 
    (Ant-Man 3, Flash, etc.) — this remains a known limitation.
    """)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.header("Under-Predictions")
    st.caption("Model predicted too low")
    st.markdown("""
    | Pattern | Example | Why Model Fails |
    |---------|---------|-----------------|
    | **Event Films** | Taylor Swift, BTS | Not theatrical releases |
    | **Gaming IP** | Mario, Pokemon | Limited historical data |
    | **Cultural Moments** | Barbie, Oppenheimer | Zeitgeist unpredictable |
    | **Horror Breakouts** | FNAF, M3GAN | Genre can overperform |
    """)

with col2:
    st.header("Over-Predictions")
    st.caption("Model predicted too high")
    st.markdown("""
    | Pattern | Example | Why Model Fails |
    |---------|---------|-----------------|
    | **Franchise Fatigue** | Ant-Man 3, Flash | Prior success != future |
    | **Legacy Sequels** | Indiana Jones 5 | Audience moved on |
    | **Brand Erosion** | Haunted Mansion | Disney fatigue |
    | **Competition** | The Marvels | Crowded release |
    """)

st.divider()

st.header("Error by Feature Value")

st.subheader("Budget vs Prediction Error (Training Set)")

_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'training_predictions_v17.json')
with open(_data_path) as _f:
    _training_preds = json.load(_f)

df_budget_error = pd.DataFrame({
    'Budget ($M)': [r['budget_m'] for r in _training_preds],
    'Prediction Error ($M)': [r['error_m'] for r in _training_preds],
    'Movie': [r['movie_title'] for r in _training_preds],
})

fig_budget = px.scatter(df_budget_error, x='Budget ($M)', y='Prediction Error ($M)',
                        hover_data=['Movie'],
                        trendline='lowess', trendline_options=dict(frac=0.3))
fig_budget.add_hline(y=0, line_dash="dash", line_color="red")
fig_budget.update_layout(height=400)
st.plotly_chart(fig_budget, use_container_width=True)

st.caption("Training-set predictions — errors are smaller than out-of-sample. Pattern shows how budget relates to prediction direction.")
st.info("**Insight**: Model tends to over-predict for very high budget films (>$150M) and under-predict for low budget films that break out.")

st.divider()

st.header("Known Model Limitations")

col1, col2 = st.columns(2)

with col1:
    st.error("**Model Struggles With**")
    st.markdown("""
    1. **Concert/Event Films**
       - Taylor Swift, BTS, Beyonce
       - No trailer data pattern
       - Different audience behavior
    
    2. **Anime Releases**
       - Demon Slayer, Dragon Ball
       - Excluded from training (HOLDOUT schema)
       - Different marketing/fanbase
    
    3. **Cultural Phenomena**
       - Unpredictable viral moments
       - Barbenheimer effect
       - Social media-driven interest
    
    4. **Franchise Fatigue**
       - Model uses prior film OW
       - Doesn't detect declining interest
       - DC/Disney struggles
    """)

with col2:
    st.success("**Model Excels At**")
    st.markdown("""
    1. **Standard Studio Releases**
       - Typical marketing campaigns
       - Predictable Google Trends patterns
       - Historical comparables exist
    
    2. **Established Franchises (stable)**
       - Marvel (when not fatigued)
       - Horror sequels
       - Animation franchises
    
    3. **Mid-Range Films**
       - $15-50M opening weekend
       - Most predictable tier
       - Largest training sample
    
    4. **Genre Films with Precedent**
       - Horror with clear comps
       - Action franchises
       - Family animation
    """)

st.divider()

st.header("Improvement Roadmap")

st.markdown("""
| Approach | Status | Expected Impact | Notes |
|----------|--------|-----------------|-------|
| **More Training Data** | Ongoing | High | 277 films and growing with each weekend |
| **TMDB Override (Rule C)** | **V16+ Deployed** | High | Addresses under-prediction of breakouts |
| **IS_MAJOR_STUDIO Feature** | **V16+ Deployed** | Medium | Helps size Disney/Universal/WB releases |
| **Longer Trends Windows** | **V17 Deployed** | Medium | ROLLING_14D, ROLLING_21D, TRENDS_EARLIEST |
| **Add Social Media Features** | Future | Medium | TikTok, Twitter engagement |
| **Franchise Fatigue Detection** | Future | Medium | Declining audience sentiment signals |
| **D-3 TMDB Override** | Future (V18) | Medium | Within-tier regressor correction at D-3 |

**V17 Strategy**: Extended rolling window features improve early-horizon predictions.
ROLLING_21D ranks #4 in SMALL regressor at -14d. Live validation ongoing.
""")

st.divider()

st.header("Error Distribution by Category (Training Set)")

_genre_groups = {}
for r in _training_preds:
    g = r['genre']
    _genre_groups.setdefault(g, []).append(abs(r['error_m']))

error_by_genre = pd.DataFrame([
    {'Category': g, 'MAE ($M)': round(np.mean(errs), 2), 'Sample Size': len(errs)}
    for g, errs in _genre_groups.items() if g != 'Other'
]).sort_values('MAE ($M)', ascending=False)

fig_genre_error = px.bar(error_by_genre, x='Category', y='MAE ($M)',
                         color='MAE ($M)', color_continuous_scale='Reds',
                         text='MAE ($M)', hover_data=['Sample Size'])
fig_genre_error.update_traces(texttemplate='$%{text:.1f}M', textposition='outside')
fig_genre_error.update_layout(height=350)
st.plotly_chart(fig_genre_error, use_container_width=True)

st.caption("Training-set MAE by genre. Horror has lowest MAE due to consistent audience behavior and predictable ceiling.")

show_cortex_badge()
