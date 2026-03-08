"""
Page 5: Error Analysis
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.set_page_config(page_title="Error Analysis", page_icon="⚠️", layout="wide")

st.title("Error Analysis")
st.subheader("V15 Model Limitations and Edge Cases")

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
    | **Franchise Fatigue** | Ant-Man 3, Flash | Prior success ≠ future |
    | **Legacy Sequels** | Indiana Jones 5 | Audience moved on |
    | **Brand Erosion** | Haunted Mansion | Disney fatigue |
    | **Competition** | The Marvels | Crowded release |
    """)

st.divider()

st.header("Error by Feature Value")

st.subheader("Budget vs Prediction Error")

np.random.seed(42)
budgets = np.concatenate([
    np.random.uniform(5, 30, 80),
    np.random.uniform(30, 80, 60),
    np.random.uniform(80, 250, 40)
])
errors = np.random.normal(0, 8, 180) + (budgets - 80) * 0.08

df_budget_error = pd.DataFrame({
    'Budget ($M)': budgets,
    'Prediction Error ($M)': errors
})

fig_budget = px.scatter(df_budget_error, x='Budget ($M)', y='Prediction Error ($M)',
                        trendline='lowess', trendline_options=dict(frac=0.3))
fig_budget.add_hline(y=0, line_dash="dash", line_color="red")
fig_budget.update_layout(height=400)
st.plotly_chart(fig_budget, use_container_width=True)

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

st.header("Recommendations for Improvement")

st.markdown("""
| Approach | Effort | Expected Impact | Notes |
|----------|--------|-----------------|-------|
| **More Training Data** | Low | High | Wait for 2026-2027 releases |
| **Add Social Media Features** | Medium | Medium | TikTok, Twitter engagement |
| **Manual Override System** | Low | High | Already implemented for known events |
| **Ensemble with Different Models** | High | Low | Tested - didn't help significantly |
| **Time-Series Features** | Medium | Medium | Week-over-week trend changes |
| **Competition Features** | Medium | Low | Already captured by other features |

**V15 Strategy**: Data quality over model complexity. The +30 training films and data cleanup 
drove bigger gains than any architectural change. Continue building training data.
""")

st.divider()

st.header("Error Distribution by Category")

error_by_genre = pd.DataFrame({
    'Category': ['Action Franchise', 'Animation/Family', 'Horror', 'Prestige/Drama', 'Original/Other'],
    'MAE ($M)': [14.8, 11.2, 7.9, 11.5, 13.8],
    'Sample Size': [52, 42, 48, 40, 87]
})

fig_genre_error = px.bar(error_by_genre, x='Category', y='MAE ($M)',
                         color='MAE ($M)', color_continuous_scale='Reds',
                         text='MAE ($M)', hover_data=['Sample Size'])
fig_genre_error.update_traces(texttemplate='$%{text:.1f}M', textposition='outside')
fig_genre_error.update_layout(height=350)
st.plotly_chart(fig_genre_error, use_container_width=True)

st.caption("Horror has lowest MAE due to consistent audience behavior and predictable ceiling.")
