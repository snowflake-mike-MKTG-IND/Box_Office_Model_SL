"""
Page 4: Interactive Prediction Tool (Static Deployment)
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import os
import joblib
import gzip

st.set_page_config(page_title="Predictions", page_icon="🔮", layout="wide")

st.title("Interactive Prediction Tool")
st.subheader("Explore Model Predictions in Real-Time")

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ow_pipeline_v14_production.joblib.gz')
    with gzip.open(model_path, 'rb') as f:
        return joblib.load(f)

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Could not load model: {e}")

st.divider()

col1, col2 = st.columns([1, 2])

popularity_examples = {
    "Indie Release (e.g., Moonlight)": 25,
    "Limited Release (e.g., Lady Bird)": 75,
    "Mid-Tier Release (e.g., A Quiet Place)": 110,
    "Moderate Buzz (e.g., Knives Out)": 150,
    "High Anticipation (e.g., Dune)": 300,
    "Blockbuster Hype (e.g., Avengers)": 450
}

star_examples = {
    "Unknown Cast": (5.0, 3.0),
    "Rising Stars (e.g., Timothée Chalamet)": (25.0, 12.0),
    "Recognizable Lead (e.g., Ryan Gosling)": (40.0, 18.0),
    "A-List Lead (e.g., Tom Cruise)": (55.0, 25.0),
    "Ensemble A-List (e.g., Barbie cast)": (70.0, 45.0),
    "Mega Stars (e.g., Avengers cast)": (95.0, 60.0)
}

trends_examples = {
    "Minimal Interest (e.g., small indie)": (15, -5),
    "Steady Interest (e.g., mid-budget drama)": (35, 5),
    "Growing Buzz (e.g., Oppenheimer pre-release)": (60, 15),
    "Viral Moment (e.g., Barbie campaign)": (85, 35),
    "Peak Hype (e.g., Endgame finale)": (100, 50)
}

with col1:
    st.header("Input Features")
    
    st.subheader("Movie Attributes")
    budget = st.slider("Budget ($M)", 1, 300, 80, help="Production budget in millions")
    popularity_choice = st.selectbox("Buzz Level", list(popularity_examples.keys()), index=3)
    tmdb_pop = popularity_examples[popularity_choice]
    st.caption(f"→ Popularity Score: {tmdb_pop}")
    runtime = st.slider("Runtime (min)", 80, 200, 120)
    
    st.subheader("Timing")
    release_month = st.selectbox("Release Month", list(range(1, 13)), index=5)
    is_peak = st.checkbox("Peak Season (Summer/Holiday)", value=True)
    
    st.subheader("Genre")
    genre = st.selectbox("Primary Genre", 
                         ["Action Franchise", "Animation/Family", "Horror", "Prestige", "Drama/Comedy/Other"])
    st.caption("Model uses genre clusters, not individual genres")
    
    st.subheader("Rating")
    rating = st.selectbox("MPAA Rating", ["G", "PG", "PG-13", "R"], index=2)
    
    st.subheader("IP/Franchise")
    ip_tier = st.selectbox("IP Tier", 
                           ["Original", "Niche IP", "Moderate IP", "High-Profile IP"],
                           index=0)
    
    st.subheader("Star Power")
    star_choice = st.selectbox("Cast Level", list(star_examples.keys()), index=3)
    max_star, avg_star = star_examples[star_choice]
    st.caption(f"→ Star Power: {max_star:.0f} max, {avg_star:.0f} avg")
    
    st.subheader("Pre-Release Interest")
    trends_choice = st.selectbox("Google Trends Level", list(trends_examples.keys()), index=2)
    rolling_7d, velocity = trends_examples[trends_choice]
    st.caption(f"→ Trends: {rolling_7d} (7-day), {velocity:+d} velocity")
    
    st.subheader("YouTube/Sentiment")
    yt_comments = st.number_input("YouTube Comments", 0, 500000, 50000)
    sentiment = st.slider("Sentiment Score", -1.0, 1.0, 0.3)
    
    horizon = st.selectbox("Prediction Horizon", ["-14 days", "-7 days", "-3 days"], index=1)

with col2:
    st.header("Prediction Results")
    
    genre_flags = {
        'GENRE_ACTION_FRANCHISE': 1 if genre == "Action Franchise" else 0,
        'GENRE_ANIMATION_FAMILY': 1 if genre == "Animation/Family" else 0,
        'GENRE_HORROR': 1 if genre == "Horror" else 0,
        'GENRE_PRESTIGE': 1 if genre == "Prestige" else 0,
        'GENRE_ORIGINAL': 1 if genre == "Drama/Comedy/Other" else 0,
    }
    
    rating_flags = {
        'RATING_G': 1 if rating == "G" else 0,
        'RATING_PG': 1 if rating == "PG" else 0,
        'RATING_PG13': 1 if rating == "PG-13" else 0,
        'RATING_R': 1 if rating == "R" else 0,
    }
    
    ip_flags = {
        'IP_ORIGINAL': 1 if ip_tier == "Original" else 0,
        'IP_NICHE': 1 if ip_tier == "Niche IP" else 0,
        'IP_MODERATE': 1 if ip_tier == "Moderate IP" else 0,
        'IP_HIGH_PROFILE': 1 if ip_tier == "High-Profile IP" else 0,
        'KNOWN_IP_TIER': ["Original", "Niche IP", "Moderate IP", "High-Profile IP"].index(ip_tier),
    }
    
    if model_loaded:
        days_out = int(horizon.split()[0])
        horizon_model = model['models'][days_out]
        
        static_features = [
            yt_comments, sentiment / 10, sentiment,
            0.4, 0.3, 0.2, 0.1,
            release_month, 1 if is_peak else 0,
            genre_flags['GENRE_ACTION_FRANCHISE'], genre_flags['GENRE_ANIMATION_FAMILY'],
            genre_flags['GENRE_HORROR'], genre_flags['GENRE_PRESTIGE'], genre_flags['GENRE_ORIGINAL'],
            rating_flags['RATING_G'], rating_flags['RATING_PG'], 
            rating_flags['RATING_PG13'], rating_flags['RATING_R'],
            ip_flags['KNOWN_IP_TIER'], ip_flags['IP_HIGH_PROFILE'], 
            ip_flags['IP_MODERATE'], ip_flags['IP_NICHE'], ip_flags['IP_ORIGINAL'],
            max_star, (max_star + avg_star) / 2, avg_star, 3,
            budget * 1e6, runtime, tmdb_pop, np.log1p(budget * 1e6)
        ]
        
        trends_features = [
            rolling_7d * 0.8, rolling_7d * 0.9, rolling_7d,
            rolling_7d * 0.7, rolling_7d * 0.8, rolling_7d * 0.85,
            velocity * 0.7, velocity * 0.85, velocity,
            rolling_7d * 30, 15, rolling_7d * 1.2, 20,
            rolling_7d * ip_flags['IP_HIGH_PROFILE'], 
            rolling_7d * genre_flags['GENRE_ACTION_FRANCHISE'],
            rolling_7d * genre_flags['GENRE_HORROR'],
            sentiment * rolling_7d, max_star * rolling_7d / 100,
            max_star * ip_flags['IP_HIGH_PROFILE'], 0.1 * rolling_7d
        ]
        
        full_features = static_features + trends_features
        
        X_static = np.array(static_features).astype(float)
        X_full = np.array(full_features).astype(float)
        
        is_small = horizon_model['stage1_classifier'].predict([X_static])[0]
        stage1_proba = horizon_model['stage1_classifier'].predict_proba([X_static])[0]
        
        if is_small == 1:
            tier = 0
            stage2_proba = [0.5, 0.5]
        else:
            upper = horizon_model['stage2_classifier'].predict([X_static])[0]
            stage2_proba = horizon_model['stage2_classifier'].predict_proba([X_static])[0]
            tier = int(upper) + 1
        
        log_ow = horizon_model['regressors'][tier].predict([X_full])[0]
        ow_pred = np.expm1(log_ow) / 1e6
        
        tier_name = model['tier_names'][tier]
        tier_colors = {'SMALL': '#17becf', 'MID': '#9467bd', 'LARGE+': '#d62728'}
        
        st.subheader("Cascade Classification Flow")
        
        fig = go.Figure()
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=stage1_proba[0] * 100,
            title={'text': "Stage 1: NON-SMALL Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#2ca02c'},
                'steps': [
                    {'range': [0, 50], 'color': '#17becf'},
                    {'range': [50, 100], 'color': '#ff7f0e'}
                ],
                'threshold': {
                    'line': {'color': 'red', 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            },
            domain={'x': [0, 0.45], 'y': [0.5, 1]}
        ))
        
        if is_small == 0:
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=stage2_proba[1] * 100,
                title={'text': "Stage 2: LARGE+ Probability"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#ff7f0e'},
                    'steps': [
                        {'range': [0, 50], 'color': '#9467bd'},
                        {'range': [50, 100], 'color': '#d62728'}
                    ],
                    'threshold': {
                        'line': {'color': 'red', 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                },
                domain={'x': [0.55, 1], 'y': [0.5, 1]}
            ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric("Predicted Tier", tier_name)
        
        with col_b:
            st.metric("Predicted OW", f"${ow_pred:.1f}M")
        
        with col_c:
            tier_ranges = {
                'SMALL': ('$0', '$15M'),
                'MID': ('$15M', '$50M'),
                'LARGE+': ('$50M', '$200M+')
            }
            low, high = tier_ranges[tier_name]
            st.metric("Tier Range", f"{low} - {high}")
        
        st.divider()
        
        st.subheader("Prediction Breakdown")
        
        st.markdown(f"""
        | Stage | Decision | Probability |
        |-------|----------|-------------|
        | Stage 1 | {'SMALL' if is_small else 'NON-SMALL'} | {max(stage1_proba)*100:.1f}% |
        | Stage 2 | {'MID' if tier == 1 else 'LARGE+' if tier == 2 else 'N/A'} | {max(stage2_proba)*100:.1f}% |
        | **Final** | **{tier_name}** | - |
        """)
        
        st.divider()
        
        st.subheader("Confidence Range")
        
        mae_by_tier = {'SMALL': 3.2, 'MID': 8.7, 'LARGE+': 24.5}
        mae = mae_by_tier[tier_name]
        
        low_pred = max(0, ow_pred - mae)
        high_pred = ow_pred + mae
        
        fig_range = go.Figure()
        fig_range.add_trace(go.Bar(
            x=[ow_pred],
            y=['Prediction'],
            orientation='h',
            marker_color=tier_colors[tier_name],
            error_x=dict(type='data', array=[mae], arrayminus=[min(mae, ow_pred)]),
            text=[f'${ow_pred:.1f}M'],
            textposition='outside'
        ))
        fig_range.update_layout(
            height=150,
            xaxis_title='Opening Weekend ($M)',
            showlegend=False,
            xaxis_range=[0, max(high_pred * 1.2, 50)]
        )
        st.plotly_chart(fig_range, use_container_width=True)
        
        st.caption(f"Based on {tier_name} tier MAE of ${mae:.1f}M: Range ${low_pred:.1f}M - ${high_pred:.1f}M")
        
    else:
        st.warning("Model not loaded. Showing demo mode.")
        
        if budget > 150:
            tier_name = "LARGE+"
            ow_pred = budget * 0.8
        elif budget > 50:
            tier_name = "MID"
            ow_pred = budget * 0.4
        else:
            tier_name = "SMALL"
            ow_pred = budget * 0.15
        
        st.metric("Predicted Tier (Demo)", tier_name)
        st.metric("Predicted OW (Demo)", f"${ow_pred:.1f}M")
