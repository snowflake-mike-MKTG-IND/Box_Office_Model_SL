import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="Recent Predictions", page_icon="🎯", layout="wide")

st.title("🎯 Recent Weekend Predictions vs Actuals")
st.markdown("Tracking V14 model predictions against actual opening weekend results. *Data sourced from Box Office Mojo (The Numbers currently under maintenance).*")

st.divider()

WEEKEND_DATA = [
    {
        "weekend": "Weekend 6",
        "dates": "Feb 6-8, 2026",
        "movies": [
            {"movie": "Send Help", "studio": "Disney", "predicted_tier": "MID", "predicted_ow": 11.5, "actual_ow": 19.1, "actual_tier": "MID", "week": 2, "note": "Holdover (OW was W5)"},
            {"movie": "Solo Mio", "studio": "Angel", "predicted_tier": "SMALL", "predicted_ow": 5.8, "actual_ow": 7.0, "actual_tier": "SMALL", "week": 1, "note": "Faith-based"},
            {"movie": "Iron Lung", "studio": "Markiplier", "predicted_tier": "MID", "predicted_ow": 12.0, "actual_ow": 17.8, "actual_tier": "MID", "week": 2, "note": "Holdover (OW was W5)"},
            {"movie": "Stray Kids: dominATE", "studio": "Bleecker Street", "predicted_tier": "SMALL", "predicted_ow": 4.0, "actual_ow": 5.7, "actual_tier": "SMALL", "week": 1, "note": "Concert/event film"},
            {"movie": "Dracula", "studio": "Vertical", "predicted_tier": "SMALL", "predicted_ow": 6.5, "actual_ow": 4.4, "actual_tier": "SMALL", "week": 1, "note": ""},
        ]
    },
    {
        "weekend": "Weekend 7",
        "dates": "Feb 13-15, 2026",
        "movies": [
            {"movie": "Wuthering Heights", "studio": "Warner Bros.", "predicted_tier": "MID", "predicted_ow": 22.0, "actual_ow": 32.8, "actual_tier": "MID", "week": 1, "note": "Emerald Fennell film; overperformed"},
            {"movie": "GOAT", "studio": "Sony", "predicted_tier": "MID", "predicted_ow": 18.5, "actual_ow": 27.2, "actual_tier": "MID", "week": 1, "note": "Tom Brady doc; overperformed"},
            {"movie": "Crime 101", "studio": "Amazon MGM", "predicted_tier": "SMALL", "predicted_ow": 10.0, "actual_ow": 14.3, "actual_tier": "SMALL", "week": 1, "note": "George Clooney/Brad Pitt"},
        ]
    },
    {
        "weekend": "Weekend 8",
        "dates": "Feb 20-22, 2026",
        "movies": [
            {"movie": "I Can Only Imagine 2", "studio": "Lionsgate", "predicted_tier": "SMALL", "predicted_ow": 8.5, "actual_ow": 7.8, "actual_tier": "SMALL", "week": 1, "note": "Faith-based sequel"},
            {"movie": "How to Make a Killing", "studio": "A24", "predicted_tier": "SMALL", "predicted_ow": 4.0, "actual_ow": 3.5, "actual_tier": "SMALL", "week": 1, "note": ""},
            {"movie": "EPiC: Elvis Concert", "studio": "Neon", "predicted_tier": "SMALL", "predicted_ow": 2.0, "actual_ow": 3.2, "actual_tier": "SMALL", "week": 1, "note": "Concert film; limited release"},
        ]
    },
    {
        "weekend": "Weekend 9",
        "dates": "Feb 27-Mar 1, 2026",
        "movies": [
            {"movie": "Scream 7", "studio": "Paramount", "predicted_tier": "LARGE+", "predicted_ow": 55.0, "actual_ow": 63.6, "actual_tier": "LARGE+", "week": 1, "note": "Horror franchise; strong opening"},
            {"movie": "21 Pilots Concert", "studio": "Trafalgar", "predicted_tier": "SMALL", "predicted_ow": 3.0, "actual_ow": 4.3, "actual_tier": "SMALL", "week": 1, "note": "Concert/event film"},
        ]
    },
]

UPCOMING = [
    {
        "weekend": "Weekend 10",
        "dates": "Mar 6-8, 2026",
        "movies": [
            {"movie": "The Bride", "studio": "Warner Bros.", "predicted_tier": "SMALL", "predicted_ow": 8.3, "note": "Maggie Gyllenhaal; horror"},
            {"movie": "Novocaine", "studio": "Paramount", "predicted_tier": "SMALL", "predicted_ow": 9.0, "note": "Jack Quaid action-comedy"},
        ]
    },
]


all_movies = []
for w in WEEKEND_DATA:
    for m in w["movies"]:
        if m["week"] == 1:
            all_movies.append({
                "Weekend": w["weekend"],
                "Movie": m["movie"],
                "Studio": m["studio"],
                "Predicted Tier": m["predicted_tier"],
                "Predicted OW ($M)": m["predicted_ow"],
                "Actual OW ($M)": m["actual_ow"],
                "Actual Tier": m["actual_tier"],
                "Error ($M)": round(m["predicted_ow"] - m["actual_ow"], 1),
                "Tier Correct": "✅" if m["predicted_tier"] == m["actual_tier"] else "❌",
            })

df = pd.DataFrame(all_movies)

col1, col2, col3, col4 = st.columns(4)
tier_correct = sum(1 for m in all_movies if m["Tier Correct"] == "✅")
tier_total = len(all_movies)
mae = df["Error ($M)"].abs().mean()
mape = (df["Error ($M)"].abs() / df["Actual OW ($M)"]).mean() * 100

col1.metric("Movies Tracked", f"{tier_total}", f"Across {len(WEEKEND_DATA)} weekends")
col2.metric("Tier Accuracy", f"{tier_correct}/{tier_total}", f"{tier_correct/tier_total*100:.0f}%")
col3.metric("Mean Abs Error", f"${mae:.1f}M")
col4.metric("MAPE", f"{mape:.1f}%")

st.divider()

st.header("Predicted vs Actual Comparison")

fig = go.Figure()
fig.add_trace(go.Bar(
    name='Predicted OW',
    x=df['Movie'],
    y=df['Predicted OW ($M)'],
    marker_color='#636EFA',
    text=[f"${v:.1f}M" for v in df['Predicted OW ($M)']],
    textposition='outside',
))
fig.add_trace(go.Bar(
    name='Actual OW',
    x=df['Movie'],
    y=df['Actual OW ($M)'],
    marker_color='#00CC96',
    text=[f"${v:.1f}M" for v in df['Actual OW ($M)']],
    textposition='outside',
))
fig.update_layout(
    barmode='group',
    height=450,
    yaxis_title="Opening Weekend ($M)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=40),
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

st.header("Prediction Error by Movie")

colors = ['#EF553B' if e < 0 else '#636EFA' for e in df['Error ($M)']]
fig_err = go.Figure(go.Bar(
    x=df['Movie'],
    y=df['Error ($M)'],
    marker_color=colors,
    text=[f"${v:+.1f}M" for v in df['Error ($M)']],
    textposition='outside',
))
fig_err.add_hline(y=0, line_dash="dash", line_color="gray")
fig_err.update_layout(
    height=350,
    yaxis_title="Prediction Error ($M)",
    margin=dict(t=20),
    annotations=[dict(
        x=0.5, y=-0.15, xref="paper", yref="paper",
        text="Blue = Under-predicted (actual higher) | Red = Over-predicted (actual lower)",
        showarrow=False, font=dict(size=11, color="gray")
    )]
)
st.plotly_chart(fig_err, use_container_width=True)

st.divider()

st.header("Weekend-by-Weekend Breakdown")

for w in WEEKEND_DATA:
    with st.expander(f"**{w['weekend']}** — {w['dates']}", expanded=False):
        rows = []
        for m in w["movies"]:
            if m["week"] == 1:
                error = round(m["predicted_ow"] - m["actual_ow"], 1)
                tier_match = "✅" if m["predicted_tier"] == m["actual_tier"] else "❌"
                rows.append({
                    "Movie": m["movie"],
                    "Studio": m["studio"],
                    "Pred Tier": m["predicted_tier"],
                    "Actual Tier": m["actual_tier"],
                    "Tier Match": tier_match,
                    "Predicted ($M)": m["predicted_ow"],
                    "Actual ($M)": m["actual_ow"],
                    "Error ($M)": error,
                    "Note": m.get("note", ""),
                })
        wdf = pd.DataFrame(rows)
        st.dataframe(wdf, use_container_width=True, hide_index=True)

st.divider()

st.header("🔮 Upcoming Predictions")
st.caption("Predictions awaiting actual results")

for u in UPCOMING:
    st.subheader(f"{u['weekend']} — {u['dates']}")
    for m in u["movies"]:
        tier_color = {"SMALL": "🟠", "MID": "🟡", "LARGE+": "🟢"}.get(m["predicted_tier"], "⚪")
        st.markdown(f"- **{m['movie']}** ({m['studio']}) — {tier_color} {m['predicted_tier']} — **${m['predicted_ow']:.1f}M** predicted | {m.get('note', '')}")

st.divider()

st.header("Accuracy by Tier")

tier_stats = df.groupby("Predicted Tier").agg(
    Count=("Movie", "count"),
    MAE=("Error ($M)", lambda x: x.abs().mean()),
    Tier_Correct=("Tier Correct", lambda x: (x == "✅").sum()),
).reset_index()
tier_stats["Tier Accuracy"] = (tier_stats["Tier_Correct"] / tier_stats["Count"] * 100).round(0).astype(int).astype(str) + "%"
tier_stats["MAE"] = tier_stats["MAE"].round(1)

col1, col2 = st.columns(2)
with col1:
    st.dataframe(tier_stats[["Predicted Tier", "Count", "MAE", "Tier Accuracy"]], use_container_width=True, hide_index=True)
with col2:
    fig_tier = px.bar(tier_stats, x="Predicted Tier", y="MAE", color="Predicted Tier",
                      color_discrete_map={"SMALL": "#FF7F0E", "MID": "#FFD700", "LARGE+": "#00CC96"},
                      text="MAE")
    fig_tier.update_traces(texttemplate='$%{text:.1f}M', textposition='outside')
    fig_tier.update_layout(height=300, showlegend=False, margin=dict(t=20))
    st.plotly_chart(fig_tier, use_container_width=True)

st.divider()
st.caption("💡 To add new weekends: update the WEEKEND_DATA list at the top of this file and move entries from UPCOMING to WEEKEND_DATA once actuals are available. Data source: Box Office Mojo (The Numbers under maintenance).")
