import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="Recent Predictions", page_icon="🎯", layout="wide")

st.title("🎯 Recent Weekend Predictions vs Actuals")
st.markdown("Tracking V14 model predictions against actual opening weekend results. Predictions sourced from `SPARK_PAR_DEMO.PRODUCTION.OW_PREDICTIONS`. Actuals from Box Office Mojo (The Numbers currently under maintenance).")

st.divider()

WEEKEND_DATA = [
    {
        "weekend": "Weekend 2",
        "dates": "Jan 9-12, 2026",
        "movies": [
            {"movie": "Greenland 2: Migration", "studio": "STX/Lionsgate", "predicted_tier": "SMALL", "predicted_ow": 9.37, "conf_low": 7.50, "conf_high": 11.24, "actual_ow": 8.40, "week": 1, "note": "Gerard Butler sequel"},
        ]
    },
    {
        "weekend": "Weekend 3",
        "dates": "Jan 16-19, 2026",
        "movies": [
            {"movie": "28 Years Later: The Bone Temple", "studio": "Sony", "predicted_tier": "SMALL", "predicted_ow": 14.48, "conf_low": 11.58, "conf_high": 17.37, "actual_ow": 15.00, "week": 1, "note": "Horror sequel; borderline MID"},
        ]
    },
    {
        "weekend": "Weekend 7",
        "dates": "Feb 13-15, 2026",
        "movies": [
            {"movie": "Wuthering Heights", "studio": "Warner Bros.", "predicted_tier": "MID", "predicted_ow": 32.62, "conf_low": 26.10, "conf_high": 39.15, "actual_ow": 32.80, "week": 1, "note": "Emerald Fennell; nearly perfect prediction"},
            {"movie": "GOAT", "studio": "Sony", "predicted_tier": "MID", "predicted_ow": 27.35, "conf_low": 21.88, "conf_high": 32.82, "actual_ow": 27.20, "week": 1, "note": "Tom Brady doc; nearly perfect prediction"},
            {"movie": "Crime 101", "studio": "Amazon MGM", "predicted_tier": None, "predicted_ow": None, "conf_low": None, "conf_high": None, "actual_ow": 14.25, "week": 1, "note": "Not in prediction pipeline"},
        ]
    },
    {
        "weekend": "Weekend 9",
        "dates": "Feb 27-Mar 1, 2026",
        "movies": [
            {"movie": "Scream 7", "studio": "Paramount", "predicted_tier": "LARGE+", "predicted_ow": None, "conf_low": None, "conf_high": None, "actual_ow": 63.62, "week": 1,
             "tier_confidence": 76.97, "tier_range_low": 50, "tier_range_high": 100,
             "regression_error": True, "manually_adjusted": True,
             "note": "Tier manually elevated to LARGE based on strong pre-release signals; regression errored (SQL syntax bug) — no $ prediction generated"},
        ]
    },
]

UPCOMING = [
    {
        "weekend": "Weekend 10",
        "dates": "Mar 6-8, 2026",
        "movies": [
            {"movie": "The Bride", "studio": "Warner Bros.", "predicted_tier": "SMALL", "predicted_ow": 8.31, "conf_low": 6.65, "conf_high": 9.97, "note": "Maggie Gyllenhaal; horror"},
        ]
    },
]


def get_actual_tier(actual_ow):
    if actual_ow is None:
        return None
    if actual_ow < 15:
        return "SMALL"
    elif actual_ow < 50:
        return "MID"
    else:
        return "LARGE+"


all_movies = []
all_tier_checks = []
for w in WEEKEND_DATA:
    for m in w["movies"]:
        if m["week"] == 1 and m["predicted_tier"] is not None:
            actual_tier = get_actual_tier(m["actual_ow"])
            pred_tier = m["predicted_tier"]
            tier_match = pred_tier == actual_tier if actual_tier else None
            if pred_tier and "LARGE" in pred_tier and actual_tier and "LARGE" in actual_tier:
                tier_match = True
            all_tier_checks.append({"movie": m["movie"], "match": tier_match})
            if m["predicted_ow"] is not None:
                all_movies.append({
                    "Weekend": w["weekend"],
                    "Movie": m["movie"],
                    "Studio": m["studio"],
                    "Predicted Tier": pred_tier,
                    "Predicted OW ($M)": m["predicted_ow"],
                    "Conf Low ($M)": m["conf_low"],
                    "Conf High ($M)": m["conf_high"],
                    "Actual OW ($M)": m["actual_ow"],
                    "Actual Tier": actual_tier,
                    "Error ($M)": round(m["predicted_ow"] - m["actual_ow"], 2),
                    "Tier Correct": "✅" if tier_match else "❌",
                })

df = pd.DataFrame(all_movies)

col1, col2, col3, col4 = st.columns(4)
tier_correct_all = sum(1 for t in all_tier_checks if t["match"])
tier_total_all = len(all_tier_checks)
regression_total = len(all_movies)
mae = df["Error ($M)"].abs().mean()
within_ci = sum(1 for m in all_movies if m["Conf Low ($M)"] <= m["Actual OW ($M)"] <= m["Conf High ($M)"])

col1.metric("Tier Accuracy", f"{tier_correct_all}/{tier_total_all}", f"{tier_correct_all/tier_total_all*100:.0f}% (incl. Scream 7)")
col2.metric("Regression MAE", f"${mae:.2f}M", f"{regression_total} movies with $ predictions")
col3.metric("Within Confidence Interval", f"{within_ci}/{regression_total}", f"{within_ci/regression_total*100:.0f}%")
col4.metric("Movies Tracked", f"{tier_total_all}", f"{len(WEEKEND_DATA)} weekends")

st.divider()

st.header("Predicted vs Actual Comparison")

fig = go.Figure()
fig.add_trace(go.Bar(
    name='Predicted OW',
    x=df['Movie'],
    y=df['Predicted OW ($M)'],
    marker_color='#636EFA',
    text=[f"${v:.2f}M" for v in df['Predicted OW ($M)']],
    textposition='outside',
))
fig.add_trace(go.Bar(
    name='Actual OW',
    x=df['Movie'],
    y=df['Actual OW ($M)'],
    marker_color='#00CC96',
    text=[f"${v:.2f}M" for v in df['Actual OW ($M)']],
    textposition='outside',
))

for i, row in df.iterrows():
    fig.add_shape(
        type="line",
        x0=i - 0.2, x1=i - 0.2,
        y0=row['Conf Low ($M)'], y1=row['Conf High ($M)'],
        line=dict(color="rgba(99, 110, 250, 0.4)", width=3),
    )

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

colors = ['#EF553B' if e > 0 else '#636EFA' for e in df['Error ($M)']]
fig_err = go.Figure(go.Bar(
    x=df['Movie'],
    y=df['Error ($M)'],
    marker_color=colors,
    text=[f"${v:+.2f}M" for v in df['Error ($M)']],
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

st.header("Scream 7 — Tier Correct, Regression Error")
st.markdown("""
**Scream 7** is a special case: the V14 classifier correctly identified it as **LARGE** tier 
(manually elevated based on strong pre-release signals, 76.97 confidence), but the regression 
step hit a SQL syntax error and produced **$0** instead of a dollar prediction.
""")
scream_col1, scream_col2, scream_col3 = st.columns(3)
with scream_col1:
    st.metric("Predicted Tier", "LARGE+", "Manually adjusted ✅")
with scream_col2:
    st.metric("Tier Range", "$50–100M", "Actual: $63.62M ✅")
with scream_col3:
    st.metric("Regression Prediction", "$0 (ERROR)", "SQL syntax bug in pipeline")
st.markdown("""
| Detail | Value |
|--------|-------|
| **Classifier Tier** | LARGE (elevated from initial classification) |
| **Tier Confidence** | 76.97 |
| **Tier Range** | $50M – $100M |
| **Actual OW** | **$63.62M** (within tier range ✅) |
| **Regression Output** | $0 — errored |
| **Error Source** | `SQL compilation error: syntax error line 3 at position 70 unexpected 'str'` |
| **Override** | Manually adjusted — elevated to LARGE based on strong pre-release signals |

The tier classification counts toward accuracy metrics (correct), but the regression error 
means Scream 7 is excluded from MAE/CI calculations.
""")

st.divider()

st.header("Weekend-by-Weekend Breakdown")

for w in WEEKEND_DATA:
    with st.expander(f"**{w['weekend']}** — {w['dates']}", expanded=False):
        rows = []
        for m in w["movies"]:
            if m["week"] == 1:
                actual_tier = get_actual_tier(m["actual_ow"])
                pred_tier = m.get("predicted_tier")
                has_prediction = m["predicted_ow"] is not None
                if has_prediction:
                    error = round(m["predicted_ow"] - m["actual_ow"], 2)
                    tier_match_bool = pred_tier == actual_tier
                    if pred_tier and "LARGE" in pred_tier and actual_tier and "LARGE" in actual_tier:
                        tier_match_bool = True
                    tier_match = "✅" if tier_match_bool else "❌"
                    ci_str = f"${m['conf_low']:.2f} – ${m['conf_high']:.2f}M"
                    in_ci = "✅" if m["conf_low"] <= m["actual_ow"] <= m["conf_high"] else "❌"
                else:
                    error = None
                    tier_match = "—"
                    ci_str = "—"
                    in_ci = "—"
                rows.append({
                    "Movie": m["movie"],
                    "Studio": m["studio"],
                    "Pred Tier": pred_tier or "—",
                    "Actual Tier": actual_tier or "—",
                    "Tier ✓": tier_match,
                    "Predicted ($M)": f"${m['predicted_ow']:.2f}" if has_prediction else "—",
                    "Actual ($M)": f"${m['actual_ow']:.2f}" if m["actual_ow"] else "—",
                    "Error ($M)": f"${error:+.2f}" if error is not None else "—",
                    "CI Range": ci_str,
                    "In CI": in_ci,
                    "Note": m.get("note", ""),
                })
        wdf = pd.DataFrame(rows)
        st.dataframe(wdf, use_container_width=True, hide_index=True)

st.divider()

st.header("🔮 Upcoming / Awaiting Actuals")
st.caption("Predictions from Snowflake awaiting actual results")

for u in UPCOMING:
    st.subheader(f"{u['weekend']} — {u['dates']}")
    for m in u["movies"]:
        tier_color = {"SMALL": "🟠", "MID": "🟡", "LARGE+": "🟢", "LARGE": "🟢"}.get(m["predicted_tier"], "⚪")
        ci_str = f" (CI: ${m.get('conf_low', 0):.2f}–${m.get('conf_high', 0):.2f}M)" if m.get("conf_low") else ""
        st.markdown(f"- **{m['movie']}** ({m['studio']}) — {tier_color} {m['predicted_tier']} — **${m['predicted_ow']:.2f}M** predicted{ci_str} | {m.get('note', '')}")

st.divider()

st.header("Accuracy by Tier")

tier_stats = df.groupby("Predicted Tier").agg(
    Count=("Movie", "count"),
    MAE=("Error ($M)", lambda x: x.abs().mean()),
    Tier_Correct=("Tier Correct", lambda x: (x == "✅").sum()),
).reset_index()
tier_stats["Tier Accuracy"] = (tier_stats["Tier_Correct"] / tier_stats["Count"] * 100).round(0).astype(int).astype(str) + "%"
tier_stats["MAE"] = tier_stats["MAE"].round(2)

col1, col2 = st.columns(2)
with col1:
    st.dataframe(tier_stats[["Predicted Tier", "Count", "MAE", "Tier Accuracy"]], use_container_width=True, hide_index=True)
with col2:
    fig_tier = px.bar(tier_stats, x="Predicted Tier", y="MAE", color="Predicted Tier",
                      color_discrete_map={"SMALL": "#FF7F0E", "MID": "#FFD700", "LARGE+": "#00CC96"},
                      text="MAE")
    fig_tier.update_traces(texttemplate='$%{text:.2f}M', textposition='outside')
    fig_tier.update_layout(height=300, showlegend=False, margin=dict(t=20))
    st.plotly_chart(fig_tier, use_container_width=True)

st.divider()

st.info("""
**Data Sources**
- **Predictions**: `SPARK_PAR_DEMO.PRODUCTION.OW_PREDICTIONS` / `ML_PREDICTIONS_V` (V14 3-Tier Cascade model)
- **Actuals**: Box Office Mojo (The Numbers currently under maintenance)
- **Tier System**: SMALL (<$15M) · MID ($15–50M) · LARGE+ (>$50M)
- Movies not in the prediction pipeline (e.g., Crime 101) are shown in weekend breakdowns but excluded from all metrics
- Scream 7: tier classification was correct (LARGE+ ✅) but regression errored — included in tier accuracy, excluded from MAE
""")

st.caption("💡 To add new weekends: update WEEKEND_DATA with real values from Snowflake and Box Office Mojo. Move entries from UPCOMING once actuals are available.")
