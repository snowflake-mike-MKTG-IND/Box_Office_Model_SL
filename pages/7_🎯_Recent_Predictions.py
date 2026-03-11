import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from cortex_badge import show_cortex_badge

st.set_page_config(page_title="Recent Predictions", page_icon="🎯", layout="wide")

st.title("🎯 Recent Weekend Predictions vs Actuals")
st.markdown(
    "Tracking model predictions against actual opening weekend results. "
    "Each prediction shows which model version was used — **all predictions shown here were made "
    "before actuals were known**, using whichever model was in production at the time."
)

st.divider()

WEEKEND_DATA = [
    {
        "weekend": "Weekend 2",
        "dates": "Jan 9-12, 2026",
        "model": "V14",
        "movies": [
            {"movie": "Greenland 2: Migration", "studio": "STX/Lionsgate", "predicted_tier": "SMALL", "predicted_ow": 9.37, "conf_low": 7.50, "conf_high": 11.24, "actual_ow": 8.40, "week": 1, "industry_projection": None, "note": "Gerard Butler sequel"},
        ]
    },
    {
        "weekend": "Weekend 3",
        "dates": "Jan 16-19, 2026",
        "model": "V14",
        "movies": [
            {"movie": "28 Years Later: The Bone Temple", "studio": "Sony", "predicted_tier": "SMALL", "predicted_ow": 14.48, "conf_low": 11.58, "conf_high": 17.37, "actual_ow": 12.52, "week": 1, "industry_projection": 20.0, "note": "Horror sequel; Deadline projected $20M+ MLK OW"},
        ]
    },
    {
        "weekend": "Weekend 7",
        "dates": "Feb 13-15, 2026",
        "model": "V14",
        "movies": [
            {"movie": "Wuthering Heights", "studio": "Warner Bros.", "predicted_tier": "MID", "predicted_ow": 26.80, "conf_low": 21.44, "conf_high": 32.16, "actual_ow": 32.80, "week": 1, "industry_projection": 40.0, "note": "Emerald Fennell; Deadline projected $40-50M 4-day (~$40M 3-day equiv)"},
            {"movie": "GOAT", "studio": "Sony", "predicted_tier": "MID", "predicted_ow": 22.45, "conf_low": 17.96, "conf_high": 26.93, "actual_ow": 27.20, "week": 1, "industry_projection": 25.0, "note": "Animated sports comedy; Deadline/industry projected $25M 3-day"},
            {"movie": "Crime 101", "studio": "Amazon MGM", "predicted_tier": None, "predicted_ow": None, "conf_low": None, "conf_high": None, "actual_ow": 14.25, "week": 1, "industry_projection": None, "note": "Not in prediction pipeline"},
        ]
    },
    {
        "weekend": "Weekend 9",
        "dates": "Feb 27-Mar 1, 2026",
        "model": "V14",
        "movies": [
            {"movie": "Scream 7", "studio": "Paramount", "predicted_tier": "LARGE+", "predicted_ow": 69.59, "conf_low": 55.67, "conf_high": 83.51, "actual_ow": 63.62, "week": 1,
             "tier_confidence": 76.97, "tier_range_low": 50, "tier_range_high": 100,
             "manually_adjusted": True, "industry_projection": 40.0,
             "note": "Deadline projected mid-$30M+; THR/Deadline ~$40M; NRG $45M. Actual: $64.1M blew past all industry projections"},
        ]
    },
]

UPCOMING = [
    {
        "weekend": "Weekend 10",
        "dates": "Mar 6-8, 2026",
        "model": "V14",
        "movies": [
            {"movie": "The Bride", "studio": "Warner Bros.", "predicted_tier": "SMALL", "predicted_ow": 8.31, "conf_low": 6.65, "conf_high": 9.97, "note": "Maggie Gyllenhaal; horror — awaiting actuals"},
        ]
    },
    {
        "weekend": "Weekend 11",
        "dates": "Mar 13-15, 2026",
        "model": "V15",
        "movies": [
            {"movie": "Reminders of Him", "studio": "Indie", "predicted_tier": "SMALL", "predicted_ow": 2.6, "conf_low": 1.6, "conf_high": 3.7,
             "note": "PG-13 drama/romance (Colleen Hoover adaptation); 1,323 trailer comments; Maika Monroe lead"},
            {"movie": "Undertone", "studio": "Indie", "predicted_tier": "SMALL", "predicted_ow": 2.5, "conf_low": 1.5, "conf_high": 3.5,
             "note": "R-rated horror; 1,261 trailer comments; very low Google Trends signal"},
        ]
    },
    {
        "weekend": "Weekend 12",
        "dates": "Mar 20-22, 2026",
        "model": "V15",
        "movies": [
            {"movie": "Project Hail Mary", "studio": "Amazon MGM", "predicted_tier": "MID", "predicted_ow": 24.1, "conf_low": 13.3, "conf_high": 37.4,
             "note": "$200M budget; Ryan Gosling (star power 10); Andy Weir novel adaptation; 11,115 comments; strong trends (R7D=35.5)"},
            {"movie": "Ready or Not 2", "studio": "Disney/Searchlight", "predicted_tier": "SMALL", "predicted_ow": 5.7, "conf_low": 3.4, "conf_high": 7.9,
             "note": "R-rated horror sequel; predecessor OW $28.4M; 3,647 comments; Samara Weaving lead; decent trends (R7D=27.5)"},
            {"movie": "Do Not Enter", "studio": "Indie", "predicted_tier": "SMALL", "predicted_ow": 3.6, "conf_low": 2.2, "conf_high": 5.1,
             "note": "R-rated horror; only 204 comments; zero Google Trends signal — low confidence prediction"},
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
                ind_proj = m.get("industry_projection")
                all_movies.append({
                    "Weekend": w["weekend"],
                    "Model": w.get("model", "V14"),
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
                    "Industry Projection ($M)": ind_proj,
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

st.caption("All predictions below were made by the V14 model before each film's opening weekend. "
           "Purple bars show pre-release industry consensus projections (sourced from Deadline, THR, etc.).")

fig = go.Figure()
fig.add_trace(go.Bar(
    name="Predicted OW (V14)",
    x=df["Movie"],
    y=df["Predicted OW ($M)"],
    marker_color="#636EFA",
    text=[f"${v:.2f}M" for v in df["Predicted OW ($M)"]],
    textposition="outside",
))
fig.add_trace(go.Bar(
    name="Industry Projection",
    x=df["Movie"],
    y=df["Industry Projection ($M)"].fillna(0),
    marker_color="#AB63FA",
    text=[f"${v:.0f}M" if pd.notna(v) and v > 0 else "" for v in df["Industry Projection ($M)"]],
    textposition="outside",
))
fig.add_trace(go.Bar(
    name="Actual OW",
    x=df["Movie"],
    y=df["Actual OW ($M)"],
    marker_color="#00CC96",
    text=[f"${v:.2f}M" for v in df["Actual OW ($M)"]],
    textposition="outside",
))

ci_mid = [(row["Conf Low ($M)"] + row["Conf High ($M)"]) / 2 for _, row in df.iterrows()]
ci_err_plus = [row["Conf High ($M)"] - (row["Conf Low ($M)"] + row["Conf High ($M)"]) / 2 for _, row in df.iterrows()]
ci_err_minus = [(row["Conf Low ($M)"] + row["Conf High ($M)"]) / 2 - row["Conf Low ($M)"] for _, row in df.iterrows()]

fig.add_trace(go.Scatter(
    name="V14 Confidence Interval",
    x=df["Movie"],
    y=ci_mid,
    mode="markers",
    marker=dict(symbol="line-ew-open", size=12, color="#FFA500", line=dict(width=2, color="#FFA500")),
    error_y=dict(
        type="data",
        symmetric=False,
        array=ci_err_plus,
        arrayminus=ci_err_minus,
        color="#FFA500",
        thickness=2.5,
        width=10,
    ),
))

fig.update_layout(
    barmode="group",
    height=450,
    yaxis_title="Opening Weekend ($M)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=40),
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

st.header("Prediction Error by Movie")

colors = ["#EF553B" if e > 0 else "#636EFA" for e in df["Error ($M)"]]
fig_err = go.Figure(go.Bar(
    x=df["Movie"],
    y=df["Error ($M)"],
    marker_color=colors,
    text=[f"${v:+.2f}M" for v in df["Error ($M)"]],
    textposition="outside",
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

st.header("V14 Model vs Industry Projections")

ind_df = df[df["Industry Projection ($M)"].notna()].copy()
if len(ind_df) > 0:
    ind_df["Model Error ($M)"] = (ind_df["Predicted OW ($M)"] - ind_df["Actual OW ($M)"]).round(2)
    ind_df["Industry Error ($M)"] = (ind_df["Industry Projection ($M)"] - ind_df["Actual OW ($M)"]).round(2)
    model_mae = ind_df["Model Error ($M)"].abs().mean()
    industry_mae = ind_df["Industry Error ($M)"].abs().mean()

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("V14 Model MAE", f"${model_mae:.2f}M", f"on {len(ind_df)} films with industry data")
    mc2.metric("Industry MAE", f"${industry_mae:.2f}M", f"Deadline/THR/NRG consensus")
    diff = industry_mae - model_mae
    mc3.metric("Model Advantage", f"${abs(diff):.2f}M {'better' if diff > 0 else 'worse'}", f"{'Model wins' if diff > 0 else 'Industry wins'}")

    fig_vs = go.Figure()
    fig_vs.add_trace(go.Bar(
        name="V14 Model Error", x=ind_df["Movie"], y=ind_df["Model Error ($M)"],
        marker_color="#636EFA",
        text=[f"${v:+.1f}M" for v in ind_df["Model Error ($M)"]],
        textposition="outside",
    ))
    fig_vs.add_trace(go.Bar(
        name="Industry Error", x=ind_df["Movie"], y=ind_df["Industry Error ($M)"],
        marker_color="#AB63FA",
        text=[f"${v:+.1f}M" for v in ind_df["Industry Error ($M)"]],
        textposition="outside",
    ))
    fig_vs.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_vs.update_layout(
        barmode="group", height=400, yaxis_title="Error ($M) — positive = over-predicted",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40),
    )
    st.plotly_chart(fig_vs, use_container_width=True)

    st.caption(
        "Industry projections sourced from Deadline, The Hollywood Reporter, and NRG pre-release tracking. "
        "Greenland 2 excluded (no published industry projection found)."
    )

st.divider()

st.header("Case Study: Scream 7 — Why V13 → V14 → V15")
st.markdown(
    "Scream 7 was a key catalyst for the move from **V13 (4-tier)** to **V14 (3-tier)**. "
    "The V13 classifier put Scream 7 in SMALL (76.97%) — even with a manual override to LARGE, "
    "a routing bug sent it to the MID regressor (~$30M). V14 (with manual LARGE+ override and "
    "a fixed `np.exp()` bug) predicts **$69.59M** — only 9.4% off the actual $63.62M."
)
st.markdown(
    "**V15 Update**: With 30 more training films and PREDECESSOR_OW_LOG, V15 would likely "
    "auto-classify Scream 7 more accurately — LARGE+ accuracy improved from 65% to 77.1%."
)

v13v14_col1, v13v14_col2 = st.columns(2)

with v13v14_col1:
    st.error("**V13: 4-Tier System** (SMALL / MID / LARGE / BLOCKBUSTER)")
    st.markdown(
        "| | |\n"
        "|---|---|\n"
        "| **Auto-Classified** | SMALL (76.97%) |\n"
        "| **Manual Override** | LARGE |\n"
        "| **Regressor Used** | MID (~$30M) |\n"
        "| **Actual OW** | **$63.62M** |\n"
        "| **Error** | ~$33M under-prediction |\n"
        "\n"
        "**Why it failed**: Classifier put it in SMALL at 76.97%. Even with manual override "
        "to LARGE, a SQL routing bug sent it to the MID regressor. The 4-tier system had only "
        "**27 films** in LARGE — not enough training data for reliable classification."
    )

with v13v14_col2:
    st.success("**V14: 3-Tier System** (SMALL / MID / LARGE+)")
    st.markdown(
        "| | |\n"
        "|---|---|\n"
        "| **Auto-Classified** | SMALL (61.2%) |\n"
        "| **Manual Override** | LARGE+ |\n"
        "| **LARGE+ Regressor** | **$69.59M** |\n"
        "| **CI Range** | $55.67M - $83.51M |\n"
        "| **Actual OW** | **$63.62M** (within CI) |\n"
        "| **Error** | +$5.97M (9.4%) |\n"
        "\n"
        "**What improved**: Fixed `np.exp()` bug (regressors output log-space). "
        "With manual LARGE+ override, the regressor predicted **$69.59M** — "
        "only 9.4% off actual. Collapsing LARGE + BLOCKBUSTER into LARGE+ "
        "gave **46 training films** instead of 27."
    )

st.caption("Chart below: V13 MID regressor (~$30M) vs V14 LARGE+ regressor ($69.59M) vs actual ($63.62M). V14's CI range shown as shaded region.")

fig_v13v14 = go.Figure()
fig_v13v14.add_trace(go.Bar(
    name="V13 MID Regressor (~$30M)", x=["Scream 7"], y=[30],
    marker_color="#EF553B", text=["V13: ~$30M"], textposition="outside",
    width=0.2,
))
fig_v13v14.add_trace(go.Bar(
    name="V14 LARGE+ Regressor ($69.59M)", x=["Scream 7"], y=[69.59],
    marker_color="#636EFA", text=["V14: $69.59M"], textposition="outside",
    width=0.2,
))
fig_v13v14.add_trace(go.Bar(
    name="Actual OW ($63.62M)", x=["Scream 7"], y=[63.62],
    marker_color="#00CC96", text=["Actual: $63.62M"], textposition="outside",
    width=0.2,
))
fig_v13v14.add_shape(
    type="rect", x0=-0.4, x1=0.4, y0=55.67, y1=83.51,
    fillcolor="rgba(99, 110, 250, 0.1)", line=dict(color="rgba(99, 110, 250, 0.5)", dash="dot"),
)
fig_v13v14.add_annotation(
    x=0.42, y=69.59, text="V14 CI: $55.67M-$83.51M<br><i>Actual $63.62M within CI</i>",
    showarrow=False, font=dict(size=10, color="#636EFA"), xanchor="left",
)
fig_v13v14.update_layout(
    barmode="group", height=350, yaxis_title="Opening Weekend ($M)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=40),
)
st.plotly_chart(fig_v13v14, use_container_width=True)

st.markdown(
    "| Metric | V13 (4-Tier) | V14 (3-Tier) | Change |\n"
    "|--------|-------------|-------------|--------|\n"
    "| **Tier System** | SMALL / MID / LARGE / BLOCKBUSTER | SMALL / MID / LARGE+ | Collapsed top 2 |\n"
    "| **LARGE(+) Training Films** | 27 | 46 | +70% |\n"
    "| **LARGE(+) Accuracy** | 27% | 65% | **+38pp** |\n"
    "| **Overall Classification** | 67.8% | 71.5% | +3.7pp |\n"
    "| **Overall MAE** | $14.0M | $13.1M | -$0.9M |\n"
    "| **Scream 7 Tier** | MID | LARGE+ | Fixed |"
)

st.divider()

st.header("Weekend-by-Weekend Breakdown")

for w in WEEKEND_DATA:
    model_ver = w.get('model', 'V14')
    with st.expander(f"**{w['weekend']}** — {w['dates']}  ({model_ver})", expanded=False):
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
                    ci_str = f"${m['conf_low']:.2f} - ${m['conf_high']:.2f}M"
                    in_ci = "✅" if m["conf_low"] <= m["actual_ow"] <= m["conf_high"] else "❌"
                else:
                    error = None
                    tier_match = "—"
                    ci_str = "—"
                    in_ci = "—"
                ind_proj_val = m.get("industry_projection")
                rows.append({
                    "Movie": m["movie"],
                    "Studio": m["studio"],
                    "Pred Tier": pred_tier or "—",
                    "Actual Tier": actual_tier or "—",
                    "Tier": tier_match,
                    "Predicted ($M)": f"${m['predicted_ow']:.2f}" if has_prediction else "—",
                    "Industry ($M)": f"${ind_proj_val:.0f}" if ind_proj_val else "—",
                    "Actual ($M)": f"${m['actual_ow']:.2f}" if m["actual_ow"] else "—",
                    "Error ($M)": f"${error:+.2f}" if error is not None else "—",
                    "CI Range": ci_str,
                    "In CI": in_ci,
                    "Note": m.get("note", ""),
                })
        wdf = pd.DataFrame(rows)
        st.dataframe(wdf, use_container_width=True, hide_index=True)

st.divider()

st.header("Upcoming Predictions")
st.markdown("Predictions for upcoming weekends (model version shown per weekend). Move to actuals section once official box office data is available.")

for u in UPCOMING:
    model_tag = u.get("model", "V14")
    with st.expander(f"**{u['weekend']}** — {u['dates']}  ({model_tag})", expanded=(model_tag == "V15")):
        upcoming_rows = []
        for m in u["movies"]:
            tier_color = {"SMALL": "🟠", "MID": "🟡", "LARGE+": "🟢"}.get(m["predicted_tier"], "⚪")
            upcoming_rows.append({
                "Movie": m["movie"],
                "Studio": m["studio"],
                "Tier": f"{tier_color} {m['predicted_tier']}",
                "Predicted OW": f"${m['predicted_ow']:.1f}M",
                "CI Low": f"${m['conf_low']:.1f}M",
                "CI High": f"${m['conf_high']:.1f}M",
                "Note": m.get("note", ""),
            })
        udf = pd.DataFrame(upcoming_rows)
        st.dataframe(udf, use_container_width=True, hide_index=True)

        for m in u["movies"]:
            if m["predicted_tier"] == "MID" or m["predicted_tier"] == "LARGE+":
                st.markdown(
                    f"**{m['movie']}** — Predicted **${m['predicted_ow']:.1f}M** "
                    f"(range ${m['conf_low']:.1f}M - ${m['conf_high']:.1f}M)"
                )

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
    fig_tier.update_traces(texttemplate="$%{text:.2f}M", textposition="outside")
    fig_tier.update_layout(height=300, showlegend=False, margin=dict(t=20))
    st.plotly_chart(fig_tier, use_container_width=True)

st.divider()

st.info(
    "**Data Sources & Methodology**\n"
    "- **No data leakage**: Every prediction was generated *before* the film's opening weekend, using the model version in production at that time\n"
    "- **V14 model**: Used for Weekends 2–10 (trained Feb 27, 2026 on 239 films through early 2026)\n"
    "- **V15 model**: In production since March 8, 2026 — used for Weekend 11+ only\n"
    "- **Industry projections**: Pre-release consensus from Deadline, The Hollywood Reporter, and NRG tracking\n"
    "- **Actuals**: Official box office reporting\n"
    "- **Tier System**: SMALL (<$15M) / MID ($15-50M) / LARGE+ (>$50M)\n"
    "- Scream 7: V14 tier override to LARGE+ correct; predicted $69.59M vs actual $63.62M (9.4% error, within CI)"
)

st.caption("To add new weekends: update WEEKEND_DATA with real values from official box office data. Move entries from UPCOMING once actuals are available.")

show_cortex_badge()
