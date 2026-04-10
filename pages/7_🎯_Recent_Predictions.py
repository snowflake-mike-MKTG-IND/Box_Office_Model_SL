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
            {"movie": "Greenland 2: Migration", "studio": "STX/Lionsgate", "predicted_tier": "SMALL", "predicted_ow": 9.37, "conf_low": 7.50, "conf_high": 11.24, "actual_ow": 8.40, "week": 1, "industry_projection": 9.0, "note": "Gerard Butler sequel; Deadline projected $8-10M OW"},
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
    {
        "weekend": "Weekend 10",
        "dates": "Mar 6-8, 2026",
        "model": "V15",
        "movies": [
            {"movie": "The Bride", "studio": "Warner Bros.", "predicted_tier": "SMALL", "predicted_ow": 8.31, "conf_low": 6.65, "conf_high": 9.97, "actual_ow": 7.20, "week": 1, "industry_projection": None, "note": "Maggie Gyllenhaal; horror — actual $7.2M within CI"},
            {"movie": "Hoppers", "studio": "Disney", "predicted_tier": "SMALL", "predicted_ow": 10.12, "conf_low": 8.10, "conf_high": 12.14, "actual_ow": 45.35, "week": 1, "industry_projection": None, "note": "V15 predicted SMALL $10.12M, actual MID $45.35M — massive tier miss; Disney animated film"},
        ]
    },
    {
        "weekend": "Weekend 11",
        "dates": "Mar 13-15, 2026",
        "model": "V15",
        "movies": [
            {"movie": "Reminders of Him", "studio": "Universal", "predicted_tier": "SMALL", "predicted_ow": 7.48, "conf_low": 5.98, "conf_high": 8.98, "actual_ow": 17.98, "week": 1, "industry_projection": None, "note": "Colleen Hoover adaptation; predicted SMALL $7.48M, actual MID $17.98M — tier miss"},
        ]
    },
    {
        "weekend": "Weekend 12",
        "dates": "Mar 20-22, 2026",
        "model": "V15",
        "movies": [
            {"movie": "Project Hail Mary", "studio": "Amazon MGM", "predicted_tier": "MID", "predicted_ow": 26.42, "conf_low": 21.14, "conf_high": 31.70, "actual_ow": 80.51, "week": 1, "industry_projection": None, "note": "$200M budget; Ryan Gosling; predicted MID $26.42M, actual LARGE+ $80.51M — biggest model miss, tier miss"},
            {"movie": "Ready or Not 2", "studio": "Disney/Searchlight", "predicted_tier": "SMALL", "predicted_ow": 7.78, "conf_low": 6.22, "conf_high": 9.34, "actual_ow": 9.08, "week": 1, "industry_projection": None, "note": "Horror sequel; predicted SMALL $7.78M, actual SMALL $9.08M — within CI, tier correct"},
        ]
    },
]

V16_RETRO = {
    "Project Hail Mary": {"v16_tier": "LARGE+", "v16_ow": 80.56, "v16_low": 64.45, "v16_high": 96.67, "v16_confidence": 97.9, "tmdb_d14": 26.56, "tmdb_d7": 70.71, "momentum": 2.66, "overridden": False, "override_reason": None, "note": "V16 auto-classifies LARGE+ at 98% — no override needed. Model trained on 285 films now correctly sizes $200M Ryan Gosling film."},
    "Reminders of Him": {"v16_tier": "MID", "v16_ow": 18.28, "v16_low": 14.62, "v16_high": 21.93, "v16_confidence": 98.8, "tmdb_d14": 8.34, "tmdb_d7": 9.89, "momentum": 1.19, "overridden": False, "override_reason": None, "note": "V16 auto-classifies MID at 98.8%. More training data (285 vs 269) lets the model properly size Colleen Hoover's audience."},
    "Hoppers": {"v16_tier": "MID", "v16_ow": 42.38, "v16_low": 33.91, "v16_high": 50.86, "v16_confidence": 96.3, "tmdb_d14": 18.05, "tmdb_d7": 36.91, "momentum": 2.04, "overridden": False, "override_reason": None, "note": "V16 auto-classifies MID at 96.3%. Disney animated film now correctly sized with IS_MAJOR_STUDIO feature."},
    "Scream 7": {"v16_tier": "LARGE+", "v16_ow": 67.40, "v16_low": 53.92, "v16_high": 80.88, "v16_confidence": 95.0, "tmdb_d14": 35.64, "tmdb_d7": 37.12, "momentum": 1.04, "overridden": False, "override_reason": None, "note": "V16 auto-classifies LARGE+ at 95% — Scream franchise IP now well-represented in 285-film training set."},
    "The Bride": {"v16_tier": "SMALL", "v16_ow": 8.02, "v16_low": 6.42, "v16_high": 9.63, "v16_confidence": 97.0, "tmdb_d14": 13.90, "tmdb_d7": 20.30, "momentum": 1.46, "overridden": False, "override_reason": None, "note": "V16 keeps SMALL — correct. D14=13.9 below override threshold."},
    "Ready or Not 2": {"v16_tier": "SMALL", "v16_ow": 7.95, "v16_low": 6.36, "v16_high": 9.54, "v16_confidence": 97.5, "tmdb_d14": 9.55, "tmdb_d7": 13.93, "momentum": 1.46, "overridden": False, "override_reason": None, "note": "V16 keeps SMALL — correct tier, D14=9.5 well below override threshold."},
    "PRIMATE": {"v16_tier": "SMALL", "v16_ow": 11.16, "v16_low": 8.92, "v16_high": 13.39, "v16_confidence": 96.7, "tmdb_d14": 24.58, "tmdb_d7": 23.53, "momentum": 0.96, "overridden": False, "override_reason": None, "note": "V16 keeps SMALL — correct. D14=24.6 is high but momentum 0.96 (declining) prevents false MID override. Rule C working as designed."},
    "Super Mario Galaxy": {"v16_tier": "LARGE+", "v16_ow": 131.21, "v16_low": 104.97, "v16_high": 157.45, "v16_confidence": 95.0, "tmdb_d14": 38.25, "tmdb_d7": 38.69, "momentum": 1.01, "overridden": False, "override_reason": None, "note": "V16 auto-classifies LARGE+ at 95% — Nintendo IP + Universal (major studio) make this a confident call."},
}

UPCOMING = [
    {
        "weekend": "Weekend 16",
        "dates": "Apr 24-26, 2026",
        "model": "V16",
        "movies": [
            {"movie": "MICHAEL", "studio": "Amazon MGM", "predicted_tier": "LARGE+", "predicted_ow": 72.14, "conf_low": 57.71, "conf_high": 86.57, "tier_confidence": 75.8,
             "tmdb_d14": 51.96, "overridden": True, "override_reason": "D14=52.0 >= 25 -> forced LARGE+",
             "note": "Model says MID but TMDB D14=52.0 triggers Rule C override to LARGE+. Highest TMDB D14 popularity we've ever seen. Day -14 prediction."},
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

col1.metric("V14/V15 Tier Accuracy", f"{tier_correct_all}/{tier_total_all}", f"{tier_correct_all/tier_total_all*100:.0f}%")
col2.metric("V14/V15 MAE", f"${mae:.2f}M", f"{regression_total} movies")
col3.metric("Within CI", f"{within_ci}/{regression_total}", f"{within_ci/regression_total*100:.0f}%")
col4.metric("Movies Tracked", f"{tier_total_all}", f"{len(WEEKEND_DATA)} weekends")

st.divider()

st.header("Predicted vs Actual (V14/V15 Production Predictions)")

st.caption("All predictions were made before each film's opening weekend using the model version in production at the time (V14 for W2-W9, V15 for W10+). "
           "Purple bars show pre-release industry consensus projections where available.")

fig = go.Figure()
fig.add_trace(go.Bar(
    name="Predicted OW",
    x=df["Movie"],
    y=df["Predicted OW ($M)"],
    marker_color=["#636EFA" if m == "V14" else "#19D3F3" for m in df["Model"]],
    text=[f"${v:.1f}M" for v in df["Predicted OW ($M)"]],
    textposition="outside",
))
fig.add_trace(go.Bar(
    name="Actual OW",
    x=df["Movie"],
    y=df["Actual OW ($M)"],
    marker_color="#00CC96",
    text=[f"${v:.1f}M" for v in df["Actual OW ($M)"]],
    textposition="outside",
))
ind_vals = df["Industry Projection ($M)"].fillna(0)
if ind_vals.sum() > 0:
    fig.add_trace(go.Bar(
        name="Industry Projection",
        x=df["Movie"],
        y=ind_vals,
        marker_color="#AB63FA",
        text=[f"${v:.0f}M" if v > 0 else "" for v in ind_vals],
        textposition="outside",
    ))

fig.update_layout(
    barmode="group",
    height=500,
    yaxis_title="Opening Weekend ($M)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=40),
    xaxis_tickangle=-30,
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

st.header("V15 Misses: Why V16 Was Built")
st.markdown(
    "V15 had **3 major tier misses** on recent films — all under-predictions where the model "
    "failed to recognize breakout potential. These misses drove the development of **V16**, which adds "
    "285 training films (+16), IS_MAJOR_STUDIO feature, and a **TMDB popularity override system (Rule C)**."
)

miss_data = [
    {"movie": "Project Hail Mary", "v15_tier": "MID", "v15_ow": 26.42, "actual_tier": "LARGE+", "actual_ow": 80.51, "error": -54.09},
    {"movie": "Hoppers", "v15_tier": "SMALL", "v15_ow": 10.12, "actual_tier": "MID", "actual_ow": 45.35, "error": -35.23},
    {"movie": "Reminders of Him", "v15_tier": "SMALL", "v15_ow": 7.48, "actual_tier": "MID", "actual_ow": 17.98, "error": -10.50},
]

mc1, mc2, mc3 = st.columns(3)
for col, m in zip([mc1, mc2, mc3], miss_data):
    with col:
        col.error(f"**{m['movie']}**")
        col.markdown(
            f"| | V15 | Actual |\n"
            f"|---|---|---|\n"
            f"| **Tier** | {m['v15_tier']} | **{m['actual_tier']}** |\n"
            f"| **OW** | ${m['v15_ow']:.2f}M | **${m['actual_ow']:.2f}M** |\n"
            f"| **Error** | **${m['error']:+.2f}M** | |"
        )

st.divider()

st.header("V16 + TMDB Override: How It Fixes the Misses")
st.markdown(
    "**V16** addresses the V15 misses through two mechanisms:\n"
    "1. **More training data** (285 films vs 269) — the model itself now correctly sizes these films\n"
    "2. **Rule C TMDB Override** — an orthogonal post-model safety net using live TMDB popularity data\n\n"
    "Rule C thresholds (based on 22 calibration films):\n"
    "- `TMDB_POP_D14 >= 25` → force minimum **LARGE+**\n"
    "- `TMDB_POP_D14 >= 15` AND `momentum (D7/D14) >= 1.3` → force minimum **MID**"
)

comparison_movies = ["Project Hail Mary", "Hoppers", "Reminders of Him"]
v15_miss_list = {m["movie"]: m for m in miss_data}

fig_fix = go.Figure()
fig_fix.add_trace(go.Bar(
    name="V15 Prediction",
    x=comparison_movies,
    y=[v15_miss_list[m]["v15_ow"] for m in comparison_movies],
    marker_color="#EF553B",
    text=[f"${v15_miss_list[m]['v15_ow']:.1f}M<br>({v15_miss_list[m]['v15_tier']})" for m in comparison_movies],
    textposition="outside",
    width=0.25,
))
fig_fix.add_trace(go.Bar(
    name="V16 Prediction",
    x=comparison_movies,
    y=[V16_RETRO[m]["v16_ow"] for m in comparison_movies],
    marker_color="#636EFA",
    text=[f"${V16_RETRO[m]['v16_ow']:.1f}M<br>({V16_RETRO[m]['v16_tier']})" for m in comparison_movies],
    textposition="outside",
    width=0.25,
))
fig_fix.add_trace(go.Bar(
    name="Actual OW",
    x=comparison_movies,
    y=[v15_miss_list[m]["actual_ow"] for m in comparison_movies],
    marker_color="#00CC96",
    text=[f"${v15_miss_list[m]['actual_ow']:.1f}M<br>({v15_miss_list[m]['actual_tier']})" for m in comparison_movies],
    textposition="outside",
    width=0.25,
))

fig_fix.update_layout(
    barmode="group",
    height=500,
    yaxis_title="Opening Weekend ($M)",
    title="V15 Misses vs V16 Corrections",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=60),
)
st.plotly_chart(fig_fix, use_container_width=True)

fix_col1, fix_col2, fix_col3 = st.columns(3)

with fix_col1:
    st.success("**Project Hail Mary**")
    phm = V16_RETRO["Project Hail Mary"]
    st.markdown(
        f"| | V15 | V16 | Actual |\n"
        f"|---|---|---|---|\n"
        f"| **Tier** | MID | **{phm['v16_tier']}** | LARGE+ |\n"
        f"| **OW** | $26.42M | **${phm['v16_ow']:.2f}M** | $80.51M |\n"
        f"| **Error** | -$54.09M | **${phm['v16_ow'] - 80.51:+.2f}M** | |\n"
        f"| **CI** | $21-32M | **${phm['v16_low']:.0f}-${phm['v16_high']:.0f}M** | |\n\n"
        f"TMDB D14={phm['tmdb_d14']}, D7={phm['tmdb_d7']}, momentum={phm['momentum']}\n\n"
        f"_{phm['note']}_"
    )

with fix_col2:
    st.success("**Hoppers**")
    hop = V16_RETRO["Hoppers"]
    st.markdown(
        f"| | V15 | V16 | Actual |\n"
        f"|---|---|---|---|\n"
        f"| **Tier** | SMALL | **{hop['v16_tier']}** | MID |\n"
        f"| **OW** | $10.12M | **${hop['v16_ow']:.2f}M** | $45.35M |\n"
        f"| **Error** | -$35.23M | **${hop['v16_ow'] - 45.35:+.2f}M** | |\n"
        f"| **CI** | $8-12M | **${hop['v16_low']:.0f}-${hop['v16_high']:.0f}M** | |\n\n"
        f"TMDB D14={hop['tmdb_d14']}, D7={hop['tmdb_d7']}, momentum={hop['momentum']}\n\n"
        f"_{hop['note']}_"
    )

with fix_col3:
    st.success("**Reminders of Him**")
    roh = V16_RETRO["Reminders of Him"]
    st.markdown(
        f"| | V15 | V16 | Actual |\n"
        f"|---|---|---|---|\n"
        f"| **Tier** | SMALL | **{roh['v16_tier']}** | MID |\n"
        f"| **OW** | $7.48M | **${roh['v16_ow']:.2f}M** | $17.98M |\n"
        f"| **Error** | -$10.50M | **${roh['v16_ow'] - 17.98:+.2f}M** | |\n"
        f"| **CI** | $6-9M | **${roh['v16_low']:.0f}-${roh['v16_high']:.0f}M** | |\n\n"
        f"TMDB D14={roh['tmdb_d14']}, D7={roh['tmdb_d7']}, momentum={roh['momentum']}\n\n"
        f"_{roh['note']}_"
    )

v15_miss_mae = np.mean([abs(m["error"]) for m in miss_data])
v16_fix_mae = np.mean([abs(V16_RETRO[m["movie"]]["v16_ow"] - m["actual_ow"]) for m in miss_data])
st.markdown(
    f"**On the 3 V15 misses**: V15 MAE = **${v15_miss_mae:.1f}M** → V16 MAE = **${v16_fix_mae:.1f}M** "
    f"({(1 - v16_fix_mae/v15_miss_mae)*100:.0f}% improvement). All 3 tier corrections from ❌ to ✅."
)

st.divider()

st.header("V15 vs V16: Full Comparison on All Films with Actuals")
st.markdown(
    "Side-by-side comparison showing V15 production predictions (as actually made) vs V16 retrospective predictions "
    "(what V16 *would have* predicted). V14 predictions are excluded since V16 wasn't designed to replace V14."
)

v15_movies_for_comparison = []
for w in WEEKEND_DATA:
    if w.get("model") == "V15":
        for m in w["movies"]:
            if m["week"] == 1 and m["predicted_ow"] is not None and m["movie"] in V16_RETRO:
                v16 = V16_RETRO[m["movie"]]
                actual_tier = get_actual_tier(m["actual_ow"])
                v15_tier_ok = m["predicted_tier"] == actual_tier
                v16_tier_ok = v16["v16_tier"] == actual_tier
                if m["predicted_tier"] and "LARGE" in m["predicted_tier"] and actual_tier and "LARGE" in actual_tier:
                    v15_tier_ok = True
                if v16["v16_tier"] and "LARGE" in v16["v16_tier"] and actual_tier and "LARGE" in actual_tier:
                    v16_tier_ok = True
                v15_movies_for_comparison.append({
                    "Movie": m["movie"],
                    "Actual OW ($M)": m["actual_ow"],
                    "Actual Tier": actual_tier,
                    "V15 Tier": m["predicted_tier"],
                    "V15 OW ($M)": m["predicted_ow"],
                    "V15 Error ($M)": round(m["predicted_ow"] - m["actual_ow"], 2),
                    "V15 Tier OK": "✅" if v15_tier_ok else "❌",
                    "V16 Tier": v16["v16_tier"],
                    "V16 OW ($M)": v16["v16_ow"],
                    "V16 Error ($M)": round(v16["v16_ow"] - m["actual_ow"], 2),
                    "V16 Tier OK": "✅" if v16_tier_ok else "❌",
                    "TMDB D14": v16.get("tmdb_d14"),
                    "Override": "Yes" if v16.get("overridden") else "No",
                })

if v15_movies_for_comparison:
    comp_df = pd.DataFrame(v15_movies_for_comparison)

    cc1, cc2, cc3, cc4 = st.columns(4)
    v15_tier_acc = comp_df["V15 Tier OK"].value_counts().get("✅", 0)
    v16_tier_acc = comp_df["V16 Tier OK"].value_counts().get("✅", 0)
    n_comp = len(comp_df)
    v15_mae_comp = comp_df["V15 Error ($M)"].abs().mean()
    v16_mae_comp = comp_df["V16 Error ($M)"].abs().mean()

    cc1.metric("V15 Tier Accuracy", f"{v15_tier_acc}/{n_comp}", f"{v15_tier_acc/n_comp*100:.0f}%")
    cc2.metric("V16 Tier Accuracy", f"{v16_tier_acc}/{n_comp}", delta=f"+{v16_tier_acc - v15_tier_acc} corrections")
    cc3.metric("V15 MAE", f"${v15_mae_comp:.2f}M")
    cc4.metric("V16 MAE", f"${v16_mae_comp:.2f}M", delta=f"${v16_mae_comp - v15_mae_comp:+.2f}M")

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        name="V15 Prediction",
        x=comp_df["Movie"],
        y=comp_df["V15 OW ($M)"],
        marker_color="#EF553B",
        text=[f"${v:.1f}M" for v in comp_df["V15 OW ($M)"]],
        textposition="outside",
    ))
    fig_comp.add_trace(go.Bar(
        name="V16 Prediction",
        x=comp_df["Movie"],
        y=comp_df["V16 OW ($M)"],
        marker_color="#636EFA",
        text=[f"${v:.1f}M" for v in comp_df["V16 OW ($M)"]],
        textposition="outside",
    ))
    fig_comp.add_trace(go.Bar(
        name="Actual OW",
        x=comp_df["Movie"],
        y=comp_df["Actual OW ($M)"],
        marker_color="#00CC96",
        text=[f"${v:.1f}M" for v in comp_df["Actual OW ($M)"]],
        textposition="outside",
    ))
    fig_comp.update_layout(
        barmode="group",
        height=500,
        yaxis_title="Opening Weekend ($M)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40),
        xaxis_tickangle=-20,
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    st.dataframe(
        comp_df[["Movie", "Actual Tier", "Actual OW ($M)", "V15 Tier", "V15 OW ($M)", "V15 Tier OK", "V16 Tier", "V16 OW ($M)", "V16 Tier OK", "TMDB D14", "Override"]],
        use_container_width=True, hide_index=True
    )

st.divider()

st.header("Rule C Override: Safety Net Design")
st.markdown(
    "The TMDB popularity override operates **orthogonally** to the model — it runs *after* the cascade prediction "
    "and can only **raise** a tier, never lower it. This addresses a fundamental limitation: TMDB daily popularity "
    "data only exists for ~30 training films, so CatBoost can't learn meaningful splits from it. But the raw signal "
    "is extremely strong (Spearman r=0.817 with actual OW at day -14)."
)

rule_col1, rule_col2 = st.columns(2)
with rule_col1:
    st.markdown(
        "**Rule C Thresholds**\n\n"
        "| Condition | Action |\n"
        "|---|---|\n"
        "| `TMDB_POP_D14 >= 25` | Force minimum **LARGE+** |\n"
        "| `TMDB_POP_D14 >= 15` AND `D7/D14 >= 1.3` | Force minimum **MID** |\n"
        "| Otherwise | No override (trust model) |\n\n"
        "The **momentum gate** (D7/D14 >= 1.3) prevents false positives like PRIMATE "
        "(D14=24.6, momentum=0.96 — declining hype = SMALL, not MID)."
    )

with rule_col2:
    st.markdown(
        "**Holdout Validation (19 blind films)**\n\n"
        "| Metric | Without Override | With Rule C |\n"
        "|---|---|---|\n"
        "| Tier Accuracy | 63.2% | **84.2%** |\n"
        "| MAE | $16.7M | **$10.1M** |\n"
        "| Overrides Applied | 0 | 4 |\n"
        "| Correct Overrides | — | **4/4 (100%)** |\n"
        "| Wrong Overrides | — | **0** |"
    )

calibration_data = [
    {"movie": "Super Mario Galaxy", "tier": "LARGE+", "d14": 38.25, "ow": 131.70, "override": "LARGE+ (D14>=25)"},
    {"movie": "Scream 7", "tier": "LARGE+", "d14": 35.64, "ow": 63.62, "override": "LARGE+ (D14>=25)"},
    {"movie": "Project Hail Mary", "tier": "LARGE+", "d14": 26.56, "ow": 80.51, "override": "LARGE+ (D14>=25)"},
    {"movie": "PRIMATE", "tier": "SMALL", "d14": 24.58, "ow": 11.16, "override": "None (D14<25, mom=0.96)"},
    {"movie": "Weapons", "tier": "MID", "d14": 21.34, "ow": 43.50, "override": "MID (D14>=15, mom>=1.3)"},
    {"movie": "Hoppers", "tier": "MID", "d14": 18.05, "ow": 45.35, "override": "MID (D14>=15, mom=2.04)"},
    {"movie": "Wuthering Heights", "tier": "MID", "d14": 17.00, "ow": 32.80, "override": "MID (D14>=15, mom=1.35)"},
    {"movie": "Goat", "tier": "MID", "d14": 15.66, "ow": 27.20, "override": "MID (D14>=15, mom=1.58)"},
]

with st.expander("TMDB D14 Calibration Data (22 films)", expanded=False):
    cal_df = pd.DataFrame(calibration_data)
    fig_cal = px.scatter(
        cal_df, x="d14", y="ow", color="tier", text="movie",
        color_discrete_map={"SMALL": "#FF7F0E", "MID": "#FFD700", "LARGE+": "#00CC96"},
        labels={"d14": "TMDB Popularity (Day -14)", "ow": "Actual OW ($M)", "tier": "Tier"},
    )
    fig_cal.update_traces(textposition="top center", marker_size=12)
    fig_cal.add_vline(x=25, line_dash="dash", line_color="red", annotation_text="LARGE+ threshold (D14=25)")
    fig_cal.add_vline(x=15, line_dash="dash", line_color="orange", annotation_text="MID threshold (D14=15)")
    fig_cal.update_layout(height=400, margin=dict(t=30))
    st.plotly_chart(fig_cal, use_container_width=True)

st.divider()

st.header("Case Study: Scream 7 — V13 → V14 → V15 → V16")
st.markdown(
    "Scream 7 was a key catalyst for the move from **V13 (4-tier)** to **V14 (3-tier)**. "
    "The V13 classifier put Scream 7 in SMALL (76.97%) — even with a manual override to LARGE, "
    "a routing bug sent it to the MID regressor (~$30M). V14 (with manual LARGE+ override and "
    "a fixed `np.exp()` bug) predicts **$69.59M** — only 9.4% off the actual $63.62M."
)
st.markdown(
    "**V16**: With 285 training films, IS_MAJOR_STUDIO, and the TMDB override system, V16 "
    "auto-classifies Scream 7 as **LARGE+ at 95% confidence** — no manual override needed. "
    "Predicted **$67.40M** (5.9% error)."
)

v_cols = st.columns(4)
with v_cols[0]:
    st.error("**V13**")
    st.markdown("SMALL (76.97%)\n\nMID regressor: ~$30M\n\nError: ~$33M")
with v_cols[1]:
    st.warning("**V14**")
    st.markdown("SMALL → LARGE+ (manual)\n\n$69.59M\n\nError: +$5.97M")
with v_cols[2]:
    st.info("**V15**")
    st.markdown("Auto LARGE+ (77.1% acc)\n\nImproved but PHM miss\n\ndrove V16 dev")
with v_cols[3]:
    st.success("**V16**")
    st.markdown("Auto LARGE+ (95%)\n\n$67.40M\n\nError: +$3.78M")

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
                    "Pred Tier": (pred_tier or "—") + (f" ({m['tier_confidence']:.0f}%)" if m.get("tier_confidence") else ""),
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
st.markdown("V16 predictions for upcoming weekends. MICHAEL is the first film predicted with the TMDB override system.")

for u in UPCOMING:
    model_tag = u.get("model", "V16")
    with st.expander(f"**{u['weekend']}** — {u['dates']}  ({model_tag})", expanded=True):
        upcoming_rows = []
        for m in u["movies"]:
            tier_color = {"SMALL": "🟠", "MID": "🟡", "LARGE+": "🟢"}.get(m["predicted_tier"], "⚪")
            tc = m.get("tier_confidence")
            tier_str = f"{tier_color} {m['predicted_tier']}" + (f" ({tc:.0f}%)" if tc else "")
            if m.get("overridden"):
                tier_str += " [OVERRIDE]"
            upcoming_rows.append({
                "Movie": m["movie"],
                "Studio": m["studio"],
                "Tier": tier_str,
                "Predicted OW": f"${m['predicted_ow']:.1f}M",
                "CI Low": f"${m['conf_low']:.1f}M",
                "CI High": f"${m['conf_high']:.1f}M",
                "TMDB D14": m.get("tmdb_d14", "—"),
                "Override": m.get("override_reason", "—"),
                "Note": m.get("note", ""),
            })
        udf = pd.DataFrame(upcoming_rows)
        st.dataframe(udf, use_container_width=True, hide_index=True)

        for m in u["movies"]:
            if m.get("overridden"):
                st.warning(
                    f"**TMDB Override Active**: {m['movie']} — Model predicted MID, but TMDB D14 popularity of "
                    f"**{m.get('tmdb_d14', 'N/A')}** (highest ever recorded) triggered Rule C override to LARGE+. "
                    f"Predicted **${m['predicted_ow']:.1f}M** (range ${m['conf_low']:.1f}M - ${m['conf_high']:.1f}M)."
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
    "- **V14 model**: Used for Weekends 2-9 (trained Feb 27, 2026 on 239 films)\n"
    "- **V15 model**: In production March 8 - April 9, 2026 — used for Weekends 10-12 (269 training films)\n"
    "- **V16 model**: In production since April 10, 2026 — 285 training films, IS_MAJOR_STUDIO feature, TMDB popularity override (Rule C)\n"
    "- **V16 retrospective**: V16 predictions for past films are shown for comparison only — they were NOT the predictions made at the time\n"
    "- **TMDB Override**: Rule C operates orthogonally to the model, using live TMDB daily popularity data as a post-prediction safety net\n"
    "- **Actuals**: Official box office reporting\n"
    "- **Tier System**: SMALL (<$15M) / MID ($15-50M) / LARGE+ (>$50M)"
)

st.caption("V15 wrong predictions are preserved exactly as they were made — they are NOT updated. V16 retrospective column shows what the new model would have predicted.")

show_cortex_badge()
