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
    {
        "weekend": "Weekend 15",
        "dates": "Apr 17-19, 2026",
        "model": "V16",
        "movies": [
            {"movie": "Lee Cronin's The Mummy", "studio": "Universal", "predicted_tier": "SMALL", "predicted_ow": 9.40, "conf_low": 7.52, "conf_high": 11.28, "actual_ow": 13.515, "week": 1, "industry_projection": None, "note": "V16 first live prediction. R-rated horror reboot, $125M budget. Predicted SMALL $9.40M, actual SMALL $13.52M — tier correct, within range but outside CI"},
        ]
    },
]

CONFIDENCE_THRESHOLDS = {
    "HIGH": {"min_prob": 0.70, "min_trends_days": 7},
    "MEDIUM": {"min_prob": 0.60, "min_trends_days": 5},
    "LOW": {"min_prob": 0.50, "min_trends_days": 0},
}

CONFIDENCE_STYLES = {
    "HIGH": {"color": "#636EFA", "opacity": 1.0, "pattern": "", "badge": "", "border": ""},
    "MEDIUM": {"color": "#636EFA", "opacity": 0.75, "pattern": "", "badge": "", "border": ""},
    "LOW": {"color": "#636EFA", "opacity": 0.45, "pattern": "/", "badge": "⚠️ LOW CONFIDENCE", "border": ""},
    "PRELIMINARY": {"color": "#636EFA", "opacity": 0.30, "pattern": "x", "badge": "🔬 PRELIMINARY", "border": "dash"},
}


def get_confidence_level(tier_prob, trends_days, days_until_release, overridden=False):
    if overridden:
        return "HIGH"
    if tier_prob >= 0.70 and trends_days >= 7:
        return "HIGH"
    if tier_prob >= 0.60 or trends_days >= 7:
        return "MEDIUM"
    if tier_prob < 0.55 and (trends_days < 5 or days_until_release > 14):
        return "PRELIMINARY"
    return "LOW"


UPCOMING = [
    {
        "weekend": "Weekend 16",
        "dates": "Apr 24-26, 2026",
        "model": "V16",
        "movies": [
            {"movie": "MICHAEL", "studio": "Lionsgate", "predicted_tier": "LARGE+", "predicted_ow": 84.85, "conf_low": 67.88, "conf_high": 101.82,
             "tmdb_d14": 51.96, "overridden": False, "horizon": "-3d",
             "tier_prob": 0.61, "trends_days": 18, "days_until_release": 4,
             "note": "Michael Jackson biopic. $155M budget, PG-13. LARGE+ at -3d (61% S2 confidence). TMDB D14=52.0, D7=46.3, momentum=0.89. R3D=60.2, R7D=52.6 (trends accelerating)."},
        ]
    },
    {
        "weekend": "Weekend 17",
        "dates": "May 1-3, 2026",
        "model": "V16",
        "movies": [
            {"movie": "The Devil Wears Prada 2", "studio": "Disney", "predicted_tier": "LARGE+", "predicted_ow": 74.40, "conf_low": 59.52, "conf_high": 89.28,
             "tmdb_d14": 25.13, "overridden": True, "override_reason": "Rule C auto-override: TMDB D14=25.13 (>=25 threshold) → LARGE+. Model base=MID.",
             "horizon": "-14d", "tier_prob": 0.95, "trends_days": 11, "days_until_release": 11,
             "note": "Rule C auto-override to LARGE+. Model base=MID. Day -14 prediction. TMDB D14=25.13 crosses Rule C threshold. R7D=61.3. D7/D3 pending."},
        ]
    },
    {
        "weekend": "Weekend 18",
        "dates": "May 8-10, 2026",
        "model": "V16",
        "movies": [
            {"movie": "Mortal Kombat II", "studio": "Warner Bros.", "predicted_tier": "SMALL", "predicted_ow": 8.67, "conf_low": 6.94, "conf_high": 10.40,
             "tmdb_d14": 18.61, "overridden": False, "horizon": "-14d",
             "tier_prob": 0.536, "trends_days": 3, "days_until_release": 18,
             "note": "Video game sequel. $68M budget, R-rated. Day -14 prediction (early — only 3 days trends). TMDB D14=18.6 (Apr 20 proxy). 32K YT comments but low star power (Karl Urban MAX=1.41). Predecessor OW=$23.2M (MK 2021). Will refine at -7d."},
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

col1.metric("V14–V16 Tier Accuracy", f"{tier_correct_all}/{tier_total_all}", f"{tier_correct_all/tier_total_all*100:.0f}%")
col2.metric("V14–V16 MAE", f"${mae:.2f}M", f"{regression_total} movies")
col3.metric("Within CI", f"{within_ci}/{regression_total}", f"{within_ci/regression_total*100:.0f}%")
col4.metric("Movies Tracked", f"{tier_total_all}", f"{len(WEEKEND_DATA)} weekends")

st.divider()

st.header("Predicted vs Actual (V14/V15/V16 Production Predictions)")

st.caption("All predictions were made before each film's opening weekend using the model version in production at the time (V14 for W2-W9, V15 for W10-W12, V16 for W15+). "
           "Purple bars show pre-release industry consensus projections where available.")

fig = go.Figure()
fig.add_trace(go.Bar(
    name="Predicted OW",
    x=df["Movie"],
    y=df["Predicted OW ($M)"],
    marker_color=["#636EFA" if m == "V14" else "#19D3F3" if m == "V15" else "#EF553B" for m in df["Model"]],
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

st.header("V16 Preview: TMDB Popularity Override System")

st.warning(
    "**Validation Status: Awaiting Live Results** — V16 was deployed April 10, 2026. "
    "Three upcoming predictions below with **confidence gates**: HIGH/MEDIUM predictions are actionable, "
    "LOW/PRELIMINARY predictions are directional only and will be revised as more data becomes available."
)

st.subheader("Holdout Validation (19 Blind Films)")
st.markdown(
    "V16 adds a **TMDB popularity override (Rule C)** that runs after the cascade prediction "
    "and can only raise a tier, never lower it. Tested against 19 films held out from training:"
)

hc1, hc2, hc3, hc4 = st.columns(4)
hc1.metric("Without Override", "63.2%", "Tier accuracy (base model)")
hc2.metric("With Rule C", "84.2%", "+21pp improvement")
hc3.metric("Overrides Applied", "4", "Out of 19 films")
hc4.metric("Override Precision", "4/4", "100% correct, 0 wrong")

st.markdown(
    "**Rule C Thresholds** (calibrated on 22 films with TMDB daily data):\n"
    "- `TMDB_POP_D14 >= 25` → force minimum **LARGE+**\n"
    "- `TMDB_POP_D14 >= 15` AND `momentum (D7/D14) >= 1.3` → force minimum **MID**\n"
    "- Otherwise → trust the model (no override)"
)

st.divider()

st.subheader("Upcoming V16 Predictions")
st.markdown("Three upcoming films predicted with V16 in production — including Rule C auto-override and **confidence gates**. Updated April 20, 2026 with fresh Google Trends + TMDB data.")

st.markdown(
    '<div style="display:flex;gap:24px;margin-bottom:16px;flex-wrap:wrap;">'
    '<div><span style="display:inline-block;width:14px;height:14px;background:#636EFA;border-radius:2px;"></span> <b>HIGH</b> — ≥70% tier probability, ≥7 days trends</div>'
    '<div><span style="display:inline-block;width:14px;height:14px;background:rgba(99,110,250,0.75);border-radius:2px;"></span> <b>MEDIUM</b> — ≥60% probability or ≥7 days trends</div>'
    '<div><span style="display:inline-block;width:14px;height:14px;background:rgba(99,110,250,0.45);border-radius:2px;"></span> ⚠️ <b>LOW</b> — borderline probability, limited data</div>'
    '<div><span style="display:inline-block;width:14px;height:14px;background:rgba(99,110,250,0.30);border:2px dashed #636EFA;border-radius:2px;"></span> 🔬 <b>PRELIMINARY</b> — very early, sparse data, expect significant revision</div>'
    '</div>',
    unsafe_allow_html=True,
)

for u in UPCOMING:
    model_tag = u.get("model", "V16")
    with st.expander(f"**{u['weekend']}** — {u['dates']}  ({model_tag})", expanded=True):
        upcoming_rows = []
        for m in u["movies"]:
            tier_color = {"SMALL": "🟠", "MID": "🟡", "LARGE+": "🟢"}.get(m["predicted_tier"], "⚪")
            conf_level = get_confidence_level(
                m.get("tier_prob", 0.5),
                m.get("trends_days", 0),
                m.get("days_until_release", 99),
                m.get("overridden", False),
            )
            style = CONFIDENCE_STYLES[conf_level]
            badge = style["badge"]
            prob_pct = m.get("tier_prob", 0)
            tier_str = f"{tier_color} {m['predicted_tier']} ({prob_pct:.0%})"
            if m.get("overridden"):
                tier_str += " [OVERRIDE]"
            if badge:
                tier_str += f" {badge}"
            upcoming_rows.append({
                "Movie": m["movie"],
                "Studio": m["studio"],
                "Horizon": m.get("horizon", "—"),
                "Tier": tier_str,
                "Confidence": conf_level,
                "Predicted OW": f"${m['predicted_ow']:.1f}M",
                "CI Range": f"${m['conf_low']:.1f} - ${m['conf_high']:.1f}M",
                "TMDB D14": m.get("tmdb_d14", "—"),
                "Trends Days": m.get("trends_days", "—"),
            })
        udf = pd.DataFrame(upcoming_rows)
        st.dataframe(udf, use_container_width=True, hide_index=True)

        for m in u["movies"]:
            conf_level = get_confidence_level(
                m.get("tier_prob", 0.5),
                m.get("trends_days", 0),
                m.get("days_until_release", 99),
                m.get("overridden", False),
            )
            if conf_level == "PRELIMINARY":
                st.warning(
                    f"🔬 **PRELIMINARY**: {m['movie']} — only {m.get('trends_days', 0)} days of trends data, "
                    f"tier probability {m.get('tier_prob', 0):.0%}. This prediction will change significantly as more data becomes available."
                )
            elif conf_level == "LOW":
                st.warning(
                    f"⚠️ **LOW CONFIDENCE**: {m['movie']} — borderline tier classification ({m.get('tier_prob', 0):.0%}). "
                    f"Treat as directional estimate."
                )
            if m.get("overridden"):
                st.info(
                    f"**Rule C Auto-Override Active**: {m['movie']} — {m.get('override_reason', '')}. "
                    f"Predicted **${m['predicted_ow']:.1f}M** (range ${m['conf_low']:.1f}M - ${m['conf_high']:.1f}M)."
                )

st.divider()

st.info(
    "**Data Sources & Methodology**\n"
    "- **No data leakage**: Every prediction in the chart above was generated *before* the film's opening weekend, using the model version in production at that time\n"
    "- **V14 model**: Used for Weekends 2-9 (trained Feb 27, 2026 on 239 films)\n"
    "- **V15 model**: In production March 8 - April 9, 2026 — used for Weekends 10-12 (269 training films)\n"
    "- **V16 model**: In production since April 10, 2026 — 285 training films, 56 features, TMDB popularity override (Rule C). Predictions refreshed April 20, 2026\n"
    "- **Confidence gates**: HIGH (≥70% tier prob + ≥7 days trends), MEDIUM (≥60% or ≥7 days), LOW (borderline), PRELIMINARY (very early/sparse data)\n"
    "- **V16 holdout**: 19 films held out from training, tested blind — the only legitimate V16 validation until live predictions accumulate\n"
    "- **Actuals**: Official box office reporting\n"
    "- **Tier System**: SMALL (<$15M) / MID ($15-50M) / LARGE+ (>$50M)"
)

st.caption("V14/V15 predictions are preserved exactly as they were made at the time — they are NOT updated retroactively.")

show_cortex_badge()
