"""Page 7: Recent Predictions — live tracking of predictions vs actuals."""
import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from theme import TIER_COLORS, apply_page_config, kpi_row, page_header, section, show_cortex_badge

apply_page_config("Recent Predictions", icon="🎯")


def load_weekends():
    p = os.path.join(os.path.dirname(__file__), "..", "data", "dashboard", "weekends.json")
    with open(p) as f:
        return json.load(f)


data = load_weekends()


def actual_tier(ow):
    if ow is None:
        return None
    if ow < 15:
        return "SMALL"
    if ow < 50:
        return "MID"
    return "LARGE+"


def confidence_level(tier_prob, trends_days, days_until, overridden):
    if overridden:
        return "HIGH"
    if tier_prob >= 0.70 and trends_days >= 7:
        return "HIGH"
    if tier_prob >= 0.60 or trends_days >= 7:
        return "MEDIUM"
    if tier_prob < 0.55 and (trends_days < 5 or days_until > 14):
        return "PRELIMINARY"
    return "LOW"


page_header(
    "Recent Predictions vs Actuals",
    "Every prediction was made before the weekend using the model in production at the time.",
)

# ---- Aggregates --------------------------------------------------------------
rows = []
for w in data["past_weekends"]:
    for m in w["movies"]:
        rows.append({
            "Weekend": w["weekend"], "Model": w["model"], "Movie": m["movie"],
            "Studio": m["studio"], "Pred Tier": m["pred_tier"],
            "Pred OW": m["pred_ow"], "Conf Low": m["conf_low"], "Conf High": m["conf_high"],
            "Actual OW": m["actual_ow"], "Actual Tier": actual_tier(m["actual_ow"]),
            "Industry": m.get("industry"),
            "Note": m.get("note", ""),
        })
df = pd.DataFrame(rows)
df["Error"] = (df["Pred OW"] - df["Actual OW"]).round(2)
df["Tier Correct"] = df["Pred Tier"] == df["Actual Tier"]
df["In CI"] = df.apply(lambda r: r["Conf Low"] <= r["Actual OW"] <= r["Conf High"], axis=1)

kpi_row([
    ("Tier accuracy", f"{df['Tier Correct'].sum()}/{len(df)}",
     f"{df['Tier Correct'].mean() * 100:.0f}%"),
    ("MAE", f"${df['Error'].abs().mean():.2f}M", f"{len(df)} films"),
    ("Within CI", f"{df['In CI'].sum()}/{len(df)}",
     f"{df['In CI'].mean() * 100:.0f}%"),
    ("Weekends tracked", str(len(data['past_weekends'])), None),
])

tab_chart, tab_past, tab_upcoming = st.tabs(
    ["Predicted vs actual", "Past weekends", "Upcoming predictions"]
)

# ---- Chart -------------------------------------------------------------------
with tab_chart:
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Predicted", x=df["Movie"], y=df["Pred OW"],
                         marker_color="#29B5E8",
                         error_y=dict(
                             type='data', symmetric=False,
                             array=(df["Conf High"] - df["Pred OW"]).tolist(),
                             arrayminus=(df["Pred OW"] - df["Conf Low"]).tolist(),
                             color="#11567F", thickness=2, width=6,
                         ),
                         text=[f"${v:.1f}M" for v in df["Pred OW"]], textposition="outside"))
    fig.add_trace(go.Bar(name="Actual", x=df["Movie"], y=df["Actual OW"],
                         marker_color="#11567F",
                         text=[f"${v:.1f}M" for v in df["Actual OW"]], textposition="outside"))
    industry = df["Industry"].fillna(0)
    if industry.sum() > 0:
        fig.add_trace(go.Bar(name="Industry projection", x=df["Movie"], y=industry,
                             marker_color="#7D44CF",
                             text=[f"${v:.0f}M" if v > 0 else "" for v in industry],
                             textposition="outside"))
    fig.update_layout(barmode="group", height=480, yaxis_title="Opening weekend ($M)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"),
                      xaxis_tickangle=-30, margin=dict(t=40))
    st.plotly_chart(fig, use_container_width=True)

    section("Accuracy by tier")
    tier_stats = (df.groupby("Pred Tier")
                  .agg(Count=("Movie", "size"),
                       MAE=("Error", lambda s: round(s.abs().mean(), 2)),
                       Correct=("Tier Correct", "sum")))
    tier_stats["Accuracy %"] = (tier_stats["Correct"] / tier_stats["Count"] * 100).round(0)
    st.dataframe(tier_stats, use_container_width=True)

# ---- Past --------------------------------------------------------------------
with tab_past:
    for w in data["past_weekends"]:
        with st.expander(f"**{w['weekend']}** — {w['dates']}  ·  {w['model']}", expanded=False):
            sub = df[df["Weekend"] == w["weekend"]][[
                "Movie", "Studio", "Pred Tier", "Actual Tier", "Tier Correct",
                "Pred OW", "Industry", "Actual OW", "Error", "In CI", "Note"
            ]].copy()
            for c in ["Pred OW", "Industry", "Actual OW"]:
                sub[c] = sub[c].apply(lambda v: f"${v:.2f}M" if pd.notna(v) else "—")
            sub["Error"] = sub["Error"].apply(lambda v: f"${v:+.2f}M")
            sub["Tier Correct"] = sub["Tier Correct"].map({True: "✓", False: "✗"})
            sub["In CI"] = sub["In CI"].map({True: "✓", False: "✗"})
            st.dataframe(sub, use_container_width=True, hide_index=True)

# ---- Upcoming ----------------------------------------------------------------
with tab_upcoming:
    st.caption(
        "Confidence gates: **HIGH** (≥70% tier prob + ≥7d trends), **MEDIUM** (≥60% or ≥7d), "
        "**LOW** (borderline), **PRELIMINARY** (very early / sparse data)."
    )
    for u in data["upcoming"]:
        with st.expander(f"**{u['weekend']}** — {u['dates']}  ·  {u['model']}", expanded=True):
            for m in u["movies"]:
                level = confidence_level(m.get("tier_prob", 0.5), m.get("trends_days", 0),
                                         m.get("days_until", 99), m.get("overridden", False))
                color = TIER_COLORS.get(m["pred_tier"], "#666")
                c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
                c1.markdown(f"**{m['movie']}**  \n_{m['studio']}_")
                c2.markdown(f"Tier: <span style='color:{color}; font-weight:600'>{m['pred_tier']}</span>",
                            unsafe_allow_html=True)
                c3.markdown(
                    f"<div style='font-weight:700; font-size:1.05rem;'>${m['pred_ow']:.1f}M</div>"
                    f"<div style='color:#666; font-size:0.85rem;'>${m['conf_low']:.1f}–${m['conf_high']:.1f}M</div>",
                    unsafe_allow_html=True,
                )
                c4.markdown(f"Horizon: {m.get('horizon', '—')}  \nConfidence: **{level}**")
                if all(k in m for k in ('p_small', 'p_mid', 'p_large')):
                    ps, pm, pl = m['p_small'], m['p_mid'], m['p_large']
                    small_c = TIER_COLORS.get('SMALL', '#999')
                    mid_c = TIER_COLORS.get('MID', '#999')
                    large_c = TIER_COLORS.get('LARGE+', '#999')
                    st.markdown(
                        f"<div style='font-size:0.82rem; color:#555; margin-top:6px;'>"
                        f"Tier probabilities: "
                        f"<span style='color:{small_c}; font-weight:600'>SMALL {ps*100:.0f}%</span> · "
                        f"<span style='color:{mid_c}; font-weight:600'>MID {pm*100:.0f}%</span> · "
                        f"<span style='color:{large_c}; font-weight:600'>LARGE+ {pl*100:.0f}%</span>"
                        f"</div>"
                        f"<div style='display:flex; height:8px; border-radius:4px; overflow:hidden; margin-top:4px; margin-bottom:4px;'>"
                        f"<div style='width:{ps*100:.1f}%; background:{small_c};'></div>"
                        f"<div style='width:{pm*100:.1f}%; background:{mid_c};'></div>"
                        f"<div style='width:{pl*100:.1f}%; background:{large_c};'></div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                if m.get("overridden"):
                    st.caption(f"Rule C active — {m.get('reason', '')}")
                if level == "PRELIMINARY":
                    st.caption(f"Preliminary — only {m.get('trends_days', 0)}d of trends data, "
                               f"tier prob {m.get('tier_prob', 0):.0%}. Expect revision.")
                st.caption(m.get("note", ""))
                st.markdown("---")

show_cortex_badge()
