"""Page 6: Model history — version timeline, velocity, hyperparameters."""
import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from theme import apply_page_config, kpi_row, page_header, section, show_cortex_badge

apply_page_config("Model history", icon="📅")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "dashboard")


@st.cache_data
def load_json(name):
    with open(os.path.join(DATA_DIR, name)) as f:
        return json.load(f)


versions = pd.DataFrame(load_json("versions.json"))
sessions = pd.DataFrame(load_json("sessions.json"))
hparams = pd.DataFrame(load_json("hyperparams.json"))

page_header(
    "Model History",
    "V2 → V24 evolution — every version, active sessions, and the hyperparameters behind them.",
)

kpi_row([
    ("Model versions", str(len(versions)), "V2 → V24"),
    ("Best CV MAE",    "$9.89M",           "V24 @ -7d"),
    ("Best CV accuracy", "77.4%",          "V24 @ -3d"),
    ("Active sessions", str(len(sessions)), f"{sessions['hours'].sum():.0f}h active work"),
])

tab_versions, tab_velocity, tab_hpt = st.tabs(
    ["Version timeline", "Development velocity", "Hyperparameter tuning"]
)

# ---------------------------------------------------------------------------
with tab_versions:
    section(
        "Accuracy evolution",
        "CV accuracy by version. V16 is holdout-with-override; everything else is 5-fold GroupKFold.",
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=versions["version"], y=versions["accuracy"],
        marker_color=versions["category"].map({
            "Features": "#29B5E8", "Architecture": "#11567F",
            "Production": "#FF9F36",
        }),
        text=[f"{v:.1f}%" for v in versions["accuracy"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{text} accuracy<br>%{customdata}<extra></extra>",
        customdata=versions["description"],
    ))
    fig.update_layout(yaxis_title="CV accuracy %", height=380, yaxis_range=[55, 90],
                      margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    section("Version details")
    display = versions[["version", "date", "description", "features", "accuracy", "mae", "notes"]].copy()
    display.columns = ["Version", "Date", "Change", "Features", "CV Acc %", "MAE ($M)", "Notes"]
    st.dataframe(display, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
with tab_velocity:
    active_days = sessions["date"].nunique()
    total_hours = sessions["hours"].sum()
    calendar_span = 120

    kpi_row([
        ("Calendar span", f"{calendar_span} days", "Jan 28 → May 27"),
        ("Active days", str(active_days), f"{active_days/calendar_span*100:.0f}% of calendar"),
        ("Total hours", f"~{total_hours:.0f}h", f"~{total_hours/active_days:.1f}h / session"),
        ("Sessions", str(len(sessions)), "Individual work blocks"),
    ])

    section("Hours per session")
    sess = sessions.copy()
    sess["datetime"] = pd.to_datetime(sess["date"])
    fig = px.bar(sess, x="datetime", y="hours", text="hours",
                 hover_data=["work"], color_discrete_sequence=["#29B5E8"])
    fig.update_traces(texttemplate="%{text}h", textposition="outside")
    fig.update_layout(height=340, xaxis_title="Date", yaxis_title="Hours",
                      yaxis_range=[0, max(sess["hours"]) + 2])
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Session notes", expanded=False):
        for _, r in sess.iterrows():
            st.markdown(f"**{r['date']}** · {r['hours']}h — {r['work']}")

# ---------------------------------------------------------------------------
with tab_hpt:
    st.caption(
        f"{hparams['tested'].sum()} configurations evaluated across "
        f"{hparams['model'].nunique()} model components via grid + random search."
    )
    st.dataframe(
        hparams,
        hide_index=True,
        use_container_width=True,
        column_config={
            "model":  st.column_config.TextColumn("Model component", width="medium"),
            "param":  "Parameter",
            "tested": st.column_config.NumberColumn("# tested", format="%d"),
            "best":   "Best value",
            "range":  "Search range",
        },
    )
    section("Why V18 → V23b matters")
    st.markdown(
        "- **V18.7 soft mixture** — probability-weighted blend across 3 tier regressors lifted "
        "MAE $11.48M → $10.42M.\n"
        "- **V20 quantile window** — 6 expanded-pool quantile regressors give every film a learned "
        "[Q10, Q90] range.\n"
        "- **V22c Dual Rule C** — TMDB D14≥25 static + D7 surge (TMDB≥35 AND GT≥70). Fixes MK2 false positive.\n"
        "- **V23b Horror routing** — Genre-first split: horror → dedicated 2-bucket regressors ($17M KMeans split). "
        "Non-horror → standard cascade. Horror accuracy 73.5%.\n"
        "- **V23b deployed to Snowflake Model Registry** — Feature Store + CustomModel + batch inference via mv.run()."
    )

show_cortex_badge()
