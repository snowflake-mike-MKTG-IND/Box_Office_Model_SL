"""Page 1: Architecture — V18 cascade + Rule C (single owner)."""
import plotly.graph_objects as go
import streamlit as st

from theme import (
    DK1,
    DK2,
    ORANGE,
    SF_BLUE,
    TIER_COLORS,
    VIOLET,
    apply_page_config,
    page_header,
    section,
    show_cortex_badge,
)

apply_page_config("Architecture", icon="🏗️")

page_header(
    "Model Architecture",
    "2-stage cascade classification · tier-specific regressors · Rule C TMDB override",
)

# -- Cascade diagram ---------------------------------------------------------
section(
    "Cascade flow",
    "Every prediction flows Stage 1 → Stage 2 → tier-specific regressor → Rule C override.",
)

fig = go.Figure()
box_w, box_h = 0.12, 0.065

boxes = [
    {"x": 0.5, "y": 0.92, "color": DK2, "label": "INPUT", "sublabel": "72 Features",
     "hover": "36 static + 23 Google Trends + 13 Wikipedia features"},
    {"x": 0.5, "y": 0.74, "color": "#2ca02c", "label": "STAGE 1",
     "sublabel": "SMALL vs NON-SMALL",
     "hover": "CatBoost classifier: i=200, d=7, lr=0.02"},
    {"x": 0.22, "y": 0.50, "color": TIER_COLORS["SMALL"], "label": "SMALL",
     "sublabel": "Regressor", "hover": "MAE loss · 600 iters · 142 films"},
    {"x": 0.5, "y": 0.50, "color": ORANGE, "label": "STAGE 2",
     "sublabel": "MID vs LARGE+", "hover": "CatBoost classifier: i=400, d=5, lr=0.03"},
    {"x": 0.36, "y": 0.28, "color": TIER_COLORS["MID"], "label": "MID",
     "sublabel": "Regressor", "hover": "RMSE loss · 800 iters · 84 films"},
    {"x": 0.64, "y": 0.28, "color": TIER_COLORS["LARGE+"], "label": "LARGE+",
     "sublabel": "Regressor", "hover": "Quantile loss (α=0.5) · 500 iters · 50 films"},
    {"x": 0.5, "y": 0.08, "color": "#e377c2", "label": "RULE C",
     "sublabel": "TMDB Override",
     "hover": "Post-prediction safety net. Can only raise tier."},
]

for b in boxes:
    fig.add_shape(type="rect", x0=b["x"] - box_w, y0=b["y"] - box_h,
                  x1=b["x"] + box_w, y1=b["y"] + box_h,
                  fillcolor=b["color"], line=dict(color=b["color"], width=2), layer="below")
    fig.add_annotation(x=b["x"], y=b["y"] + 0.02, text=f"<b>{b['label']}</b>",
                       showarrow=False, font=dict(color="white", size=13))
    fig.add_annotation(x=b["x"], y=b["y"] - 0.02, text=b["sublabel"],
                       showarrow=False, font=dict(color="white", size=10))
    fig.add_trace(go.Scatter(x=[b["x"]], y=[b["y"]], mode="markers",
                             marker=dict(size=1, color="rgba(0,0,0,0)"),
                             hoverinfo="text", hovertext=b["hover"], showlegend=False))

conns = [
    (0.5, 0.855, 0.5, 0.805, "#666"), (0.5, 0.675, 0.5, 0.625, "#666"),
    (0.35, 0.625, 0.22, 0.565, TIER_COLORS["SMALL"]),
    (0.5, 0.625, 0.5, 0.565, ORANGE),
    (0.5, 0.435, 0.5, 0.385, "#666"),
    (0.42, 0.385, 0.36, 0.345, TIER_COLORS["MID"]),
    (0.58, 0.385, 0.64, 0.345, TIER_COLORS["LARGE+"]),
    (0.22, 0.435, 0.35, 0.145, "#e377c2"),
    (0.36, 0.215, 0.42, 0.145, "#e377c2"),
    (0.64, 0.215, 0.58, 0.145, "#e377c2"),
]
for x0, y0, x1, y1, c in conns:
    fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor=c)

fig.update_layout(showlegend=False, height=520, margin=dict(l=20, r=20, t=20, b=20),
                  xaxis=dict(range=[0, 1], visible=False, fixedrange=True),
                  yaxis=dict(range=[0, 1], visible=False, fixedrange=True, scaleanchor="x"),
                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# -- Tabbed details ----------------------------------------------------------
tab_tiers, tab_configs, tab_rulec, tab_features = st.tabs(
    ["Tier boundaries", "Classifier & regressor configs", "Rule C (TMDB override)", "Feature set"]
)

with tab_tiers:
    st.markdown(
        """
        | Tier | Revenue range | Training films |
        |------|---------------|----------------|
        | **SMALL** | < $15M | 142 (52%) |
        | **MID** | $15–50M | 84 (30%) |
        | **LARGE+** | ≥ $50M | 50 (18%) |
        """
    )
    st.caption(
        "Why 3 tiers and not 4? V13's 4-tier split had only 27 LARGE and 19 BLOCKBUSTER films — "
        "LARGE accuracy cratered at 27%. Consolidating into LARGE+ (50 films) was the breakthrough "
        "behind V14's jump to 71.5%."
    )

with tab_configs:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Stage 1 — SMALL vs NON-SMALL**")
        st.code(
            "CatBoostClassifier(\n"
            "    iterations=200,\n"
            "    depth=7,\n"
            "    learning_rate=0.02,\n"
            ")",
            language="python",
        )
    with c2:
        st.markdown("**Stage 2 — MID vs LARGE+**")
        st.code(
            "CatBoostClassifier(\n"
            "    iterations=400,\n"
            "    depth=5,\n"
            "    learning_rate=0.03,\n"
            "    l2_leaf_reg=5,\n"
            "    subsample=0.8,\n"
            ")",
            language="python",
        )

    st.markdown("**Tier-specific regressors**")
    rc1, rc2, rc3 = st.columns(3)
    for col, tier, body, note in [
        (rc1, "SMALL",
         "iterations=600\ndepth=5\nlearning_rate=0.02\nl2_leaf_reg=5\nloss='MAE'",
         "MAE loss for many low-variance films"),
        (rc2, "MID",
         "iterations=800\ndepth=6\nlearning_rate=0.015\nl2_leaf_reg=5\nloss='RMSE'",
         "Balanced loss across the widest revenue band"),
        (rc3, "LARGE+",
         "iterations=500\ndepth=5\nlearning_rate=0.02\nl2_leaf_reg=8\nloss='Quantile:α=0.5'",
         "Quantile loss handles high-variance blockbusters"),
    ]:
        with col:
            st.markdown(f"**{tier}**")
            st.code(body, language="python")
            st.caption(note)

with tab_rulec:
    st.markdown(
        "Rule C is a **post-prediction safety net**: it runs after the cascade and "
        "can only raise a tier, never lower it. It exists because TMDB daily popularity data "
        "is only available for ~30 training films — too sparse for CatBoost to learn — but the raw "
        "signal correlates strongly with actual OW (Spearman r = 0.817 at D-14)."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            **Rules**

            | Condition | Action |
            |---|---|
            | `TMDB_POP_D14 ≥ 25` | Force minimum **LARGE+** |
            | `TMDB_POP_D14 ≥ 15` AND `D7/D14 ≥ 1.3` | Force minimum **MID** |
            | Otherwise | Trust the model |
            """
        )
    with c2:
        st.markdown(
            """
            **Holdout validation (19 blind films)**

            | Metric | Without | With Rule C |
            |---|---|---|
            | Tier accuracy | 63.2% | **84.2%** |
            | Overrides applied | 0 | 4 |
            | Correct / wrong | — | **4 / 0** |
            """
        )
        st.caption(
            "The momentum gate (D7/D14 ≥ 1.3) prevents false positives — e.g. PRIMATE had "
            "D14 = 24.6 but momentum 0.96, correctly kept as SMALL."
        )

with tab_features:
    st.caption("V18 uses 72 features. Breakdown by family:")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Static (36)**")
        st.markdown(
            "- YouTube + sentiment (7)\n"
            "- Movie attributes (8) — budget, runtime, TMDB pop, month, peak, "
            "predecessor OW, `IS_MAJOR_STUDIO`\n"
            "- Star power (4)\n"
            "- Genre (5) · Rating (4) · IP/Franchise (5)\n"
            "- TMDB daily (3) — D14, D7, momentum"
        )
    with c2:
        st.markdown("**Google Trends (23)**")
        st.markdown(
            "- Rolling: 3D / 5D / 7D / 14D / 21D + priors (9)\n"
            "- Velocity 3D / 5D / 7D (3)\n"
            "- Cumulative, volatility, peak, days_with_data, earliest (5)\n"
            "- Interactions (6) — Trends × IP / Genre / Star / Intent"
        )
    with c3:
        st.markdown("**Wikipedia (13) — V18 new**")
        st.markdown(
            "- Rolling pageviews 3D / 7D / 14D + priors\n"
            "- Velocity (7D / prior 7D)\n"
            "- Peak day, cumulative, anchor day\n"
            "- Log-transformed versions\n\n"
            "→ corr(14D, OW) = **0.749** (vs Trends ~0.50)"
        )

show_cortex_badge()
