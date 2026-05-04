"""Page 1: Architecture — V21 cascade (V18.7 soft mixture + quantile window + Rule D tentpole gate + guarded Rule C)."""
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
    "V21 Model Architecture",
    "V18.7 soft-mixture cascade · 6 expanded-pool quantile regressors · V20-Clip · Rule D (Static Tentpole Gate) · guarded Rule C",
)

# -- Cascade diagram ---------------------------------------------------------
section(
    "V21 cascade flow",
    "Stage 1 → Stage 2 → 3 tier regressors (soft mixture) → V20-Clip window → Rule D (tentpole gate) → guarded Rule C.",
)

fig = go.Figure()
box_w, box_h = 0.14, 0.04

boxes = [
    {"x": 0.5, "y": 0.94, "color": DK2, "label": "INPUT", "sublabel": "72 Features",
     "hover": "36 static + 23 Google Trends + 13 Wikipedia features"},
    {"x": 0.5, "y": 0.78, "color": "#2ca02c", "label": "STAGE 1",
     "sublabel": "SMALL vs NON-SMALL",
     "hover": "CatBoost classifier: i=200, d=7, lr=0.02"},
    {"x": 0.18, "y": 0.58, "color": TIER_COLORS["SMALL"], "label": "SMALL",
     "sublabel": "Regressor", "hover": "MAE loss · 600 iters · 142 films"},
    {"x": 0.5, "y": 0.58, "color": ORANGE, "label": "STAGE 2",
     "sublabel": "MID vs LARGE+", "hover": "CatBoost classifier: i=400, d=5, lr=0.03"},
    {"x": 0.38, "y": 0.40, "color": TIER_COLORS["MID"], "label": "MID",
     "sublabel": "Regressor", "hover": "RMSE loss · 800 iters · 84 films"},
    {"x": 0.62, "y": 0.40, "color": TIER_COLORS["LARGE+"], "label": "LARGE+",
     "sublabel": "Regressor", "hover": "Quantile loss (α=0.5) · 500 iters · 51 films"},
    {"x": 0.5, "y": 0.22, "color": VIOLET, "label": "V20-CLIP",
     "sublabel": "Adaptive window",
     "hover": "6 expanded-pool quantile regressors → clip soft mixture to [Q10, Q90]"},
    {"x": 0.3, "y": 0.06, "color": "#22c55e", "label": "RULE D (tentpole)",
     "sublabel": "Static gate · guard < $60M",
     "hover": "Budget>=$125M + IP>=3 + Star>=9 + PredLog>=18.5 + MajorStudio. Fires first. 100% precision in backtest."},
    {"x": 0.7, "y": 0.06, "color": "#e377c2", "label": "RULE C (TMDB)",
     "sublabel": "Momentum · guard < $60M",
     "hover": "TMDB D7/D14 >= 25 -> LARGE+. Only fires if Rule D didn't."},
]

for b in boxes:
    fig.add_shape(type="rect", x0=b["x"] - box_w, y0=b["y"] - box_h,
                  x1=b["x"] + box_w, y1=b["y"] + box_h,
                  fillcolor=b["color"], line=dict(color=b["color"], width=2), layer="below")
    fig.add_annotation(x=b["x"], y=b["y"] + 0.012, text=f"<b>{b['label']}</b>",
                       showarrow=False, font=dict(color="white", size=12))
    fig.add_annotation(x=b["x"], y=b["y"] - 0.015, text=b["sublabel"],
                       showarrow=False, font=dict(color="white", size=9))
    fig.add_trace(go.Scatter(x=[b["x"]], y=[b["y"]], mode="markers",
                             marker=dict(size=1, color="rgba(0,0,0,0)"),
                             hoverinfo="text", hovertext=b["hover"], showlegend=False))

# Box half-height is 0.04. Arrows connect from bottom of upper box to top of lower box.
conns = [
    # INPUT(0.94) -> STAGE 1(0.78)
    (0.5, 0.90, 0.5, 0.82, "#666"),
    # STAGE 1(0.78) -> SMALL(0.58) + STAGE 2(0.58)
    (0.5, 0.74, 0.18, 0.62, TIER_COLORS["SMALL"]),
    (0.5, 0.74, 0.5, 0.62, ORANGE),
    # STAGE 2(0.58) -> MID(0.40) + LARGE+(0.40)
    (0.5, 0.54, 0.38, 0.44, TIER_COLORS["MID"]),
    (0.5, 0.54, 0.62, 0.44, TIER_COLORS["LARGE+"]),
    # SMALL(0.58) / MID(0.40) / LARGE+(0.40) -> V20-CLIP(0.22)
    (0.18, 0.54, 0.42, 0.26, VIOLET),
    (0.38, 0.36, 0.46, 0.26, VIOLET),
    (0.62, 0.36, 0.54, 0.26, VIOLET),
    # V20-CLIP(0.22) -> RULE D(0.06) and RULE C(0.06)
    (0.42, 0.18, 0.3, 0.10, "#22c55e"),
    (0.58, 0.18, 0.7, 0.10, "#e377c2"),
]
for x0, y0, x1, y1, c in conns:
    fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor=c)

# Rule C bypass: when override fires, LARGE+ regressor feeds RULE C directly, skipping V20-Clip.
fig.add_annotation(x=0.64, y=0.06, ax=0.76, ay=0.40, xref="x", yref="y", axref="x", ayref="y",
                   showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.8,
                   arrowcolor="#e377c2", opacity=0.8)
fig.add_annotation(x=0.86, y=0.22, text="<i>Rule C bypass</i><br>(TMDB override uses<br>LARGE+ directly)",
                   showarrow=False, font=dict(color="#e377c2", size=9), align="left")
fig.add_annotation(x=0.24, y=0.06, ax=0.62, ay=0.40, xref="x", yref="y", axref="x", ayref="y",
                   showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.8,
                   arrowcolor="#22c55e", opacity=0.8)
fig.add_annotation(x=0.04, y=0.22, text="<i>Rule D bypass</i><br>(static tentpole<br>gate → LARGE+)",
                   showarrow=False, font=dict(color="#22c55e", size=9), align="left")

fig.update_layout(showlegend=False, height=720, margin=dict(l=20, r=20, t=20, b=20),
                  xaxis=dict(range=[0, 1], visible=False, fixedrange=True),
                  yaxis=dict(range=[0, 1], visible=False, fixedrange=True),
                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# -- Tabbed details ----------------------------------------------------------
tab_tiers, tab_configs, tab_window, tab_ruled, tab_rulec, tab_features = st.tabs(
    ["Tier boundaries", "Classifier & regressor configs", "V20 quantile window",
     "Rule D (V21 tentpole gate)", "Rule C (TMDB, V20)", "Feature set"]
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

with tab_window:
    st.markdown(
        "**V20 adds 6 expanded-pool quantile regressors** (Q10 and Q90 for each of SMALL / MID / "
        "LARGE+). Each tier regressor is trained on its own films **plus 30% of each neighbor tier**, "
        "so the window can stretch when the classifier is uncertain."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            **Window construction**
            1. For a film with tier probabilities `(p_S, p_M, p_L)`, compute the soft-mixture
               point = `p_S·reg_S + p_M·reg_M + p_L·reg_L`.
            2. Predict expanded `Q10_t` and `Q90_t` for each tier.
            3. **Window low** = min of `Q10_t` weighted by probability; **window high** = max of
               `Q90_t` weighted by probability.
            4. Post-hoc sort-and-clamp to prevent quantile crossing.
            5. **V20-Clip point** = `clip(soft_mixture, window_low, window_high)`.
            """
        )
    with c2:
        st.code(
            "QREG_CONFIG = {\n"
            "    'iterations': 500,\n"
            "    'depth': 5,\n"
            "    'learning_rate': 0.02,\n"
            "    'l2_leaf_reg': 8,\n"
            "    'loss_function': 'Quantile:alpha=0.10/0.90',\n"
            "}\n"
            "NEIGH_PCT = 0.30  # include 30% of each neighbor tier",
            language="python",
        )
        st.caption(
            "CV result: V20-Clip alone → MAE $10.29M (-10.4% vs V18.0). Adding guarded Rule C "
            "→ $9.48M (-17.4%). See Performance page and V20 Model Story for the full arc."
        )

with tab_ruled:
    st.markdown(
        "**Rule D (Static Tentpole Gate)** is new in V21. It fires **before** Rule C when "
        "momentum signals (Trends, TMDB) are too sparse to be reliable — typically at D-18 or earlier."
    )
    st.markdown(
        """
        **Gate conditions (ALL must be true):**
        | Condition | Threshold | Rationale |
        |-----------|-----------|-----------|
        | Budget | ≥ $125M | Mega-budget = studio conviction |
        | IP Tier | ≥ 3 (high-profile) | Proven franchise with built-in audience |
        | Star Power | ≥ 9 | A-list lead with OW track record |
        | Predecessor OW (log) | ≥ 18.5 (~$108M+) | Prior installment was massive |
        | Major Studio | = 1 | Disney, WB, Universal, Paramount, Sony, Fox |
        | **Guard** | V20-Clip < $60M | Don't override if model already predicts high |

        **Action:** Force LARGE+ regressor output (skip classifier).

        **Backtest:** 100% precision (10/10 films that match → all were LARGE+). Zero false positives.
        Eliminates Aquaman 2 ($28M, pred_log=18.03), Indiana Jones 5 ($60M, pred_log=18.42), Fast X ($67M, pred_log=18.06).

        **First production use:** The Mandalorian & Grogu (May 4, 2026 at D-18).
        Classifier said MID 85% → Rule D overrode to LARGE+ $70.5M.
        """
    )

with tab_rulec:
    st.markdown(
        "Rule C is a **post-prediction safety net**: it runs after V20-Clip and "
        "can only raise a tier, never lower it. It exists because TMDB daily popularity data "
        "is only available for ~30 training films — too sparse for CatBoost to learn — but the raw "
        "signal correlates strongly with actual OW (Spearman r = 0.817 at D-14). "
        "**V20 adds a guard: Rule C only fires when the V20-Clip point is below $60M**, so already-"
        "large predictions (e.g. Oppenheimer-style tentpoles) aren't double-lifted."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            **Rules**

            | Condition | Action |
            |---|---|
            | `V20_CLIP ≥ $60M` | **Guard** — skip Rule C, trust V20 |
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
    st.caption("V20 inherits the full V18 feature set unchanged — 72 features. Breakdown by family:")
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
        st.markdown("**Wikipedia (13) — V18 new, reused in V20**")
        st.markdown(
            "- Rolling pageviews 3D / 7D / 14D + priors\n"
            "- Velocity (7D / prior 7D)\n"
            "- Peak day, cumulative, anchor day\n"
            "- Log-transformed versions\n\n"
            "→ corr(14D, OW) = **0.749** (vs Trends ~0.50)"
        )

show_cortex_badge()
