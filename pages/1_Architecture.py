"""Page 1: Architecture — V24 cascade (Escape Velocity Detection + Demand Dominance)."""
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
    "V24 Model Architecture",
    "Escape velocity detection · Demand dominance · Horror routing · 3-tier cascade · V20-Clip · Rules C/D/E/F/G",
)

# -- Cascade diagram ---------------------------------------------------------
section(
    "V24 cascade flow",
    "Horror gate → Horror 2-bucket (or) Stage 1 → Stage 2 → 3 tier regressors (soft mixture) → V20-Clip → Rule E/F/G (velocity overrides) → Rule C/D.",
)

fig = go.Figure()
box_w, box_h = 0.14, 0.04

boxes = [
    {"x": 0.5, "y": 0.94, "color": DK2, "label": "INPUT", "sublabel": "72 Features",
     "hover": "36 static + 23 Google Trends + 13 Wikipedia features"},
    {"x": 0.5, "y": 0.80, "color": "#dc2626", "label": "HORROR?",
     "sublabel": "Genre routing",
     "hover": "If GENRE_HORROR=1 → dedicated horror path. Otherwise → standard cascade."},
    {"x": 0.18, "y": 0.64, "color": "#dc2626", "label": "HORROR CLF",
     "sublabel": "Small vs Large ($17M)",
     "hover": "CatBoost classifier: small (<$17M) vs large (≥$17M). Split determined by log-space KMeans on 68 horror films."},
    {"x": 0.08, "y": 0.48, "color": "#f87171", "label": "HORROR-S",
     "sublabel": "Regressor <$17M",
     "hover": "MAE loss · 500 iters · 41 films. Trained only on horror films <$17M."},
    {"x": 0.28, "y": 0.48, "color": "#991b1b", "label": "HORROR-L",
     "sublabel": "Regressor ≥$17M",
     "hover": "RMSE loss · 600 iters · 27 films. Trained only on horror films ≥$17M."},
    {"x": 0.68, "y": 0.64, "color": "#2ca02c", "label": "STAGE 1",
     "sublabel": "SMALL vs NON-SMALL",
     "hover": "CatBoost classifier: i=300, d=7, lr=0.015. Non-horror only."},
    {"x": 0.55, "y": 0.48, "color": TIER_COLORS["SMALL"], "label": "SMALL",
     "sublabel": "Regressor", "hover": "MAE loss · 600 iters"},
    {"x": 0.68, "y": 0.48, "color": ORANGE, "label": "STAGE 2",
     "sublabel": "MID vs LARGE+", "hover": "CatBoost classifier: i=400, d=5, lr=0.03"},
    {"x": 0.62, "y": 0.32, "color": TIER_COLORS["MID"], "label": "MID",
     "sublabel": "Regressor", "hover": "RMSE loss · 800 iters"},
    {"x": 0.78, "y": 0.32, "color": TIER_COLORS["LARGE+"], "label": "LARGE+",
     "sublabel": "Regressor", "hover": "Quantile loss (α=0.5) · 500 iters"},
    {"x": 0.68, "y": 0.16, "color": VIOLET, "label": "V20-CLIP",
     "sublabel": "Adaptive window",
     "hover": "6 quantile regressors → clip soft mixture to [Q10, Q90]"},
    {"x": 0.5, "y": 0.04, "color": "#e377c2", "label": "RULES C–G",
     "sublabel": "Override gates",
     "hover": "Rule E: demand override. Rule F: escape velocity. Rule G: demand dominance. Rule C/D: TMDB + tentpole."},
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
    (0.5, 0.90, 0.5, 0.84, "#666"),
    (0.5, 0.76, 0.18, 0.68, "#dc2626"),
    (0.5, 0.76, 0.68, 0.68, "#2ca02c"),
    (0.18, 0.60, 0.08, 0.52, "#f87171"),
    (0.18, 0.60, 0.28, 0.52, "#991b1b"),
    (0.68, 0.60, 0.55, 0.52, TIER_COLORS["SMALL"]),
    (0.68, 0.60, 0.68, 0.52, ORANGE),
    (0.68, 0.44, 0.62, 0.36, TIER_COLORS["MID"]),
    (0.68, 0.44, 0.78, 0.36, TIER_COLORS["LARGE+"]),
    (0.55, 0.44, 0.60, 0.20, VIOLET),
    (0.62, 0.28, 0.64, 0.20, VIOLET),
    (0.78, 0.28, 0.72, 0.20, VIOLET),
    (0.68, 0.12, 0.5, 0.08, "#e377c2"),
]
for x0, y0, x1, y1, c in conns:
    fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor=c)

fig.add_annotation(x=0.18, y=0.56, text="<i>Weighted blend:</i><br>OW = (1-p_large)·reg_S + p_large·reg_L",
                   showarrow=False, font=dict(color="#dc2626", size=9), align="center")

fig.update_layout(showlegend=False, height=720, margin=dict(l=20, r=20, t=20, b=20),
                  xaxis=dict(range=[0, 1], visible=False, fixedrange=True),
                  yaxis=dict(range=[0, 1], visible=False, fixedrange=True),
                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# -- Tabbed details ----------------------------------------------------------
tab_tiers, tab_horror, tab_configs, tab_window, tab_velocity, tab_ruled, tab_rulec, tab_features = st.tabs(
    ["Tier boundaries", "Horror routing", "Non-horror configs", "V20 quantile window",
     "Rules E/F/G (V24 new)", "Rule D (tentpole gate)", "Rule C (TMDB)", "Feature set"]
)

with tab_tiers:
    st.markdown(
        """
        | Tier | Revenue range | Training films |
        |------|---------------|----------------|
        | **SMALL** | < $15M | 147 (51%) |
        | **MID** | $15–50M | 87 (30%) |
        | **LARGE+** | ≥ $50M | 53 (18%) |
        """
    )
    st.caption(
        "287 training films total. Horror subset: 68 films (41 small, 27 large by the $17M horror split)."
    )

with tab_horror:
    st.markdown(
        "**V23b introduces horror-first routing.** Before any tier classification, the model checks "
        "`GENRE_HORROR=1`. If true, the movie bypasses the standard cascade entirely."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            **Horror 2-bucket architecture**

            1. **Horror classifier** — CatBoost binary: small (<$17M) vs large (≥$17M)
            2. **Horror-Small regressor** — trained on 41 horror films <$17M (MAE loss)
            3. **Horror-Large regressor** — trained on 27 horror films ≥$17M (RMSE loss)
            4. **Weighted blend** — `OW = (1-p_large)·reg_S + p_large·reg_L`

            The $17M split was determined by log-space KMeans on 68 horror training films.
            """
        )
    with c2:
        st.code(
            "HORROR_SPLIT = $17M  # log-space KMeans\n\n"
            "HORROR_CLASSIFIER = {\n"
            "    iterations: 300, depth: 5,\n"
            "    learning_rate: 0.02\n"
            "}\n\n"
            "HORROR_SMALL_REG = {\n"
            "    iterations: 500, depth: 5,\n"
            "    lr: 0.02, loss: 'MAE'\n"
            "}\n\n"
            "HORROR_LARGE_REG = {\n"
            "    iterations: 600, depth: 5,\n"
            "    lr: 0.015, loss: 'RMSE'\n"
            "}",
            language="python",
        )
    st.markdown(
        "**Why?** Budget-anchoring bias: horror films are almost always low-budget ($1-30M) "
        "but can break out to $50M+. The standard cascade's budget feature anchors predictions low. "
        "Separating horror removes this tension. CV result: **73.5% accuracy** on 68 horror films, **$8.73M MAE**."
    )

with tab_configs:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Stage 1 — SMALL vs NON-SMALL** (non-horror only)")
        st.code(
            "CatBoostClassifier(\n"
            "    iterations=300,\n"
            "    depth=7,\n"
            "    learning_rate=0.015,\n"
            ")",
            language="python",
        )
    with c2:
        st.markdown("**Stage 2 — MID vs LARGE+** (non-horror only)")
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

    st.markdown("**Tier-specific regressors** (non-horror)")
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

with tab_velocity:
    st.markdown(
        "**V24 introduces Escape Velocity Detection** — three rules that override the classifier "
        "when demand signals are undeniable. Motivated by the Backrooms case (V23c predicted $16.5M, "
        "actual tracking $60-80M from $10-11M Thursday previews)."
    )
    st.markdown("---")
    st.markdown("### Rule E — Demand Override (V23c)")
    st.markdown(
        "When `ROLLING_7D ≥ 200` AND `YT_COMMENTS ≥ 10,000`, P_SMALL is zeroed and probabilities "
        "renormalized. **Rationale:** No SMALL film in 287 training examples has both GT7>200 and YT>10K."
    )
    st.markdown("### Rule F — Escape Velocity (V24 new)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            **Conditions (ALL must be true):**
            | Signal | Threshold |
            |--------|-----------|
            | ROLLING_7D | ≥ 200 |
            | LOG_SLOPE_14_TO_3 | ≥ 0.04 |
            | YT_COMMENTS | ≥ 5,000 |

            **Action:** Zero out P_SMALL, renormalize.

            **Backtest:** 0 false positives in 287-film training set.
            Historical minimum OW when rule fires: $21.6M (Nosferatu).
            """
        )
    with c2:
        st.code(
            "LOG_SLOPE_14_TO_3 =\n"
            "  (LN(ROLLING_3D) - LN(ROLLING_14D)) / 11\n\n"
            "# Measures exponential acceleration\n"
            "# of demand in the 14-to-3 day window.\n"
            "#\n"
            "# Backrooms: 0.0422\n"
            "# Longlegs:  0.159 (but YT=1250 < 5K)\n"
            "# Threshold: 0.04",
            language="python",
        )
    st.markdown("### Rule G — Demand Dominance (V24 new)")
    st.markdown(
        "When Rule F fires, the prediction is blended toward the **LARGE+ regressor** based on "
        "demand intensity. Budget becomes irrelevant — demand has taken over."
    )
    st.markdown(
        """
        | R7D Level | LARGE+ Weight | Blend Formula |
        |-----------|--------------|---------------|
        | 200–300 | 40% | `0.6·pred_MID + 0.4·pred_L+` |
        | 300–400 | 60% | `0.4·pred_MID + 0.6·pred_L+` |
        | > 400 | 80% | `0.2·pred_MID + 0.8·pred_L+` |
        """
    )
    st.markdown(
        "**Backrooms at D-3:** R7D=326 → 60% weight → `0.4×$26.9M + 0.6×$69.6M` = **$52.5M** "
        "(vs V23c's $16.5M). At D0 with R7D=455 → 80% weight → **$65.4M**."
    )
    st.success(
        "**Design principle:** When demand signals (GT velocity + YouTube social proof) exceed "
        "ALL historical SMALL-tier examples, budget becomes irrelevant. Let demand dominate."
    )

    st.markdown("### New Velocity Features (V24)")
    st.markdown(
        "Added to `OW_PREDICTION_FEATURES_V` as CatBoost inputs + rule triggers:"
    )
    st.markdown(
        """
        | Feature | Formula | Purpose |
        |---------|---------|---------|
        | `VEL_3v7` | ROLLING_3D / ROLLING_7D | Short-term acceleration |
        | `VEL_7v14` | ROLLING_7D / ROLLING_14D | Medium-term acceleration |
        | `LOG_SLOPE_14_TO_3` | (LN(R3D) - LN(R14D)) / 11 | Exponential ramp rate |
        """
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
    st.caption("V24 uses 75 features (72 from V18 + 3 velocity features):")
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
        st.markdown("**Google Trends (26)**")
        st.markdown(
            "- Rolling: 3D / 5D / 7D / 14D / 21D + priors (9)\n"
            "- Velocity 3D / 5D / 7D (3)\n"
            "- **VEL_3v7 / VEL_7v14 / LOG_SLOPE (3) ← V24 new**\n"
            "- Cumulative, volatility, peak, days_with_data, earliest (5)\n"
            "- Interactions (6) — Trends × IP / Genre / Star / Intent"
        )
    with c3:
        st.markdown("**Wikipedia (13)**")
        st.markdown(
            "- Rolling pageviews 3D / 7D / 14D + priors\n"
            "- Velocity (7D / prior 7D)\n"
            "- Peak day, cumulative, anchor day\n"
            "- Log-transformed versions\n\n"
            "→ corr(14D, OW) = **0.749** (vs Trends ~0.50)"
        )

show_cortex_badge()
