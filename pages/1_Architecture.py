"""Page 1: Architecture — V28-A rule-free learned meta-combiner."""
import plotly.graph_objects as go
import streamlit as st

from theme import (
    DK1,
    DK2,
    ORANGE,
    SF_BLUE,
    TEAL,
    TIER_COLORS,
    VIOLET,
    apply_page_config,
    page_header,
    section,
    show_cortex_badge,
)

apply_page_config("Architecture", icon="🏗️")

page_header(
    "V28-A Model Architecture",
    "Rule-free learned meta-combiner · CatBoost + TabPFN soft-vote base · per-tier regressors · conformal bands · calibrated breakout odds",
)

st.info(
    "**Retired in V25 → V28-A.** The hand-coded rule stack — horror routing, V20-Clip, and "
    "Rules C / D / E / F / G — was **removed**. V28-A learns the *combine* step those rules used to "
    "hard-code, via a small meta-combiner. It is fully rule-free. (The detailed rule history lives on "
    "the V24 / V25 Model Story pages.)"
)

# -- Cascade diagram ---------------------------------------------------------
section(
    "V28-A flow",
    "Features → CatBoost + TabPFN soft-vote classifier → per-tier $ regressors → soft mixture → "
    "learned meta-combiner g → FINAL = 0.7·g + 0.3·mixture → conformal bands + breakout odds.",
)

fig = go.Figure()
box_w, box_h = 0.15, 0.042

boxes = [
    {"x": 0.5, "y": 0.93, "color": DK2, "label": "INPUT", "sublabel": "Features + OOF-stack",
     "hover": "STATIC_WIKI feature set (36 static + Google Trends + Wikipedia) plus one stacked OOF point estimate (log-OW from a regressor on the full feature set)."},
    {"x": 0.5, "y": 0.78, "color": SF_BLUE, "label": "CLASSIFIER", "sublabel": "CatBoost + TabPFN (soft-vote)",
     "hover": "Single 3-class tier classifier. CatBoost (tuned via the V27 sweep) and TabPFN (pretrained transformer) each predict_proba; probabilities are AVERAGED → tier probs (SMALL / MID / LARGE+). No two-stage cascade."},
    {"x": 0.18, "y": 0.62, "color": TIER_COLORS["SMALL"], "label": "SMALL reg", "sublabel": "$ point",
     "hover": "CatBoost regressor trained on SMALL films (MAE loss). Produces a SMALL-tier dollar point."},
    {"x": 0.5, "y": 0.62, "color": ORANGE, "label": "MID reg", "sublabel": "$ point",
     "hover": "CatBoost regressor trained on MID films (RMSE loss). Produces a MID-tier dollar point."},
    {"x": 0.82, "y": 0.62, "color": VIOLET, "label": "LARGE+ reg", "sublabel": "$ point",
     "hover": "CatBoost regressor trained on LARGE+ films (Quantile α=0.5). Missing tier falls back to the global stack point."},
    {"x": 0.5, "y": 0.46, "color": TEAL, "label": "SOFT MIXTURE", "sublabel": "Σ prob · point",
     "hover": "Probability-weighted blend of the three tier points: mix = Σ_t prob_t · point_t."},
    {"x": 0.5, "y": 0.30, "color": "#0E7C5B", "label": "META-COMBINER g", "sublabel": "CatBoost d3 · MAE",
     "hover": "Learned combine step. CatBoost(iters=400, depth=3, lr=0.03, l2=6, MAE) over 7 meta-features = [log tier-points (3), tier-probs (3), log mixture (1)] → log(OW). Fit on inner-OOF base preds (nested stacking)."},
    {"x": 0.5, "y": 0.16, "color": DK1, "label": "FINAL", "sublabel": "0.7·g + 0.3·mixture",
     "hover": "FINAL = 0.7 · exp(g) + 0.3 · mixture (BLEND_W = 0.7). The learned combiner does most of the work; the mixture term stabilizes it."},
    {"x": 0.27, "y": 0.03, "color": "#7D44CF", "label": "CONFORMAL BANDS", "sublabel": "bear / base / bull",
     "hover": "Per-tier expanded-pool quantile regressors (q10/q90) widened by a CQR conformal constant (calibrated on a held-out 20% slice), sampled → bear / base / bull range."},
    {"x": 0.73, "y": 0.03, "color": "#B83280", "label": "BREAKOUT ODDS", "sublabel": "P(LARGE+) + flag",
     "hover": "breakout_prob = P(LARGE+). Calibrated buckets + an upside flag when P(LARGE+) ≥ 0.30 while the point tier is below LARGE+."},
]

for b in boxes:
    fig.add_shape(type="rect", x0=b["x"] - box_w, y0=b["y"] - box_h,
                  x1=b["x"] + box_w, y1=b["y"] + box_h,
                  fillcolor=b["color"], line=dict(color=b["color"], width=2), layer="below")
    fig.add_annotation(x=b["x"], y=b["y"] + 0.013, text=f"<b>{b['label']}</b>",
                       showarrow=False, font=dict(color="white", size=12))
    fig.add_annotation(x=b["x"], y=b["y"] - 0.016, text=b["sublabel"],
                       showarrow=False, font=dict(color="white", size=9))
    fig.add_trace(go.Scatter(x=[b["x"]], y=[b["y"]], mode="markers",
                             marker=dict(size=1, color="rgba(0,0,0,0)"),
                             hoverinfo="text", hovertext=b["hover"], showlegend=False))

# Arrows connect from bottom of upper box to top of lower box (box half-height 0.042).
conns = [
    (0.5, 0.888, 0.5, 0.822, "#888"),
    (0.5, 0.738, 0.18, 0.662, SF_BLUE),
    (0.5, 0.738, 0.5, 0.662, SF_BLUE),
    (0.5, 0.738, 0.82, 0.662, SF_BLUE),
    (0.18, 0.578, 0.5, 0.502, TEAL),
    (0.5, 0.578, 0.5, 0.502, TEAL),
    (0.82, 0.578, 0.5, 0.502, TEAL),
    (0.5, 0.418, 0.5, 0.342, "#0E7C5B"),
    (0.5, 0.258, 0.5, 0.202, DK1),
    (0.5, 0.118, 0.27, 0.072, "#7D44CF"),
    (0.5, 0.118, 0.73, 0.072, "#B83280"),
]
for x0, y0, x1, y1, c in conns:
    fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor=c)

# annotate the meta-feature path + blend
fig.add_annotation(x=0.80, y=0.38, text="<i>probs + tier points also<br>feed g as meta-features</i>",
                   showarrow=False, font=dict(color="#0E7C5B", size=9), align="center")
fig.add_annotation(x=0.80, y=0.225, text="<i>mixture also feeds<br>FINAL (0.3 weight)</i>",
                   showarrow=False, font=dict(color=DK1, size=9), align="center")

fig.update_layout(showlegend=False, height=720, margin=dict(l=20, r=20, t=20, b=20),
                  xaxis=dict(range=[0, 1], visible=False, fixedrange=True),
                  yaxis=dict(range=[0, 1], visible=False, fixedrange=True),
                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# -- Tabbed details ----------------------------------------------------------
tab_tiers, tab_clf, tab_regs, tab_meta, tab_bands, tab_breakout, tab_leak, tab_features = st.tabs(
    ["Tier boundaries", "Base classifier", "Per-tier regressors", "Meta-combiner (g)",
     "Uncertainty bands", "Breakout layer", "Leakage control", "Feature set"]
)

with tab_tiers:
    st.markdown(
        """
        | Tier | Revenue range | Backtest films |
        |------|---------------|----------------|
        | **SMALL** | < $15M | 148 (51%) |
        | **MID** | $15–50M | 88 (31%) |
        | **LARGE+** | ≥ $50M | 52 (18%) |
        """
    )
    st.caption(
        "288-film leak-safe backtest. The classifier predicts these three tiers; each has its own "
        "dollar regressor, and the meta-combiner reconciles the probabilities and points into one number."
    )

with tab_clf:
    st.markdown(
        "**A single 3-class tier classifier — no cascade, no horror gate.** Two diverse learners vote:"
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "**CatBoost** (gradient-boosted trees)\n\n"
            "- Tuned via the V27 hyperparameter sweep\n"
            "- Inputs: `STATIC_WIKI` features + one OOF-stack point\n"
            "- Captures sharp threshold interactions"
        )
    with c2:
        st.markdown(
            "**TabPFN** (pretrained transformer)\n\n"
            "- Foundation model for small-table classification\n"
            "- Runs locally (torch); no per-film training\n"
            "- Diverse errors vs. trees → better borderline calls"
        )
    st.code(
        "probs = (catboost.predict_proba(X) + tabpfn.predict_proba(X)) / 2   # soft-vote average\n"
        "# member agreement ~87.5%, corr ~0.95 — diversity is what lifts borderline tiers",
        language="python",
    )
    st.caption("Soft-vote averaging beat every single-model and 3-member variant tested in the V27/V28 sweeps.")

with tab_regs:
    st.markdown(
        "Each tier has a dedicated CatBoost **dollar regressor** (trained on that tier's films) that "
        "produces a per-tier point estimate. These are unchanged from the tuned V25/V27 base."
    )
    rc1, rc2, rc3 = st.columns(3)
    for col, tier, body, note in [
        (rc1, "SMALL", "iterations=600\ndepth=5\nlearning_rate=0.02\nl2_leaf_reg=5\nloss='MAE'",
         "MAE for many low-variance films"),
        (rc2, "MID", "iterations=800\ndepth=6\nlearning_rate=0.015\nl2_leaf_reg=5\nloss='RMSE'",
         "Balanced loss across the widest band"),
        (rc3, "LARGE+", "iterations=500\ndepth=5\nlearning_rate=0.02\nl2_leaf_reg=8\nloss='Quantile:α=0.5'",
         "Quantile loss handles blockbuster variance"),
    ]:
        with col:
            st.markdown(f"**{tier}**")
            st.code(body, language="python")
            st.caption(note)
    st.caption("A missing tier regressor falls back to a global stacked-regressor point, so every film always has 3 points.")

with tab_meta:
    st.markdown(
        "**The headline of V28-A.** Instead of hand-coded rules deciding when to trust which tier, a "
        "small regressor **learns how to combine** the base outputs into a single dollar prediction."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "**Meta-features (7)** fed to g:\n"
            "- `log(point_SMALL, point_MID, point_LARGE+)` (3)\n"
            "- `prob_SMALL, prob_MID, prob_LARGE+` (3)\n"
            "- `log(soft mixture)` (1)\n\n"
            "**Target:** `log(OW)`."
        )
    with c2:
        st.code(
            "G_PARAMS = dict(\n"
            "    iterations=400, depth=3,\n"
            "    learning_rate=0.03, l2_leaf_reg=6,\n"
            "    loss_function='MAE',\n"
            ")\n\n"
            "FINAL = 0.7 * exp(g) + 0.3 * mixture",
            language="python",
        )
    st.success(
        "**Why it works:** the mixture-mean hedges across far-apart tier points when the classifier is "
        "uncertain. g learns when to commit to a tier vs. hedge — the data-driven analog of the old "
        "C / D / G demand rules, with zero hand-tuning."
    )

with tab_bands:
    st.markdown(
        "Every prediction ships with a **bear / base / bull** range, not just a point."
    )
    st.markdown(
        "1. Per-tier **expanded-pool quantile regressors** predict `q10` and `q90` (each tier trained on "
        "its own films plus a slice of neighbors).\n"
        "2. A **CQR conformal** widening constant (`Qlog`), calibrated on a held-out **20%** slice "
        "(`CAL_FRAC = 0.20`), corrects the quantiles so coverage holds out-of-sample.\n"
        "3. The tier quantiles are sampled by the tier probabilities and recentred on FINAL → "
        "**bear (low) / base (FINAL) / bull (high)**."
    )
    st.caption("Band coverage ≈ 85% in backtest. See the Recent Predictions page for live ranges.")

with tab_breakout:
    st.markdown(
        "V28-A reports a **calibrated probability of a breakout** (opening ≥ $50M / LARGE+), because for "
        "the biggest films the point estimate sits at a measured noise floor — the odds are more honest "
        "than a single number."
    )
    st.markdown(
        """
        | P(LARGE+) bucket | Wording | Actual LARGE+ rate |
        |---|---|---|
        | < 15% | (quiet) | **1%** |
        | 15–30% | "~1 in 5" | **17%** |
        | 30–50% | "~1 in 3" | **39%** |
        | ≥ 50% | "better than even" | **87%** |
        """
    )
    st.code(
        "breakout_prob = P(LARGE+)\n"
        "breakout_flag = (P(LARGE+) >= 0.30) and (point_tier < LARGE+)   # upside the point call misses",
        language="python",
    )
    st.caption("The flag is the Marketing / P&A 'lean in' trigger: real upside above the base call.")

with tab_leak:
    st.markdown(
        "**Nested stacking** keeps the learned combiner honest — every test film is scored by base "
        "models *and* a g that never saw it during training."
    )
    st.markdown(
        "- **Outer:** 5-fold `GroupKFold(MOVIE_ID)` → the reported out-of-fold predictions.\n"
        "- **Inner:** within each outer-train fold, a second 5-fold `GroupKFold` produces OOF base "
        "predictions for every outer-train film — **g is fit on those**, never on its own in-sample base preds.\n"
        "- Base models are then refit on the full outer-train fold and applied to the outer-test fold; "
        "g is applied there.\n"
        "- Grouping by `MOVIE_ID` prevents a film's multiple horizon rows from leaking across folds."
    )
    st.caption("This is why the backtest numbers are trustworthy: no film grades its own homework, at any layer.")

with tab_features:
    st.caption("Base feature universe (~72) feeding the classifier/regressors, plus the OOF-stack point and 7 learned meta-features:")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Static (36)**")
        st.markdown(
            "- YouTube + sentiment (7)\n"
            "- Movie attributes (8) — budget, runtime, TMDB pop, month, peak, predecessor OW, `IS_MAJOR_STUDIO`\n"
            "- Star power (4)\n"
            "- Genre (5) · Rating (4) · IP/Franchise (5)\n"
            "- TMDB daily (3) — D14, D7, momentum"
        )
    with c2:
        st.markdown("**Google Trends (~23)**")
        st.markdown(
            "- Rolling 3D / 5D / 7D / 14D / 21D + priors\n"
            "- Velocity 3D / 5D / 7D · VEL_3v7 / VEL_7v14 / LOG_SLOPE\n"
            "- Cumulative, volatility, peak, days_with_data, earliest\n"
            "- Interactions — Trends × IP / Genre / Star / Intent"
        )
    with c3:
        st.markdown("**Wikipedia (13)**")
        st.markdown(
            "- Rolling pageviews 3D / 7D / 14D + priors\n"
            "- Velocity (7D / prior 7D)\n"
            "- Peak day, cumulative, anchor day\n"
            "- Log-transformed versions\n\n"
            "→ corr(14D, OW) = **0.749**"
        )
    st.caption(
        "Plus: 1 OOF-stack point (a stacked regressor's log-OW) into the classifier, and the 7 meta-features "
        "into g. The velocity features remain as model inputs — but no longer trigger any hand-coded rule."
    )

show_cortex_badge()
