"""V30 Opening Weekend Prediction Model — Home / navigation hub."""
import json
import os

import streamlit as st

from theme import (
    SF_BLUE,
    apply_page_config,
    freshness_caption,
    kpi_row,
    page_header,
    section,
    show_cortex_badge,
)

apply_page_config("V30 OW Prediction Model", icon="🎬")


@st.cache_data
def _perf():
    p = os.path.join(os.path.dirname(__file__), "data", "performance_v30.json")
    with open(p) as f:
        return json.load(f)


PERF = _perf()
OOF = PERF["temporal_oof"]
HO = PERF["holdout_2026"]

page_header(
    "V30 Opening Weekend Prediction Model",
    "Pedigree-gated distributional ensemble · CatBoost + Linear blend → 50% HDR / HDR50 / Bayes-risk points · demand-forward ≥$50M flag · Snowflake Model Registry",
)

# -- Cortex Code velocity hero ----------------------------------------------
st.markdown(
    f"""
    <div style="background: linear-gradient(135deg, {SF_BLUE} 0%, #11567F 100%);
                padding: 1.75rem 2rem; border-radius: 14px; color: white;
                margin-bottom: 1.5rem;">
      <div style="font-size: 0.82rem; letter-spacing: 0.08em; text-transform: uppercase;
                  opacity: 0.85; margin-bottom: 0.35rem;">
        ❄️ Built with Cortex Code
      </div>
      <div style="font-size: 1.55rem; font-weight: 700; line-height: 1.25;
                  margin-bottom: 0.4rem;">
        An entire ML re-architecture in ~2 days.
      </div>
      <div style="font-size: 0.98rem; opacity: 0.95; line-height: 1.45; max-width: 880px;">
        V30 replaces the CatBoost 3-tier <b>classifier + per-tier regressors</b> with a
        <b>pedigree-gated distributional ensemble</b> — a CatBoost + Linear blend that emits a full
        predictive distribution (50% highest-density region, a density-weighted point, and a
        risk-adjusted Bayes point) plus a calibrated large-film confidence flag. One person + one AI
        designed, ran <b>150+ ablations</b>, validated on a true future holdout, registered the model,
        updated the skill fleet, and shipped this dashboard — in about <b>two days of working hours</b>.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

kpi_row([
    ("Re-architecture", "~2 days", "vs ~6–10 wks traditional DS"),
    ("Approach shift", "classifier → distributional", "regression + uncertainty"),
    ("Team size", "1 + AI", "vs 2–3 data scientists"),
    ("Validation", "true 2025→2026 holdout", "leakage-audited"),
])
st.caption(
    "Illustrative velocity: a distributional re-architecture + 150+ ablations + temporal-holdout harness "
    "+ registry deployment would traditionally be a **multi-week** effort for a small DS team hand-coding "
    "tests, CV, and MLOps. No procurement, no handoffs, no vendor integration."
)

# -- Model performance hero --------------------------------------------------
section("Performance — true future holdout (train ≤2025 → predict 2026)")
kpi_row([
    ("V30 MAPE", f"{HO['v30']['mape_pct']:.0f}%", f"v28b {HO['v28b_asauthored']['mape_pct']:.0f}%"),
    ("Asymmetric loss (r=2)", f"{HO['v30']['aloss_r2']:.3f}", f"v28b {HO['v28b_asauthored']['aloss_r2']:.3f}"),
    ("Flop over-prediction", f"{HO['v30']['lowband_over_pct']:.0f}%", f"v28b {HO['v28b_asauthored']['lowband_over_pct']:.0f}%"),
    ("50% HDR coverage", f"{OOF['hdr50_coverage_pct']:.0f}%", "target 50% (calibrated)"),
])
freshness_caption(
    "8-block quarterly rolling-origin temporal CV + true 2025→2026 holdout (26 films, small-sample → wide CIs). "
    "Registered as SPARK_PAR_DEMO.ML_PIPELINE.OW_PREDICTION_V30.",
    "2026-07-11",
)
st.info(
    "**Accuracy without industry-critical datasets.** V30 uses NO box-office tracking, NO pre-sales/ticketing, "
    "NO Rotten Tomatoes, and NO theater count. TMDB popularity is **excluded** — the holdout confirmed it is "
    "time-inconsistent leakage that *hurt* the prior model on real future data.",
    icon="🔒",
)

# -- Navigation grid ---------------------------------------------------------
section("Explore V30", "Each page owns one topic.")

NAV = [
    ("Architecture", "Pedigree-gating → CatBoost+Linear blend → predictive mixture → 50% HDR / HDR50 / Bayes point + demand-forward flag.",
     "pages/1_Architecture.py"),
    ("Features", "V30 feature stack-ranking — pedigree enters only through demand interactions.",
     "pages/2_Features.py"),
    ("Performance", "Temporal holdout + generational comparison V16 → V30.",
     "pages/3_Performance.py"),
    ("Predict", "The triple output explained, with upcoming releases (Spider-Man, Super Troopers 3, One Night Only).",
     "pages/4_Predictions.py"),
    ("Errors", "A Wikipedia data bug and a pedigree trap — found, fixed, and the demand-forward flag that resulted.",
     "pages/5_Errors.py"),
    ("V30 Re-architecture Story", "How the classifier→distributional shift was designed and validated in ~2 days.",
     "pages/12_V30_Rearchitecture.py"),
]

cols = st.columns(3)
for i, (title, desc, page) in enumerate(NAV):
    with cols[i % 3]:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(desc)
            st.page_link(page, label="Open →")

# -- Model history / legacy --------------------------------------------------
section("Model History / Legacy", "Prior production generations — preserved for historical prediction wins.")
LEGACY = [
    ("Version Timeline", "Version-by-version evolution and HPT.", "pages/6_Timeline.py"),
    ("Recent Predictions", "Historical tracking of predictions vs actual weekends.", "pages/7_Recent_Predictions.py"),
    ("V28-A Story", "Rule-free learned meta-combiner (prior production).", "pages/11_V28A_Model_Story.py"),
    ("V25 Story", "Demand-driven classifier.", "pages/10_V25_Model_Story.py"),
    ("V24 Story", "Cascade + routing.", "pages/9_V24_Model_Story.py"),
    ("Wikipedia Sprint", "The 49-minute Wikipedia feature sprint.", "pages/8_Wikipedia_Integration.py"),
]
lcols = st.columns(3)
for i, (title, desc, page) in enumerate(LEGACY):
    with lcols[i % 3]:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(desc)
            st.page_link(page, label="Open →")

# -- What's new --------------------------------------------------------------
section("What's new in V30")
st.markdown(
    "- **Classifier → distributional ensemble.** The 3-tier CatBoost classifier + per-tier regressors is "
    "replaced by a CatBoost + Linear ½/½ blend on ln(OW) that produces a full predictive distribution.\n"
    "- **Pedigree gating.** The 12 standalone pedigree features (budget, star, IP, predecessor, studio) are "
    "removed; pedigree re-enters only through 8 demand-gated interactions. \"Demand must confirm pedigree\" — "
    "this is what suppresses hype-flop over-prediction (e.g. Supergirl).\n"
    "- **Triple output.** 50% highest-density region (HDR) + HDR50_MEAN best-estimate + a Bayes-optimal "
    "risk-adjusted point (τ = 1/(1+r) quantile under an asymmetric cost).\n"
    "- **Demand-forward confidence flag (V30).** A calibrated ≥$50M flag built on demand signals only "
    "(Google Trends, Wikipedia, YouTube — no pedigree): 50% recall / 100% precision / 0 flop false-positives on "
    "the 2026 holdout, vs 0% for the earlier pedigree-based flag — annotation only, never routes the point.\n"
    "- **Validated on true future data.** Train ≤2025 → predict 2026: aLoss "
    f"{HO['v30']['aloss_r2']:.3f} vs v28b {HO['v28b_asauthored']['aloss_r2']:.3f}; flop over-prediction "
    f"{HO['v30']['lowband_over_pct']:.0f}% vs {HO['v28b_asauthored']['lowband_over_pct']:.0f}%.\n"
)

section("Previously")
st.markdown(
    "- **V28-B** — horizon-normalized demand classification, CatBoost multiclass + per-tier regressors, range-clip.\n"
    "- **V28-A** — rule-free learned meta-combiner (CatBoost + TabPFN soft-vote).\n"
    "- **V25** — demand-driven classifier; Google Trends moved into tier assignment.\n"
    "- **V22c** — 2-stage cascade + hybrid Rule C/D overrides.\n"
    "- **V18** — +13 Wikipedia pageview features lifted CV accuracy to 77.2%.\n"
)

show_cortex_badge()
