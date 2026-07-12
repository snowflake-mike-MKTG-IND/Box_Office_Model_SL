"""V30 Re-architecture Story — the classifier→distributional shift, done in ~2 days."""
import streamlit as st

from theme import SF_BLUE, VIOLET, apply_page_config, page_header, section, kpi_row, show_cortex_badge

apply_page_config("V30 · Re-architecture Story", icon="⚡")
page_header(
    "The V30 Re-architecture",
    "From a CatBoost classifier to a distributional, pedigree-gated ensemble — in about two days",
)

section("What actually changed")
st.markdown(
    "For 28 versions the model was fundamentally a **classifier**: assign a film to SMALL / MID / LARGE+, then "
    "run a per-tier regressor. V30 abandons that framing entirely. It is a **distributional regressor** — a "
    "CatBoost + Linear blend that emits a full predictive distribution, from which we read a best-estimate point, "
    "a risk-adjusted point, and an honest uncertainty band. Pedigree features are **gated** behind demand so the "
    "model stops over-predicting hype-flops, and the whole thing is validated on a **true future holdout**."
)

section("Two days, end to end")
kpi_row([
    ("Elapsed", "~2 days", "active working hours"),
    ("Experiments", "150+", "ablations + feature tests"),
    ("Models compared", "5 classifiers", "for the recall ceiling"),
    ("Deliverables", "model + skills + 2 apps + paper", "shipped"),
])
st.markdown(
    "In that window, working with Cortex Code, we: built the temporal rolling-origin CV harness; ran the "
    "interaction-feature study; discovered and validated pedigree-gating; built the predictive-distribution + "
    "HDR / Bayes-quantile output layer; exhaustively closed the tier-routing avenue (five model architectures); "
    "ran the true 2025→2026 holdout against v28b; registered V30 to the Snowflake Model Registry; updated the "
    "box-office skill fleet; and rebuilt this dashboard."
)

section("What this would cost in a legacy workflow (illustrative)")
st.markdown(
    "| Phase | Traditional DS team | V30 with Cortex Code |\n"
    "|---|---|---|\n"
    "| Temporal CV harness + leakage audit | ~1 week | hours |\n"
    "| Feature engineering + ablation study (150+) | ~2–3 weeks | ~1 day |\n"
    "| Distributional / conformal output layer | ~1–2 weeks | hours |\n"
    "| Routing experiments (5 architectures) | ~1–2 weeks | ~1 day |\n"
    "| Holdout validation + write-up | ~1 week | hours |\n"
    "| Registry deployment + app updates | ~1 week | hours |\n"
    "| **Total** | **~6–10 weeks, 2–3 people** | **~2 days, 1 person + AI** |\n"
)
st.caption("Estimates are illustrative of a typical hand-coded ML workflow, not a measured benchmark. The point is order-of-magnitude compression of the iteration loop.")

section("Why the new framing is better")
st.markdown(
    "- **Decision-aligned.** Greenlight and marketing-spend decisions are asymmetric — over-predicting a flop is "
    "far costlier than under-predicting a hit. A classifier tier can't express that; a distribution + a "
    "Bayes-optimal quantile can.\n"
    "- **Honest uncertainty.** A wide 50% HDR on an ambiguous tentpole is *information*, not a failure.\n"
    "- **Flop-safe by construction.** Pedigree-gating cut flop over-prediction from 24% → 5% on the true holdout.\n"
    "- **No industry-critical data.** No tracking, ticketing, Rotten Tomatoes, or theater count — and TMDB "
    "excluded as leakage. The signal comes from public pre-release demand (search, Wikipedia, YouTube)."
)
show_cortex_badge()
