"""Page 11: V28-A Model Story — how we removed every hand-coded rule and replaced them with a
learned meta-combiner, then reframed the unpredictable top tier as calibrated breakout odds."""
from __future__ import annotations

import streamlit as st

from theme import TIER_COLORS, apply_page_config, kpi_row, page_header, section, show_cortex_badge

apply_page_config("V28-A Model Story", icon="🎯")
page_header(
    "V28-A Model Story",
    "How a rule-free learned meta-combiner replaced the hand-coded rule stack — and why the top tier "
    "is now reported as calibrated breakout odds instead of a hero number.",
)

kpi_row(
    [
        ("Hand-coded rules", "0", "down from 5 (C/D/E/F/G)"),
        ("CV accuracy", "77.7%", "same-basis, 287 films"),
        ("CV MAE", "$9.99M", "rule-free @ -7d"),
        ("Breakout calibration", ">50% → 87%", "P(LARGE+) bucket hit-rate"),
    ]
)

st.caption(
    "V25 made the classifier demand-driven but still leaned on a stack of hand-coded rules (demand gates, "
    "tentpole overrides, dual-signal alignment). V28-A removes them entirely: a small learned combiner does "
    "the job the rules were approximating, and the unpredictable top tier is surfaced as a probability, not a point."
)

# ---- The Problem ----
section("The Problem: Rules Don't Generalize, and Breakouts Sit at a Noise Floor")

st.markdown("""
Two separate problems pushed us past the rule era:

**1. Hand-coded rules are hindsight-fit.** By V25–V27 the model carried five overrides — demand-deficit
discounts, the Rule D tentpole gate, Rule C dual-signal alignment, and more. Each was tuned to fix a
specific film we'd already seen (Masters of the Universe, MK2, …). That makes the backtest look good but
is exactly the kind of thing that breaks on the *next* film, because the thresholds encode our hindsight,
not a generalizable signal.

**2. The top tier has a noise floor.** Two films with near-identical pre-release demand routinely open
$50M+ apart. No feature available at -7 days resolves that gap. So past a point, pushing for a sharper
LARGE+ dollar estimate isn't modeling — it's overfitting to luck.
""")

st.error(
    "**Key insight**: a rule that fires on a threshold is a frozen guess. The same decision — *should this "
    "film be lifted or discounted?* — can be **learned** from the base model's own outputs, where it adapts "
    "per film instead of tripping a fixed wire."
)

# ---- The Fix ----
section("The Fix: A Learned Meta-Combiner + Calibrated Breakout Odds")

st.markdown("""
### 1. Base layer (unchanged in spirit)

A soft-voting **3-class tier classifier** (CatBoost + TabPFN, probabilities averaged) over the static +
Wikipedia feature set plus one stacked OOF point, then **three per-tier $ regressors**. Their
probability-weighted blend is the **soft mixture** `Σ prob · point`.

### 2. The rule stack → one learned combiner `g`

Every hand-coded rule is replaced by a single small regressor that reads the base layer and outputs log-OW:

- `g` = CatBoost(iterations=400, depth=3, lr=0.03, l2_leaf_reg=6, **loss=MAE**)
- over **7 meta-features**: log of each tier's $ point (×3), the three class probabilities (×3), and the
  log soft mixture (×1)
- **Final OW = 0.7 · exp(g) + 0.3 · mixture**

The combiner's own importances show it leaning on the **mixture (≈29%)** and the **class-probability
distribution (≈15% each)** above any single tier point — it trusts the demand-driven distribution, which
is what the old rules were clumsily trying to encode.

### 3. Leakage-safe nested stacking

`g` is fit on **inner-OOF** base predictions inside each outer fold, so it never sees a film whose base
predictions trained it. Outer 5-fold GroupKFold on `MOVIE_ID`; base refit on full outer-train, applied to
outer-test.

### 4. The top tier becomes odds, not a point

Instead of a rule lifting tentpoles, V28-A reports a **calibrated P(LARGE+)** plus bear/base/bull bands
(expanded-pool quantile regressors widened by a CQR conformal constant). The probability buckets are
validated against actual outcomes.
""")

# ---- CV Results ----
section("Results: Same Accuracy, Zero Rules")

st.markdown("""
| Metric | V25 (rules) | V28-A (rule-free) |
|--------|-------------|-------------------|
| CV accuracy (same-basis, 287) | 77.4% | **77.7%** |
| CV MAE (same-basis, -7d) | $9.88M | **$9.99M** |
| Hand-coded rules | 5 | **0** |
| Leak-safe backtest (288 films) | — | **75.3% / $10.96M** |

**Per-tier backtest MAE:** SMALL **$3.85M** (n=148), MID **$8.60M** (n=88), LARGE+ **$35.16M** (n=52).
""")

st.success(
    "**The point of V28-A is not a higher number — it's the same accuracy with no hand-coded rules.** "
    "We traded five brittle overrides for one combiner that generalizes, and stopped pretending the LARGE+ "
    "dollar point is knowable. The $35M top-tier MAE is the honest noise floor, not a defect."
)

st.markdown("""
#### Breakout odds are calibrated

| P(LARGE+) bucket | Reported as | Actual LARGE+ rate |
|---|---|---|
| < 15% | unlikely | ~1% |
| 15–30% | ~1 in 5 | ~17% |
| 30–50% | ~1 in 3 **(FLAG)** | ~39% |
| > 50% | better than even | ~87% |

A film is **flagged** when P(LARGE+) ≥ 0.30 even if its point estimate lands in a lower tier — the case
where the dollar figure under-reads a real breakout risk.
""")

# ---- Impact ----
section("Impact: Upcoming Releases")

st.markdown("""
| Film | Horizon | V28-A point | P(LARGE+) | Read |
|------|---------|-------------|-----------|------|
| **Disclosure Day** | D-7 | MID **$35.6M** | **32%** | **Breakout watch** — point says MID, but ~1-in-3 odds of LARGE+ → flagged |
| The Furious | D-7 | SMALL $9.1M | 3% | Clean SMALL, no breakout signal |
| The Death of Robin Hood | D-14 | SMALL $6.3M | 3% | Clean SMALL at the wider horizon |
| Girls Like Girls | D-14 | SMALL $3.5M | 1% | Clean SMALL |
""")

st.info(
    "Disclosure Day is the case V28-A is built for: the point estimate is MID, but the model honestly says "
    "there's roughly a 1-in-3 chance it opens LARGE+. The old rule stack would have given a single hard "
    "answer; V28-A gives the user the actual decision — *watch this one* — with a calibrated probability behind it."
)

show_cortex_badge()
