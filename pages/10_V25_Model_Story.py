"""Page 10: V25 Model Story — how demand signals replaced budget as the primary tier driver."""
from __future__ import annotations

import streamlit as st

from theme import TIER_COLORS, apply_page_config, kpi_row, page_header, section, show_cortex_badge

apply_page_config("V25 Model Story", icon="📡")
page_header(
    "V25 Model Story",
    "How Masters of the Universe exposed budget dominance — and how we made demand the primary classifier signal.",
)

kpi_row(
    [
        ("Problem film", "Masters of the Universe", "$200M budget, R7D=58.8"),
        ("V24 prediction", "$62.9M (LARGE+)", "Budget-driven"),
        ("V25 prediction", "$46.8M (MID)", "Demand-driven"),
        ("CV improvement", "+1.1% accuracy", "at D-3 (75.6→76.7%)"),
    ]
)

st.caption(
    "Masters of the Universe (Amazon MGM, Jun 5, 2026). $200M budget, IP tier 3, Jared Leto. "
    "V24 automatically classified LARGE+ because the classifier never saw demand signals. "
    "But R7D=58.8 at D-3 is weaker than EVERY $150M+ film that opened $50M+ in our training set."
)

# ---- The Problem ----
section("The Problem: Budget Dominance")

st.markdown("""
The V24 model used a **two-stage cascade architecture** where the Stage 1 classifier
(tier assignment) only saw `STATIC_WIKI` features — which included budget, IP tier, star power,
and YouTube comments, but **zero Google Trends features**.

This meant:
- A $200M film with ZERO search interest would still get classified LARGE+
- A $10M film with EXPLOSIVE search interest (like Backrooms) would still get classified SMALL
- Budget was effectively the tier-assignment feature, with everything else just noise

The demand signals (Google Trends rolling averages) only influenced the **regressors within a tier**,
never the tier assignment itself.
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Films Held Back by Budget")
    st.markdown("""
    | Film | Budget | R7D at D-3 | Actual OW |
    |------|--------|-----------|-----------|
    | Obsession | $14M | 407 (peak) | $17.2M |
    | Backrooms | $10M | 326 | TBD |
    | FNAF | $20M | 215 | $80.0M |
    | Scream 7 | $35M | 127 | $63.6M |
    """)

with col2:
    st.markdown("#### Films Boosted by Budget Alone")
    st.markdown("""
    | Film | Budget | R7D at D-3 | Actual OW |
    |------|--------|-----------|-----------|
    | Argylle | $200M | 147 | $17.4M |
    | Napoleon | $165M | 183 | $20.6M |
    | Aquaman 2 | $205M | 90 | $27.7M |
    | Red One | $250M | 147 | $32.1M |
    """)

st.error(
    "**Key insight**: R7D at D-3 is MORE correlated with opening weekend (r=0.676) than BUDGET (r=0.669). "
    "And it's LESS collinear with budget (r=0.41) than YouTube comments (r=0.63), making it the ideal "
    "independent demand signal."
)

# ---- The Fix ----
section("The Fix: Demand-Driven Classifier")

st.markdown("""
### 1. Google Trends Features Added to Stage 1 Classifier

Six demand features now visible to the tier classifier:
- `ROLLING_3D` — 3-day search momentum (r=0.687 with OW)
- `ROLLING_7D` — 7-day demand signal (r=0.676, low budget collinearity at 0.41)
- `ROLLING_14D` — 14-day baseline
- `TRENDS_PEAK_SO_FAR` — viral spike detection
- `VEL_3V7` — acceleration (is demand growing or fading?)
- `LOG_SLOPE_14_TO_3` — long-term trajectory

### 2. Shallower Trees (Depth 7 → 5)

Prevents CatBoost from memorizing "budget > $150M → LARGE+" splits.
Forces the model to build more generalizable rules that consider demand.

### 3. Demand-Deficit Discount

When the classifier says LARGE+ (>30% probability) but R7D is below the demand floor:
- D-3 floor: R7D < 80 → discount LARGE+ by 50%
- D-7 floor: R7D < 55 → discount LARGE+ by 50%
- D-14 floor: R7D < 30 → discount LARGE+ by 50%

This catches Masters of the Universe: R7D=58.8 < floor of 80.

### 4. Rule D Demand Gate

The "tentpole override" (Rule D) now requires `R7D >= 60` as confirmation.
No more automatic LARGE+ for high-budget films with weak search interest.

### 5. Rule C Dual-Signal Alignment

- **C-Static** (TMDB-driven): Now requires R7D >= 80 confirmation
- **C-Demand** (GT-driven): R7D >= 80 + YT >= 5000 → pushes to MID regressor (not LARGE+)
""")

# ---- CV Results ----
section("CV Results: V24 vs V25 (287 films, 5-fold)")

st.markdown("""
| Horizon | V24 Accuracy | V25 Accuracy | V24 MAE | V25 MAE | False SMALL |
|---------|-------------|-------------|---------|---------|-------------|
| D-14 | 77.0% | 77.0% | $10.28M | $10.28M | — |
| **D-7** | 76.7% | **77.4%** | $10.36M | **$9.88M** | 25 → **22** |
| **D-3** | 75.6% | **76.7%** | $10.82M | **$10.47M** | 21 → **20** |
""")

st.success(
    "**Better at every horizon where demand matters.** D-7 gains +0.7% accuracy and -$0.48M MAE. "
    "D-3 gains +1.1% accuracy and -$0.35M MAE. "
    "False SMALL misclassifications drop by 3 at D-7 (the films the old model missed because it couldn't see demand)."
)

# ---- Impact ----
section("Impact: Weekend 22 Predictions")

st.markdown("""
| Film | V24 Pred | V25 Pred | Why |
|------|----------|----------|-----|
| **Masters of Universe** | LARGE+ $62.9M | **MID $46.8M** | Demand-deficit: R7D=58.8 < floor 80 |
| **Scary Movie 6** | MID $16.6M | **MID $26.2M** | Rule C-Demand: R7D=106 confirms breakout |
| **Disclosure Day** | LARGE+ $74.2M | **MID $28.2M** | Classifier naturally assigns MID with demand visible |
| **The Furious** | LARGE+ $72.9M | **SMALL $10.4M** | Rule C no longer fires without GT confirmation |
""")

st.info(
    "The model now responds to what audiences are ACTUALLY searching for, "
    "not just what studios spent. A $200M film with weak demand gets the same skepticism "
    "as Argylle ($17M OW from $200M budget)."
)

show_cortex_badge()
