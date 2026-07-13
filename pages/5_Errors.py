"""V30 Errors — the Wikipedia data bug, the pedigree trap, and the demand-forward fix."""
import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from theme import SF_BLUE, VIOLET, apply_page_config, page_header, section, show_cortex_badge

apply_page_config("V30 · Errors", icon="🔎")
page_header(
    "Where V30 Misses — a Data Bug and a Pedigree Trap, Now Fixed",
)


@st.cache_data
def _preds():
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "cv_predictions_v30.json")) as f:
        return json.load(f)


P = _preds()
df = pd.DataFrame(P)
df["signed_err_m"] = df["predicted_ow_m"] - df["actual_ow_m"]
df["ape"] = (df["signed_err_m"].abs() / df["actual_ow_m"].clip(lower=1e-6) * 100)

section("The error profile is deliberately asymmetric")
c1, c2, c3 = st.columns(3)
over = df[df["predicted_ow_m"] > 1.5 * df["actual_ow_m"]]
low = df[df["actual_ow_m"] < 60]
c1.metric("Median APE", f"{df['ape'].median():.0f}%")
c2.metric("Flop over-predictions (<$60M, >1.5×)", f"{(len(over[over['actual_ow_m']<60]) / max(len(low),1) * 100):.0f}%")
c3.metric("Under vs over (all films)", f"{(df['signed_err_m']<0).mean()*100:.0f}% under")
st.caption(
    "V30 is tuned so that when it errs, it errs **low** — under-predicting is cheap, over-predicting a flop is the "
    "cardinal sin. The distribution of signed errors leans negative by design."
)

section("What looked like 'demand-quiet giants' was a data bug")
big_miss = df[(df["actual_ow_m"] >= 60) & (df["predicted_ow_m"] < df["actual_ow_m"] * 0.7)].sort_values("actual_ow_m", ascending=False)
st.markdown(
    "We originally believed the big films V30 under-shot simply had **quiet pre-release demand**. A July 2026 "
    "data-integrity audit proved that wrong. **13 of the largest films — including Backrooms, The Mandalorian & "
    "Grogu, Tron: Ares, M3GAN 2.0, Argylle — had their Wikipedia pageviews keyed to the wrong article** "
    "(a redirect, a spurious `(film)` disambiguator, or an entirely different movie). Because the Wikimedia "
    "pageviews API does not follow redirects, the model saw **~0 Wikipedia demand** for films whose true demand "
    "was **top-decile**. Backrooms' real pre-release Wikipedia peak was ~66K/day (recorded as 2). Once corrected, "
    "these films' Wikipedia demand percentile jumped from bottom-decile to 0.6–0.9. The 'signal ceiling' was "
    "substantially a **data bug**, not a modeling limit."
)
if len(big_miss):
    show = big_miss.head(10)[["movie_title", "actual_ow_m", "predicted_ow_m", "bayes_ow_m", "p_large"]].rename(
        columns={"movie_title": "Film", "actual_ow_m": "Actual $M", "predicted_ow_m": "HDR50 $M",
                 "bayes_ow_m": "Bayes $M", "p_large": "P(large)"})
    st.dataframe(show, use_container_width=True, hide_index=True)

section("The second issue: a pedigree-laden flag — fixed with a demand-forward flag")
st.markdown(
    "Even after the data fix, V30's original **≥$50M confidence flag leaned on static pedigree** (budget, star "
    "power, IP). On the 2026 holdout it caught **0%** of the large films — because 2026's giants were **low-pedigree "
    "originals** (Backrooms, Project Hail Mary). We rebuilt the flag to be **demand-forward** (Google Trends, "
    "Wikipedia, YouTube — no pedigree). On the same holdout it now catches **50% of ≥$50M films at 100% precision "
    "with zero flop false-positives**. This demand-forward flag ships in **V30** (deployed to the Model Registry)."
)

section("The fix: a demand-QUALITY gate that lifts confident large films flop-safely")
st.markdown(
    "The breakthrough came from separating demand **volume** from demand **quality**. The event films that were "
    "under-predicted (Michael, Project Hail Mary, Mandalorian) have **positive/neutral net audience intent**, while the "
    "look-alike high-demand flops that made the model conservative (Snow White −25, Gladiator II −13, Furiosa, The "
    "Marvels) have **strongly negative intent** — high curiosity, low ticket intent. Volume alone can't tell them "
    "apart; quality can.\n"
    "- **Demand-quality gate (+Q):** net-intent-gated demand features were added to the regressor. On the 245-film "
    "out-of-fold set this **held the flop over-prediction rate flat** while improving large-film calibration — the "
    "first lever to do so after four prior attempts (ungated expert, composite gate, explicit interactions, specialist "
    "routing) all failed flop-safety.\n"
    "- **Track B — confidence-conditional point:** for films the demand-forward flag is confident are large (P ≥ 0.4), "
    "the point is lifted from the flop-safe centre toward an elevated quantile of the predictive distribution. This "
    "raises the true event films (Michael, PHM, Mandalorian, and tentpoles like Spider-Man) **without touching the "
    "negative-intent flops**, which never trip the flag.\n"
    "- **Track C — demand-implied upside:** every flagged film also reports a P78 'demand-implied ceiling', the honest "
    "high end of a wide, right-skewed distribution.\n"
    "- **The residual honest limit:** films whose demand genuinely matches a mixed population (an original horror like "
    "Backrooms) keep a conservative point with a high upside band — the model won't over-commit where the outcome is "
    "genuinely bimodal, and that is the correct behaviour under a flop-averse loss."
)
show_cortex_badge()
