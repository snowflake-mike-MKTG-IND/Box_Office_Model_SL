"""V30 Architecture — pedigree-gated distributional ensemble."""
import streamlit as st

from theme import apply_page_config, page_header, section, show_cortex_badge

apply_page_config("V30 · Architecture", icon="🏗️")
page_header(
    "V30 Architecture",
    "Pedigree-gated CatBoost + Linear blend → predictive distribution → triple output + confidence flag",
)

section("The pipeline")
st.graphviz_chart(
    """
    digraph {
      rankdir=LR; node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11];
      edge [fontname="Helvetica" fontsize=9 color="#6B7280"];

      feats [label="Pre-release features\\n(NO tracking / RT / ticketing /\\ntheater count / TMDB)" fillcolor="#EFF6FF"];
      gate  [label="Pedigree GATING\\n12 standalone pedigree feats REMOVED\\npedigree re-enters via 8 demand\\ninteractions (demand × pedigree)" fillcolor="#F3E8FF"];
      cb    [label="CatBoost\\nln(OW), RMSE\\n(interpolation)" fillcolor="#E0F2FE"];
      lin   [label="Linear (ElasticNet)\\nln(OW)\\n(extrapolation)" fillcolor="#E0F2FE"];
      mix   [label="50/50 predictive MIXTURE\\ncenter + empirical OOF\\nlog-residual library" fillcolor="#FEF3C7"];
      hdr   [label="50% HDR interval" fillcolor="#EDE9FE"];
      hm    [label="HDR50_MEAN\\nbest-estimate point" fillcolor="#EDE9FE"];
      bz    [label="BAYES r2 point\\nτ = 1/(1+r) quantile\\n(risk-adjusted)" fillcolor="#EDE9FE"];
      rf    [label="Demand-forward ≥$50M flag\\n(calibrated; annotation only)" fillcolor="#DCFCE7"];

      feats -> gate;
      gate -> cb; gate -> lin;
      cb -> mix; lin -> mix;
      mix -> hdr; mix -> hm; mix -> bz;
      feats -> rf [style=dashed label="demand signals only\\n(no pedigree)"];
    }
    """
)

section("Two ideas do the work")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**1 · Pedigree gating — flop-safety**")
    st.markdown(
        "Standalone pedigree (budget, star ×4, IP ×5, predecessor, studio) is **removed** from the regressor. "
        "It only re-enters as **interactions with demand percentile** (`ROLLING_7D_PCTILE × {star, IP, action, "
        "predecessor, …}`). A film is only treated as big when **search/wiki/intent demand confirms the pedigree** "
        "— which is exactly what stops hype-flops (e.g. Supergirl) from being over-predicted."
    )
with c2:
    st.markdown("**2 · Distributional output — honest uncertainty**")
    st.markdown(
        "Rather than a single number, V30 builds a **predictive distribution**: each model's center plus its "
        "empirical out-of-fold log-residuals, mixed 50/50. From it we report the **50% highest-density region**, "
        "the density-weighted **HDR50_MEAN** best-estimate, and a **Bayes-optimal risk-adjusted point** that "
        "deliberately leans conservative when over-prediction is costly."
    )

section("The risk-adjusted point (math)")
st.markdown(
    "Under an asymmetric cost where over-predicting is *r×* worse than under-predicting,"
)
st.latex(r"L_r(p, a) = r\cdot\max\!\big(0,\ \ln\tfrac{p}{a}\big) + \max\!\big(0,\ \ln\tfrac{a}{p}\big)")
st.markdown(
    "the loss-minimizing point estimate is the **τ-quantile** of the predictive distribution with"
)
st.latex(r"\tau = \frac{1}{1+r}\qquad\Rightarrow\qquad r=2 \;\Rightarrow\; \tau = \tfrac{1}{3}\ \ (\text{P33})")
st.caption("This is why the published Bayes point sits below the median — it prices in that over-predicting a flop is the cardinal sin.")

section("What changed from V28-B")
st.markdown(
    "| | V28-B (prior) | V30 (current) |\n"
    "|---|---|---|\n"
    "| Core | 3-tier CatBoost **classifier** + per-tier regressors | CatBoost + Linear **distributional blend** |\n"
    "| Pedigree | standalone features | **gated** through demand interactions |\n"
    "| Output | single point + range-clip | **50% HDR + HDR50 + Bayes** point |\n"
    "| Large films | tier routing | **demand-forward flag** (annotation only) |\n"
    "| TMDB | used | **excluded** (holdout-confirmed leakage) |\n"
    "| Validated on | CV + backtest | **true 2025→2026 holdout** |\n"
)
show_cortex_badge()
