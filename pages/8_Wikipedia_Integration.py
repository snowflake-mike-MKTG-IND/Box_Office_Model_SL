"""Page 8: Development Story — Wikipedia sprint + Cortex Code velocity."""
import pandas as pd
import plotly.express as px
import streamlit as st

from theme import (
    apply_page_config,
    freshness_caption,
    kpi_row,
    page_header,
    section,
    show_cortex_badge,
)

apply_page_config("Development story", icon="📚")

page_header(
    "Development Story",
    "How V18 got built with Cortex Code — the 49-minute Wikipedia sprint and the velocity story around it.",
)

tab_sprint, tab_signal, tab_variants, tab_velocity = st.tabs(
    ["The 49-minute sprint", "Why Wikipedia works", "Variant study", "Overall velocity"]
)

# ---------------------------------------------------------------------------
with tab_sprint:
    st.markdown(
        "> *\"NRG Tracking, Ticket Sales, and Quorum Tracking are inaccessible for a demo. "
        "Does Wikipedia provide enough historical data to have full coverage over all our movies?\"*"
    )
    st.markdown(
        "**Answer**: yes. In 49 wall-clock minutes we ingested Wikipedia pageviews for 277 films, "
        "engineered 13 features across 3 horizons, tested 4 placement variants, ran 108 hyperparameter "
        "configurations, and confirmed **+3.6pp** CV accuracy with **-$0.39M** MAE — all with "
        "no vendor, no procurement, no waiting."
    )

    kpi_row([
        ("Wall-clock time", "~49 min", "idea → tuned model"),
        ("Movies ingested", "277", "100% coverage"),
        ("Pageview rows", "8,455", "daily granularity"),
        ("Total pageviews", "193.3M", "D-30 → D+0 per film"),
    ])
    freshness_caption("Wikimedia REST + MediaWiki Action APIs", "2026-04-20")

    phases = pd.DataFrame([
        {"Phase": "1. Title mapping (OpenSearch)",                   "Minutes": 3.1,  "Output": "275/277 mapped"},
        {"Phase": "2a. Pageview fetch (REST API)",                   "Minutes": 1.2,  "Output": "6,975 rows"},
        {"Phase": "2b. Canonical resolution via summary redirects",  "Minutes": 0.55, "Output": "+28 films"},
        {"Phase": "2c–f. URL encoding + year-suffix audit + manual", "Minutes": 11.0, "Output": "277/277 coverage"},
        {"Phase": "3. Feature engineering (SQL DDL)",                "Minutes": 0.02, "Output": "13 × 3 horizons"},
        {"Phase": "4. V18 baseline training (3 variants)",           "Minutes": 3.3,  "Output": "Placement confirmed"},
        {"Phase": "5. HPT across horizons (108 configs)",            "Minutes": 11.4, "Output": "Best S1/S2 HPs"},
    ])

    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.bar(phases, x="Minutes", y="Phase", orientation="h",
                     text="Minutes", color="Minutes", color_continuous_scale="Blues")
        fig.update_traces(texttemplate="%{text:.1f} min", textposition="outside")
        fig.update_layout(height=420, xaxis_range=[0, 14],
                          yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown(
            """
            **Pipeline**

            1. **Ingest** — Wikimedia REST + MediaWiki Action APIs @ ~50 req/s.
            2. **Resolve** — OpenSearch, summary-API redirects, year-suffix audit,
               manual overrides for unicode edge-cases (`MaXXXine`, `Thunderbolts*`).
            3. **Engineer** — rolling/velocity/peak + log transforms, keyed by
               release date, stored in `PRODUCTION.WIKIPEDIA_FEATURES_V`.
            """
        )

# ---------------------------------------------------------------------------
with tab_signal:
    c1, c2 = st.columns(2)
    with c1:
        corr = pd.DataFrame({
            "Signal": ["Wikipedia 14d views", "Google Trends 14d (typical)", "YT Comments"],
            "Correlation with OW": [0.749, 0.50, 0.55],
        })
        fig = px.bar(corr, x="Signal", y="Correlation with OW",
                     text="Correlation with OW", color="Correlation with OW",
                     color_continuous_scale="Blues")
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(yaxis_range=[0, 1], height=360)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown(
            """
            **Why Wikipedia beats Trends as a pre-release signal**

            - **Research-intent behavior** — users actively looking up upcoming films.
            - **Less noise** than search queries (no ambiguity around title terms).
            - **Global consistency** — one article across countries.
            - **Daily granularity** over 31 days captures the marketing lift.
            - **Free, public, documented API** — no vendor lock-in.
            """
        )

# ---------------------------------------------------------------------------
with tab_variants:
    st.caption("Fair 5-fold GroupKFold CV, 276 dedup films, held at V17.2 hyperparameters.")
    variants = pd.DataFrame({
        "Variant": ["A: V17.2 baseline", "B: Wiki in regressor only",
                    "C: Wiki in classifier only", "D: Wiki in both"],
        "D-14 Acc": [72.5, 72.5, 75.7, 75.7],
        "D-7 Acc":  [72.8, 72.8, 75.4, 75.4],
        "D-3 Acc":  [72.1, 72.1, 74.6, 74.6],
        "D-7 MAE ($M)": [11.67, 11.82, 11.02, 11.22],
    })
    st.dataframe(variants, use_container_width=True, hide_index=True)
    st.markdown(
        "Wikipedia features primarily help the **classifier** — they sharpen the SMALL/MID/LARGE+ "
        "boundary. The regressor's marginal benefit is small because log-transformed budget, star "
        "power, and Trends already capture most of the magnitude signal."
    )

    section("Retuned V18 classifier hyperparameters")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Stage 1 — SMALL vs NON-SMALL**")
        st.code(
            "iterations=200    # was 300 in V17.2\n"
            "depth=7\n"
            "learning_rate=0.02  # was 0.03",
            language="python",
        )
    with c2:
        st.markdown("**Stage 2 — MID vs LARGE+**")
        st.code(
            "iterations=400    # was 300\n"
            "depth=5\n"
            "learning_rate=0.03  # was 0.02\n"
            "l2_leaf_reg=5, subsample=0.8\n"
            "colsample_bylevel=0.9",
            language="python",
        )

# ---------------------------------------------------------------------------
with tab_velocity:
    kpi_row([
        ("Calendar span", "83 days", "Jan 28 → Apr 21"),
        ("Active sessions", "10", "~41h active work"),
        ("Snowflake artifacts", "664+", "tables + views + procs"),
        ("ML experiments", "79", "+108 HP configs"),
    ])
    st.markdown(
        "One person, one AI, a handful of part-time sessions. Cortex Code handled SQL generation, "
        "pipeline debugging, feature discovery, architecture pivots (4-tier → 3-tier), HP tuning, "
        "and this dashboard itself."
    )

    section("Signature sprints")
    st.markdown(
        "- **Feb 15** — V2 → V10 feature engineering marathon in a single 8-hour session.\n"
        "- **Feb 27–28** — 3-tier cascade redesign + hybrid star power + first production deploy.\n"
        "- **Apr 10** — V16 TMDB override: signal discovery → 5-way holdout test → SP + "
        "dashboard update in ~3 hours.\n"
        "- **Apr 20** — V17 extended Trends features end-to-end in one session.\n"
        "- **Apr 20 (evening)** — **V18 Wikipedia sprint, 49 minutes**.\n"
        "- **Apr 21** — full data-integrity pass against The-Numbers.com (276 films)."
    )

    section("Bottom line")
    st.markdown(
        "**Snowflake + Cortex Code lets data-science teams iterate at the speed of curiosity.** "
        "A customer asked 'would Wikipedia help?' at 7:00 PM. By 7:52 PM the question was answered "
        "with a quantified, production-ready model improvement."
    )

show_cortex_badge()
