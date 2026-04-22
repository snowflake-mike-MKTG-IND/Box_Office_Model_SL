"""Shared theme and layout helpers for the V18 Box Office Model dashboard."""
from __future__ import annotations

import streamlit as st

# Snowflake brand palette
SF_BLUE = "#29B5E8"
DK1 = "#262626"
DK2 = "#11567F"
TEAL = "#71D3DC"
ORANGE = "#FF9F36"
VIOLET = "#7D44CF"
MUTED = "#6B7280"

# Single authoritative tier palette used across every chart
TIER_COLORS = {
    "SMALL": SF_BLUE,
    "MID": ORANGE,
    "LARGE+": VIOLET,
}

# Version color for trend charts
VERSION_COLOR = DK2

APP_CSS = f"""
<style>
  /* Tighten default Streamlit spacing and make sections feel quieter */
  .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }}
  h1, h2, h3 {{ color: {DK1}; letter-spacing: -0.01em; }}
  h1 {{ font-weight: 700; }}
  h2 {{ font-weight: 600; margin-top: 1.5rem; }}
  h3 {{ font-weight: 600; color: {DK2}; }}
  /* Make st.metric values bolder and a bit larger */
  div[data-testid="stMetricValue"] {{ font-weight: 700; }}
  /* Remove the tiny red/green arrow-only delta padding quirk */
  div[data-testid="stMetricDelta"] {{ font-size: 0.85rem; }}
  /* Section spacer replaces heavy dividers */
  .section-spacer {{ height: 1.75rem; }}
  /* Navigation card style for the home page */
  .nav-card {{
      border: 1px solid #E5E7EB;
      border-radius: 12px;
      padding: 1rem 1.1rem;
      background: #FAFBFC;
      height: 100%;
  }}
  .nav-card h4 {{ margin: 0 0 0.35rem 0; color: {DK2}; font-size: 1rem; }}
  .nav-card p {{ margin: 0; color: {MUTED}; font-size: 0.85rem; line-height: 1.35; }}
</style>
"""


def apply_page_config(title: str, icon: str = "🎬") -> None:
    """Standard page config + inject app CSS."""
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str | None = None) -> None:
    """Consistent page title + subtitle. No dividers or emoji in the title."""
    st.title(title)
    if subtitle:
        st.caption(subtitle)
    st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)


def section(title: str, caption: str | None = None) -> None:
    """Lightweight section break. Replaces st.header + st.divider pairs."""
    st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)
    st.subheader(title)
    if caption:
        st.caption(caption)


def kpi_row(metrics: list[tuple[str, str, str | None]]) -> None:
    """Render a row of up to 4 metrics. Each metric = (label, value, delta)."""
    cols = st.columns(len(metrics))
    for col, (label, value, delta) in zip(cols, metrics):
        if delta:
            col.metric(label, value, delta)
        else:
            col.metric(label, value)


def freshness_caption(source: str, updated: str) -> None:
    """Small caption under KPI rows so stale content is obvious."""
    st.caption(f"Source: {source} · Updated: {updated}")


def nav_card(title: str, description: str, page: str) -> None:
    """Clickable nav card for the home page. Uses st.page_link under the hood."""
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(description)
        st.page_link(page, label="Open →")


def show_cortex_badge() -> None:
    """Sidebar footer badge. Kept minimal; no raw HTML blocks."""
    with st.sidebar:
        st.divider()
        st.caption("Built with")
        st.markdown(
            f"<a href='https://docs.snowflake.com/en/user-guide/cortex-code/cortex-code' "
            f"target='_blank' style='color:{SF_BLUE}; text-decoration:none; "
            f"font-weight:600;'>❄️ Cortex Code</a>",
            unsafe_allow_html=True,
        )
        st.caption("AI-Assisted ML Development")
