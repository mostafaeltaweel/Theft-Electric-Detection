"""
UI Design System & Utility Helpers for ETD-XAI.
Provides CSS injection for a clean, light CRM-style dashboard and custom HTML components.
"""
import streamlit as st
from src.config import (
    COLOR_PRIMARY, COLOR_CYAN, COLOR_SUCCESS, COLOR_DANGER, COLOR_WARNING,
    COLOR_CARD, COLOR_BORDER, COLOR_BG, COLOR_TEXT, COLOR_TEXT_MUTED, COLOR_TEXT_LABEL,
    TINT_PRIMARY, TINT_CYAN, TINT_SUCCESS, TINT_DANGER, TINT_WARNING,
)

_ACCENT_TO_TINT = {
    COLOR_PRIMARY: TINT_PRIMARY,
    COLOR_CYAN: TINT_CYAN,
    COLOR_SUCCESS: TINT_SUCCESS,
    COLOR_DANGER: TINT_DANGER,
    COLOR_WARNING: TINT_WARNING,
}


def inject_custom_css():
    """Injects a clean, light, low-contrast, large-type CSS theme into Streamlit."""
    css = f"""
    <style>
    html {{ font-size: 17px; }}
    .stApp {{
        background: {COLOR_BG};
        color: {COLOR_TEXT};
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }}
    .stMarkdown, .stText, p, li, label {{
        font-size: 1rem;
    }}

    header[data-testid="stHeader"] {{
        background: rgba(247, 248, 250, 0.9) !important;
        backdrop-filter: blur(10px) !important;
    }}

    /* ============================================================
       Centered, max-width page container — reduces the "everything
       stretched too wide / lots of empty space" look on large screens
       ============================================================ */
    .block-container {{
        max-width: 1500px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        padding-top: 5.5rem !important;   /* clears the fixed nav bar below */
    }}
    @media (max-width: 1600px) {{
        .block-container {{ max-width: 96% !important; }}
    }}
    @media (max-width: 900px) {{
        .block-container {{ padding-left: 1rem !important; padding-right: 1rem !important; }}
    }}

    .glass-card {{
        background: {COLOR_CARD};
        border: 1px solid {COLOR_BORDER};
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(16, 24, 40, 0.04);
    }}

    /* KPI Metric Cards — tight spacing, large value, no wasted height */
    .kpi-card {{
        border-radius: 12px;
        padding: 16px 20px;
        text-align: left;
    }}
    .kpi-title {{
        color: {COLOR_TEXT_LABEL};
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 8px;
    }}
    .kpi-value {{
        color: {COLOR_TEXT};
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        line-height: 1.1;
    }}
    .kpi-desc {{
        color: {COLOR_TEXT_MUTED};
        font-size: 0.85rem;
        margin-top: 6px;
    }}

    .badge-status {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
    }}
    .badge-success {{ background: {TINT_SUCCESS}; color: {COLOR_SUCCESS}; border: 1px solid rgba(76, 175, 125, 0.25); }}
    .badge-danger  {{ background: {TINT_DANGER};  color: {COLOR_DANGER};  border: 1px solid rgba(224, 118, 122, 0.25); }}
    .badge-primary {{ background: {TINT_PRIMARY}; color: {COLOR_PRIMARY}; border: 1px solid rgba(91, 127, 219, 0.25); }}

    .action-tile {{
        background: {COLOR_CARD};
        border: 1px solid {COLOR_BORDER};
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s ease;
    }}
    .action-tile:hover {{ border-color: {COLOR_PRIMARY}; box-shadow: 0 2px 8px rgba(91, 127, 219, 0.12); }}
    .action-label {{ font-weight: 600; color: {COLOR_TEXT}; font-size: 0.95rem; }}

    div[data-testid="stDataFrame"] {{
        background: {COLOR_CARD};
        border-radius: 10px;
        border: 1px solid {COLOR_BORDER};
        font-size: 1rem;
    }}

    .section-header {{
        font-size: 1.25rem;
        font-weight: 700;
        color: {COLOR_TEXT};
        margin: 28px 0 12px 0;
    }}

    /* ============================================================
       Navigation Bar (st.tabs) — full-width, fixed at top, shadow,
       bottom border, active-tab highlight, generous side padding.
       IMPORTANT: only the tab-list (button row) is fixed — NOT the
       tab-panel (page content), which must scroll normally.
       ============================================================ */
    div[data-testid="stTabs"] div[data-baseweb="tab-list"] {{
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 999;
        display: flex;
        gap: 0;
        margin: 0;
        background: {COLOR_CARD};
        border-bottom: 1px solid {COLOR_BORDER};
        box-shadow: 0 2px 8px rgba(16, 24, 40, 0.06);
        padding: 0 2.5rem;
    }}
    div[data-testid="stTabs"] div[data-baseweb="tab-panel"] {{
        position: relative;
    }}
    div[data-testid="stTabs"] button[data-baseweb="tab"] {{
        flex: 0 0 auto;
        height: 58px;
        padding: 0 28px;
        background-color: transparent;
        color: {COLOR_TEXT_MUTED};
        font-weight: 600;
        font-size: 1.05rem;
        border: none;
        border-bottom: 3px solid transparent;
        transition: background-color 0.15s ease, color 0.15s ease, border-color 0.15s ease;
    }}
    div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {{
        background-color: {TINT_PRIMARY};
        color: {COLOR_PRIMARY};
    }}
    div[data-testid="stTabs"] button[aria-selected="true"] {{
        color: {COLOR_PRIMARY};
        border-bottom: 3px solid {COLOR_PRIMARY};
        background-color: {TINT_PRIMARY};
    }}
    div[data-testid="stTabs"] div[data-baseweb="tab-highlight"] {{
        display: none;   /* replaced by the border-bottom on the active button above */
    }}
    div[data-testid="stTabs"] button[data-baseweb="tab"] p {{
        font-size: 1.05rem;
        font-weight: 600;
    }}
    @media (max-width: 900px) {{
        div[data-testid="stTabs"] div[data-baseweb="tab-list"] {{ padding: 0 1rem; }}
        div[data-testid="stTabs"] button[data-baseweb="tab"] {{ padding: 0 16px; font-size: 0.95rem; }}
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_kpi_card(title: str, value: str, desc: str, icon: str = "", accent_color: str = COLOR_PRIMARY):
    """Renders a compact, pastel-tinted KPI card (light theme).
    Built as a single-line HTML string on purpose — a blank line inside an
    HTML block makes Streamlit's markdown parser end the block early and
    render the remaining tags as literal text instead of HTML."""
    tint = _ACCENT_TO_TINT.get(accent_color, TINT_PRIMARY)
    icon_html = f'<span style="font-size: 1.2rem; opacity: 0.85;">{icon}</span>' if icon else ""
    html = (
        f'<div class="kpi-card" style="background:{tint};">'
        f'<div style="display: flex; justify-content: space-between; align-items: flex-start;">'
        f'<div class="kpi-title">{title}</div>{icon_html}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-desc">{desc}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_metric_box(label: str, value: str, accent_color: str = COLOR_PRIMARY):
    """Renders a small pastel-tinted metric box — used for st.metric() replacements.
    Single-line HTML (see render_kpi_card note above for why)."""
    tint = _ACCENT_TO_TINT.get(accent_color, TINT_PRIMARY)
    html = (
        f'<div style="background:{tint}; border-radius:10px; padding:14px 16px;">'
        f'<div style="color:{COLOR_TEXT_LABEL}; font-size:0.8rem; font-weight:600; '
        f'text-transform:uppercase; letter-spacing:0.04em; margin-bottom:8px;">{label}</div>'
        f'<div style="color:{COLOR_TEXT}; font-size:1.6rem; font-weight:800; line-height:1.1;">{value}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)
