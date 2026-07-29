"""
ETD-XAI Enterprise v2.0  —  Electricity Theft Detection using Explainable AI
============================================================================
Main Streamlit Entrypoint.
Full-screen slideshow landing page; dashboard loads only after entering.
"""
import base64
import glob
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="ETD-XAI Enterprise",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

from src.config import MODEL_PATH, BASE_DIR
from src.utils import inject_custom_css
from src.database import init_db, ensure_schema
from src.model_loader import load_active_model, is_model_loaded
from src.dashboard import render_dashboard_page

inject_custom_css()

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True
)


def render_hero_slideshow(height_css: str = "100vh", radius: str = "0px"):
    """Auto-playing fade carousel using every image in assets/slideshow/."""
    slideshow_dir = BASE_DIR / "assets" / "slideshow"
    paths = sorted(
        glob.glob(str(slideshow_dir / "*.png"))
        + glob.glob(str(slideshow_dir / "*.jpg"))
        + glob.glob(str(slideshow_dir / "*.jpeg"))
    )
    if not paths:
        return

    slot_seconds = 5
    fade_seconds = 0.8
    n = len(paths)
    total = n * slot_seconds

    fade_in_pct = (fade_seconds / total) * 100
    hold_end_pct = ((slot_seconds - fade_seconds) / total) * 100
    fade_out_pct = (slot_seconds / total) * 100

    slides_html = []
    for i, p in enumerate(paths):
        ext = Path(p).suffix.replace(".", "")
        b64 = base64.b64encode(Path(p).read_bytes()).decode()
        slides_html.append(
            f'<div class="hero-slide" style="animation-delay:{i * slot_seconds}s; '
            f'background-image:url(data:image/{ext};base64,{b64});"></div>'
        )

    st.markdown(
        f"""
        <style>
        .hero-slideshow {{
            position: relative;
            width: 100%;
            height: {height_css};
            border-radius: {radius};
            overflow: hidden;
            margin: 0;
        }}
        .hero-slide {{
            position: absolute;
            inset: 0;
            background-size: cover;
            background-position: center;
            opacity: 0;
            animation: heroFade {total}s ease-in-out infinite;
        }}
        @keyframes heroFade {{
            0% {{ opacity: 0; }}
            {fade_in_pct:.2f}% {{ opacity: 1; }}
            {hold_end_pct:.2f}% {{ opacity: 1; }}
            {fade_out_pct:.2f}% {{ opacity: 0; }}
            100% {{ opacity: 0; }}
        }}
        </style>
        <div class="hero-slideshow">
            {''.join(slides_html)}
        </div>
        """,
        unsafe_allow_html=True
    )


def render_landing_page():
    """Full-bleed, full-viewport landing screen with the entry button
    overlaid directly on top of the image — no extra motion/animation
    on the button itself."""

    # Remove Streamlit's default page padding so the image is truly full-screen
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
            padding-left: 0rem !important;
            padding-right: 0rem !important;
            max-width: 100% !important;
        }
        header[data-testid="stHeader"] { background: transparent !important; }

        /* Pull the button up so it sits visually inside the image, near the
           bottom edge. No hover animation / transform — flat and static. */
        div.stButton {
            margin-top: -110px;
            display: flex;
            justify-content: center;
        }
        div.stButton > button {
            transition: none !important;
            transform: none !important;
            box-shadow: none !important;
        }
        div.stButton > button:hover {
            transform: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    render_hero_slideshow(height_css="100vh", radius="0px")

    _, mid, _ = st.columns([2, 1, 2])
    with mid:
        if st.button("Enter Application", use_container_width=True, type="primary"):
            st.session_state.app_entered = True
            st.rerun()


# ---------------------------------------------------------
# Gate: show landing page first, dashboard only after entering
# ---------------------------------------------------------
if "app_entered" not in st.session_state:
    st.session_state.app_entered = False

if not st.session_state.app_entered:
    render_landing_page()
    st.stop()

# Schema migration is cheap (a few PRAGMA/ALTER/CREATE INDEX checks) and is
# run UNCACHED on every single app start — this guarantees new columns like
# 'source'/'upload_id' always exist before any upload is processed, even if
# the cached bootstrap step below doesn't re-run after a deploy.
ensure_schema()

# Boot Database & Model ONCE (only needed once the user is inside the app)
@st.cache_resource(show_spinner="Booting SQLite Database & CNN-LSTM Model Engine...")
def bootstrap_app():
    init_db()
    return load_active_model(MODEL_PATH)

bootstrap_app()
if not is_model_loaded():
    load_active_model(MODEL_PATH)

render_dashboard_page()

st.markdown(
    """
    <div style="text-align: center; color: #8a90a0; font-size: 0.85rem; margin-top: 40px; padding: 20px 0; border-top: 1px solid rgba(0,0,0,0.06);">
        Electricity Theft Detection using Explainable AI (ETD-XAI) &copy; 2026 Enterprise Graduation Project
    </div>
    """,
    unsafe_allow_html=True
)
