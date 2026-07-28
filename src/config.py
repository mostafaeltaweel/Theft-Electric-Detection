"""
Central Configuration Module for ETD-XAI Enterprise.
All directory paths, model constants, and database settings are managed here.
"""
import os
import tempfile
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"

# File Paths
DATASET_PATH = ASSETS_DIR / "Book4-7-4months.csv"
MODEL_PATH = ASSETS_DIR / "tl_cnnlstm_final.keras"
SCALER_PATH = ASSETS_DIR / "stat_scaler.pkl"
MODEL_CONFIG_PATH = ASSETS_DIR / "model_config.json"
LOGO_PATH = ASSETS_DIR / "logo.png"

# Writable Data Directory (compatible with Streamlit Cloud & Local environments)
DATA_DIR = Path(os.environ.get("ETD_DATA_DIR", str(Path(tempfile.gettempdir()) / "etd_xai")))
try:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    DATA_DIR = Path(tempfile.gettempdir())

DB_PATH = Path(os.environ.get("DATABASE_PATH", str(DATA_DIR / "etd_xai.db")))
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# App Branding
APP_TITLE = "Electricity Theft Detection using Explainable AI"
APP_SUBTITLE = "Enterprise Real-Time Grid Surveillance & Anomaly Detection System"
APP_VERSION = "2.0.0"

# ============================================================
# Design System Palette — v3.0 (Light, realistic CRM-style)
# Matches: white page, pastel-tinted KPI cards, soft area chart
# ============================================================

# Page & surface
COLOR_BG = "#f7f8fa"              # page background (very light gray)
COLOR_CARD = "#ffffff"            # card surface (pure white)
COLOR_BORDER = "#eaecef"          # hairline border for cards/tables

# Text
COLOR_TEXT = "#1f2430"            # main heading text (near-black, not pure black)
COLOR_TEXT_MUTED = "#8a90a0"      # secondary/description text
COLOR_TEXT_LABEL = "#6b7280"      # small uppercase labels above KPI values

# Core accent colors (soft/pastel versions — used for text, icons, chart lines)
COLOR_PRIMARY = "#5b7fdb"         # soft blue
COLOR_CYAN = "#4fb0c6"            # soft teal
COLOR_SUCCESS = "#4caf7d"         # soft green
COLOR_DANGER = "#e0767a"          # soft rose (not pure red — less alarming)
COLOR_WARNING = "#d9a441"         # soft amber

# Pastel tint backgrounds for KPI cards (used behind the accent colors above)
TINT_PRIMARY = "#eef2fc"
TINT_CYAN = "#eaf6f8"
TINT_SUCCESS = "#eaf7f0"
TINT_DANGER = "#fbeeee"
TINT_WARNING = "#fbf3e6"

# Chart-specific colors
CHART_LINE_NORMAL = COLOR_PRIMARY
CHART_FILL_NORMAL = "rgba(91, 127, 219, 0.14)"
CHART_LINE_DANGER = COLOR_DANGER
CHART_FILL_DANGER = "rgba(224, 118, 122, 0.12)"
CHART_GRIDLINE = "#eef0f3"

# Feature & Data Constants
SEQUENCE_LENGTH = 120
N_STAT_FEATURES = 59
DEFAULT_THRESHOLD = 0.5

# ------------------------------------------------------------
# Synthetic calendar mapping for the day1..day120 columns.
# The dataset only carries a day index (day1, day2, ...), not real dates,
# so we anchor it to a fixed start date and treat every month as exactly
# 30 days (not a variable-length calendar month) per project requirements.
# Change DATA_START_DATE if the real collection start date is known.
# ------------------------------------------------------------
DATA_START_DATE = "2024-01-01"
DAYS_PER_MONTH = 30
DAYS_PER_QUARTER = DAYS_PER_MONTH * 3   # 90
DAYS_PER_YEAR = DAYS_PER_MONTH * 12     # 360 (12 x 30-day months, not 365)
