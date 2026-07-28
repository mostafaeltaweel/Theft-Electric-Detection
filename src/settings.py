"""
Settings Page Module for ETD-XAI.
Provides system diagnostics, model information, and environment configuration settings.
"""
import streamlit as st
from src.config import DATASET_PATH, MODEL_PATH, DB_PATH
from src.model_loader import get_model_metadata
from src.database import get_system_db_meta

def render_settings():
    """Renders system settings and model diagnostics panel."""
    st.markdown(
        """
        <div class="glass-card" style="margin-bottom: 20px;">
            <h2 style="margin: 0; font-weight: 700;">⚙️ System Settings & Diagnostics</h2>
            <p style="color: #94a3b8; margin: 4px 0 0 0;">Inspect model parameters, database paths, and environment readiness.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    meta = get_model_metadata()
    db_meta = get_system_db_meta()
    
    st.markdown("#### Model Specifications")
    m1, m2 = st.columns(2)
    with m1:
        st.json({
            "Active Model": meta["name"],
            "Architecture": meta["architecture"],
            "Input Shape": meta["input_shape"],
            "Output Shape": meta["output_shape"],
            "Total Parameters": meta["total_params_fmt"],
            "Loaded at": meta["load_time"]
        })
    with m2:
        st.json({
            "TensorFlow Version": meta["tf_version"],
            "Keras Version": meta["keras_version"],
            "SQLite Version": db_meta["sqlite_version"],
            "SQLite File Size": db_meta["db_size"],
            "Total Consumer Records": db_meta["consumers_records"]
        })
        
    st.markdown("#### Environment Paths")
    st.code(f"""
DATASET_PATH = {DATASET_PATH}
MODEL_PATH   = {MODEL_PATH}
DB_PATH      = {DB_PATH}
    """, language="python")
