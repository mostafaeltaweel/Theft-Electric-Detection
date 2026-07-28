"""
History Page Module for ETD-XAI.
Renders prediction history and export utilities.
"""
import streamlit as st
import pandas as pd
from src.database import get_db

def render_history():
    """Renders the prediction history logs."""
    st.markdown(
        """
        <div class="glass-card" style="margin-bottom: 20px;">
            <h2 style="margin: 0; font-weight: 700;">📜 Prediction History</h2>
            <p style="color: #94a3b8; margin: 4px 0 0 0;">Historical record of single and batch model inferences stored in SQLite.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    with get_db() as conn:
        df_manual = pd.read_sql_query("SELECT id, customer_id, probability, risk_score, status, predicted_at, model_name FROM manual_predictions ORDER BY id DESC;", conn)
        df_batch = pd.read_sql_query("SELECT id, customer_id, probability, risk_score, status, timestamp FROM predictions ORDER BY id DESC LIMIT 500;", conn)
        
    tab1, tab2 = st.tabs(["🔮 Single Predictions", "📦 Batch Inferences"])
    
    with tab1:
        if not df_manual.empty:
            st.dataframe(df_manual, use_container_width=True, hide_index=True)
            csv = df_manual.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export CSV", csv, "single_predictions_history.csv", "text/csv")
        else:
            st.info("No single prediction records found.")
            
    with tab2:
        if not df_batch.empty:
            st.dataframe(df_batch, use_container_width=True, hide_index=True)
            csv_b = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export Batch CSV", csv_b, "batch_predictions_history.csv", "text/csv")
        else:
            st.info("No batch inference records found.")
