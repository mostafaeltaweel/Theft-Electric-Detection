"""
Analytics Page Module for ETD-XAI.
Provides interactive data exploration, statistical distributions, and consumer profile lookups.
"""
import json
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.config import COLOR_PRIMARY, COLOR_CYAN, COLOR_SUCCESS, COLOR_DANGER
from src.database import get_db, get_kpi_stats

def render_analytics():
    """Renders the Data Analytics & Insights page."""
    st.markdown(
        """
        <div class="glass-card" style="margin-bottom: 20px;">
            <h2 style="margin: 0; font-weight: 700;">📈 Data Analytics & Anomaly Insights</h2>
            <p style="color: #94a3b8; margin: 4px 0 0 0;">Deep consumption statistics and individual consumer waveform inspection.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    tab1, tab2 = st.tabs(["🔍 Consumer Waveform Search", "📊 Dataset Statistical Overview"])
    
    with tab1:
        st.markdown("#### Search Consumer Timeline")
        with get_db() as conn:
            consumers = conn.execute("SELECT CONS_NO FROM consumers LIMIT 200;").fetchall()
            
        c_list = [c["CONS_NO"] for c in consumers] if consumers else ["CUST_000001"]
        selected_cid = st.selectbox("Select Consumer ID", c_list)
        
        if selected_cid:
            with get_db() as conn:
                row = conn.execute("SELECT * FROM consumers WHERE CONS_NO = ?;", (selected_cid,)).fetchone()
                
            if row and row["readings_json"]:
                readings = json.loads(row["readings_json"])
                flag = row["FLAG"]
                status_label = "Theft (Class 1)" if flag == 1 else "Normal (Class 0)"
                badge_class = "badge-danger" if flag == 1 else "badge-success"
                
                st.markdown(f"**Status**: <span class='badge-status {badge_class}'>{status_label}</span> | **Avg Consumption**: {row['avg_cons']:.2f} kWh", unsafe_allow_html=True)
                
                days = [f"Day {i+1}" for i in range(len(readings))]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=days, y=readings,
                    mode='lines+markers',
                    name=selected_cid,
                    line=dict(color=COLOR_PRIMARY if flag == 0 else COLOR_DANGER, width=2),
                    marker=dict(size=4)
                ))
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(15, 23, 42, 0.6)',
                    font=dict(color='#94a3b8', family='Inter'),
                    height=350,
                    margin=dict(l=20, r=20, t=20, b=20),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)', title="kWh")
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("#### Dataset Overview")
        stats = get_kpi_stats()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows in SQLite", f"{stats['total_consumers']:,}")
        c2.metric("Theft Count", f"{stats['theft_cases']:,}")
        c3.metric("Normal Count", f"{stats['normal_consumers']:,}")
        
        with get_db() as conn:
            df_sample = pd.read_sql_query("SELECT CONS_NO, FLAG, avg_cons, total_cons, zero_days FROM consumers LIMIT 1000;", conn)
            
        if not df_sample.empty:
            st.markdown("##### Sample Consumer Metadata (First 1,000 Rows)")
            st.dataframe(df_sample, use_container_width=True)
