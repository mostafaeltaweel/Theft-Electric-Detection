"""
Prediction Page Module for ETD-XAI.
Renders Single Consumer Prediction & Batch CSV Upload workflows.
"""
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.config import COLOR_PRIMARY, COLOR_SUCCESS, COLOR_DANGER, COLOR_WARNING
from src.predictor import predict_one, predict_sequences, classify
from src.database import save_manual_prediction

def render_predict_page():
    """Renders manual single prediction & batch prediction UI."""
    st.markdown(
        """
        <div class="glass-card" style="margin-bottom: 20px;">
            <h2 style="margin: 0; font-weight: 700;">⚡ Electricity Theft Prediction Engine</h2>
            <p style="color: #94a3b8; margin: 4px 0 0 0;">Run real-time CNN-LSTM inference on individual consumer kWh sequences or batch CSV files.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    subtab = st.session_state.get("predict_subtab", "Single")
    tab1, tab2 = st.tabs(["🔮 Single Consumer Prediction", "📦 Batch CSV Prediction"])
    
    with tab1:
        st.markdown("#### Enter 120-Day Consumption Readings (kWh)")
        st.caption("Provide comma-separated daily reading values or generate sample readings.")
        
        col_btn1, col_btn2 = st.columns(2)
        sample_type = col_btn1.radio("Sample Sequence Generator", ["Normal Sample", "Theft Sample"], horizontal=True)
        
        if sample_type == "Normal Sample":
            np.random.seed(42)
            default_readings = (1800 + np.random.normal(0, 150, 120)).clip(100, None).round(1).tolist()
        else:
            np.random.seed(42)
            base = (1800 + np.random.normal(0, 150, 120)).round(1)
            base[60:] = np.random.uniform(0, 30, 60).round(1) # Sudden drop to near zero
            default_readings = base.tolist()
            
        readings_str = st.text_area("Readings (120 floats)", value=", ".join(map(str, default_readings)), height=100)
        cust_id = st.text_input("Customer ID", value="CUST_DEMO_001")
        
        if st.button("🚀 Run Prediction", type="primary", use_container_width=True):
            try:
                vals = [float(x.strip()) for x in readings_str.split(",") if x.strip()]
                if len(vals) < 10:
                    st.error("Please provide at least 10 reading values.")
                else:
                    with st.spinner("Running CNN-LSTM model inference..."):
                        res = predict_one(vals, customer_id=cust_id)
                        save_manual_prediction(res)
                        
                    st.markdown("---")
                    st.markdown("### Prediction Verdict")
                    
                    p1, p2, p3, p4 = st.columns(4)
                    status_color = COLOR_DANGER if res["prediction"] == 1 else COLOR_SUCCESS
                    
                    p1.metric("Status", res["status"])
                    p2.metric("Theft Probability", f"{res['probability']*100:.2f}%")
                    p3.metric("Risk Score", f"{res['risk_score']} / 100")
                    p4.metric("Risk Level", res["risk_level"])
                    
                    # Sequence Plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=vals,
                        mode='lines',
                        name='kWh Consumption',
                        line=dict(color=status_color, width=2.5)
                    ))
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(15, 23, 42, 0.6)',
                        font=dict(color='#94a3b8', family='Inter'),
                        title=f"Consumption Profile — {cust_id}",
                        height=300,
                        margin=dict(l=20, r=20, t=35, b=20),
                        yaxis=dict(title="kWh", gridcolor='rgba(255,255,255,0.06)')
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error parsing inputs or running prediction: {e}")

    with tab2:
        st.markdown("#### Upload CSV File for Batch Prediction")
        st.caption("Upload a CSV file containing customer ID and reading columns.")
        
        file = st.file_uploader("Upload CSV Dataset", type=["csv", "xlsx"])
        if file is not None:
            try:
                df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
                st.write(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")
                st.dataframe(df.head(5), use_container_width=True)
                
                if st.button("🚀 Process Batch Predictions", type="primary"):
                    with st.spinner("Processing batch inference..."):
                        # Extract reading columns
                        reading_cols = [c for c in df.columns if c not in ("CONS_NO", "FLAG", "ID", "customer_id")]
                        mat = df[reading_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy()
                        
                        probs = predict_sequences(mat)
                        results = []
                        for i, p in enumerate(probs):
                            cid = str(df.iloc[i].get("CONS_NO", f"CUST_{i+1:06d}"))
                            item = classify(float(p))
                            item["Customer ID"] = cid
                            results.append(item)
                            
                        res_df = pd.DataFrame(results)
                        st.success(f"Processed {len(res_df):,} predictions successfully!")
                        st.dataframe(res_df[["Customer ID", "status", "probability", "risk_score", "risk_level"]], use_container_width=True)
                        
                        csv_data = res_df.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download Prediction Results CSV", csv_data, "batch_prediction_results.csv", "text/csv")
            except Exception as e:
                st.error(f"Error processing batch file: {e}")
