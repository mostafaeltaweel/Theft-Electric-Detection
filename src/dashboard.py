"""
Enterprise Dashboard Module for ETD.
Reads 100% directly from SQLite database via SQL aggregation queries for sub-100ms rendering.
Provides isolated dataset upload management via src/upload_manager.py.
"""
import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.config import (
    COLOR_PRIMARY, COLOR_CYAN, COLOR_SUCCESS, COLOR_DANGER, COLOR_WARNING,
    CHART_LINE_NORMAL, CHART_FILL_NORMAL, CHART_LINE_DANGER, CHART_FILL_DANGER,
    CHART_GRIDLINE, COLOR_TEXT_MUTED,
    DATA_START_DATE, DAYS_PER_MONTH, DAYS_PER_QUARTER, DAYS_PER_YEAR
)
from src.database import (
    get_dashboard_kpis, 
    get_consumption_profiles_sql, 
    get_consumer_by_id, 
    get_all_consumer_ids, 
    get_recent_prediction_history,
    get_system_db_meta,
    get_uploadable_batches,
    delete_consumers_by_upload
)
from src.upload_manager import process_user_uploaded_file, get_recent_uploads, export_results_to_excel
from src.utils import render_kpi_card, render_metric_box

def aggregate_series(daily_values, level: str):
    """Aggregates per-day average kWh values into Week/Month/Quarter/Year SUMS.

    The day1..day120 columns are treated as a real date column: day1 maps to
    DATA_START_DATE (src/config.py) and every subsequent day column is one
    calendar day later. Every month is fixed at exactly 30 days (not a
    variable-length Gregorian month), so quarters are 90 days and years are
    360 days (12 x 30), per project convention.
    """
    n = len(daily_values)
    dates = pd.date_range(start=DATA_START_DATE, periods=n, freq="D")

    if level == "Day":
        return [d.strftime("%d %b %Y") for d in dates], list(daily_values)

    bucket_size = {"Week": 7, "Month": DAYS_PER_MONTH, "Quarter": DAYS_PER_QUARTER, "Year": DAYS_PER_YEAR}[level]
    n_buckets = math.ceil(n / bucket_size)

    labels, sums = [], []
    for b in range(n_buckets):
        start = b * bucket_size
        end = min(start + bucket_size, n)
        sums.append(float(np.sum(daily_values[start:end])))
        block_start_date = dates[start]

        if level == "Week":
            labels.append(f"W{b + 1} ({block_start_date.strftime('%d %b')})")
        elif level == "Month":
            labels.append(block_start_date.strftime("%b %Y"))
        elif level == "Quarter":
            quarter_num = (b % 4) + 1
            labels.append(f"Q{quarter_num} {block_start_date.year}")
        else:  # Year
            labels.append(str(block_start_date.year))

    return labels, sums


def render_dashboard_page():
    """Renders the high-performance enterprise dashboard."""

    # Tabs — plain text, no emoji, large box style, sticky at the top (see utils.py CSS)
    tab_dash, tab_upload = st.tabs(["Enterprise Dashboard", "Upload & Evaluate Dataset"])

    with tab_dash:
        # Fetch SQL KPI Aggregations (<20ms response time)
        kpis = get_dashboard_kpis()

        # ---------------------------------------------------------
        # 1. KPI Cards — no emoji icons, large numbers, no wasted space
        # ---------------------------------------------------------
        c1, c2, c3 = st.columns(3, gap="medium")

        with c1:
            render_kpi_card("Total Consumers", f"{kpis['total_consumers']:,}", "Official dataset in SQLite", icon="", accent_color=COLOR_PRIMARY)
        with c2:
            render_kpi_card("Theft Cases", f"{kpis['theft_cases']:,}", f"{kpis['theft_rate_pct']}% theft rate", icon="", accent_color=COLOR_DANGER)
        with c3:
            render_kpi_card("Normal Consumers", f"{kpis['normal_consumers']:,}", "Verified legitimate usage", icon="", accent_color=COLOR_SUCCESS)

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        # ---------------------------------------------------------
        # 2. Charts Row (Consumption Trend & Donut Split via SQL)
        # ---------------------------------------------------------
        col_left, col_right = st.columns([7, 5])

        with col_left:
            header_col, selector_col = st.columns([3, 2])
            with header_col:
                st.markdown("<div class='section-header'>Average Daily Consumption Trend</div>", unsafe_allow_html=True)
            with selector_col:
                agg_level = st.selectbox(
                    "Aggregation",
                    ["Day", "Week", "Month", "Quarter", "Year"],
                    index=0,
                    key="trend_agg_level",
                    label_visibility="collapsed"
                )

            normal_avg, theft_avg, n_days = get_consumption_profiles_sql()

            # Aggregate (sum) both series according to the selected period —
            # Day returns the raw values unchanged; Week/Month/Quarter/Year sum
            # the daily averages into that period's total.
            x_labels, normal_agg = aggregate_series(normal_avg, agg_level)
            _, theft_agg = aggregate_series(theft_avg, agg_level)

            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=x_labels, y=normal_agg, mode='lines', name='Normal Customers',
                line=dict(color=CHART_LINE_NORMAL, width=3, shape='spline'),
                fill='tozeroy', fillcolor=CHART_FILL_NORMAL,
                hovertemplate='<b>Normal</b>: %{y:.2f} kWh<extra></extra>'
            ))
            fig_line.add_trace(go.Scatter(
                x=x_labels, y=theft_agg, mode='lines', name='Theft Customers',
                line=dict(color=CHART_LINE_DANGER, width=3, shape='spline'),
                hovertemplate='<b>Theft</b>: %{y:.2f} kWh<extra></extra>'
            ))
            fig_line.update_layout(
                paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
                font=dict(color=COLOR_TEXT_MUTED, family='Inter', size=13),
                margin=dict(l=10, r=10, t=10, b=10),
                height=380,
                transition=dict(duration=400, easing='cubic-in-out'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12)),
                xaxis=dict(showgrid=False, tickfont=dict(size=10), color=COLOR_TEXT_MUTED),
                yaxis=dict(showgrid=True, gridcolor=CHART_GRIDLINE, tickfont=dict(size=11), title="Energy Consumption (kWh)", color=COLOR_TEXT_MUTED)
            )
            st.plotly_chart(fig_line, use_container_width=True, config={'displayModeBar': False})

        with col_right:
            st.markdown("<div class='section-header'>Dataset Class Distribution</div>", unsafe_allow_html=True)
            donut_df = pd.DataFrame({
                "Category": ["Normal Customers", "Theft Customers"],
                "Count": [kpis["normal_consumers"], kpis["theft_cases"]]
            })
            fig_donut = px.pie(
                donut_df, values="Count", names="Category", hole=0.62,
                color="Category", color_discrete_map={"Normal Customers": CHART_LINE_NORMAL, "Theft Customers": CHART_LINE_DANGER}
            )
            fig_donut.update_traces(textposition='inside', textinfo='percent', hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Ratio: %{percent}')
            fig_donut.update_layout(
                paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
                font=dict(color=COLOR_TEXT_MUTED, family='Inter', size=13),
                margin=dict(l=10, r=10, t=10, b=10),
                height=380, showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5, font=dict(size=12))
            )
            st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        # ---------------------------------------------------------
        # 3. Consumer Search — type a Consumer ID, matches filter live
        # ---------------------------------------------------------
        st.markdown("<div class='section-header'>Consumer Search</div>", unsafe_allow_html=True)

        c_list = get_all_consumer_ids(limit=300)
        if c_list:
            search_term = st.text_input(
                "Type a Consumer ID to search",
                placeholder="Start typing a Consumer ID...",
                key="consumer_search_box"
            )

            if search_term:
                matches = [c for c in c_list if search_term.lower() in c.lower()]
            else:
                matches = c_list

            if not matches:
                st.warning("No consumer matches that search.")
                selected_cid = None
            else:
                selected_cid = st.selectbox(
                    f"Matching Consumers ({len(matches):,})",
                    matches
                )

            if selected_cid:
                row = get_consumer_by_id(selected_cid)
                if row and row.get("readings"):
                    readings = row["readings"]
                    badge_cls = "badge-danger" if row["prediction"] == 1 else "badge-success"

                    st.markdown(
                        f"**Customer**: `{selected_cid}` | **Verdict**: <span class='badge-status {badge_cls}'>{row['status']}</span> | "
                        f"**Risk Score**: `{row['risk_score']} / 100` | **Risk Level**: `{row['risk_level']}`",
                        unsafe_allow_html=True
                    )

                    fig_ind = go.Figure()
                    fig_ind.add_trace(go.Scatter(
                        x=days, y=readings, mode='lines+markers', name=selected_cid,
                        line=dict(color=CHART_LINE_DANGER if row["prediction"] == 1 else CHART_LINE_NORMAL, width=3),
                        marker=dict(size=5)
                    ))
                    fig_ind.update_layout(
                        paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
                        font=dict(color=COLOR_TEXT_MUTED, family='Inter', size=13),
                        height=320,
                        margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(showgrid=False, tickfont=dict(size=10)),
                        yaxis=dict(showgrid=True, gridcolor=CHART_GRIDLINE, title="kWh", tickfont=dict(size=11))
                    )
                    st.plotly_chart(fig_ind, use_container_width=True, config={'displayModeBar': False})

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        # ---------------------------------------------------------
        # 4. Recent Prediction History Table (<20ms)
        # ---------------------------------------------------------
        st.markdown("<div class='section-header'>Recent Predictions History</div>", unsafe_allow_html=True)
        recent_df = get_recent_prediction_history(limit=5)
        st.dataframe(recent_df, use_container_width=True, hide_index=True)


    with tab_upload:
        st.markdown("<div class='section-header'>Upload Custom Dataset File (CSV / Excel)</div>", unsafe_allow_html=True)
        st.caption("Uploaded records are merged into the main `consumers` table (new CONS_NO → inserted, existing CONS_NO → updated), "
                    "so they immediately appear in the Enterprise Dashboard KPIs, charts, and Consumer Search. "
                    "Each record is tagged with its Upload ID so it can be removed later from the 'Delete Uploaded Data' section below "
                    "without ever affecting the original system dataset.")

        file = st.file_uploader("Upload dataset file", type=["csv", "xlsx"])

        if file is not None:
            if st.button("Process & Ingest File", type="primary"):
                with st.spinner("Processing file, stripping FLAG for zero-leakage inference, and saving to SQLite..."):
                    res = process_user_uploaded_file(file)

                st.success(f"File **{res['filename']}** processed! Created Upload ID #{res['upload_id']}.")

                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                u1, u2, u3 = st.columns(3, gap="medium")
                with u1:
                    render_metric_box("Total Records", f"{res['total_records']:,}", accent_color=COLOR_PRIMARY)
                with u2:
                    render_metric_box("Theft Cases", f"{res['theft_cases']:,}", accent_color=COLOR_DANGER)
                with u3:
                    render_metric_box("Theft Rate", f"{res['theft_rate_pct']}%", accent_color=COLOR_WARNING)

                if res["has_flag"] and res["metrics"]:
                    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
                    st.markdown("<div class='section-header'>Model Evaluation Metrics (Ground Truth Comparison)</div>", unsafe_allow_html=True)
                    m = res["metrics"]
                    em1, em2, em3, em4, em5 = st.columns(5, gap="small")
                    with em1:
                        render_metric_box("Accuracy", f"{m['accuracy']:.4f}", accent_color=COLOR_PRIMARY)
                    with em2:
                        render_metric_box("Precision", f"{m['precision']:.4f}", accent_color=COLOR_CYAN)
                    with em3:
                        render_metric_box("Recall", f"{m['recall']:.4f}", accent_color=COLOR_SUCCESS)
                    with em4:
                        render_metric_box("F1 Score", f"{m['f1_score']:.4f}", accent_color=COLOR_WARNING)
                    with em5:
                        render_metric_box("ROC-AUC", f"{(m['roc_auc'] or 0):.4f}", accent_color=COLOR_DANGER)

                st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
                st.markdown("<div class='section-header'>Prediction Results Table</div>", unsafe_allow_html=True)
                st.dataframe(res["df_results"], use_container_width=True, hide_index=True)

                excel_bytes = export_results_to_excel(res["df_results"])
                st.download_button("Export Results to Excel", excel_bytes, f"upload_{res['upload_id']}_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Uploaded Datasets History (uploads table)</div>", unsafe_allow_html=True)
        recent_uploads_df = get_recent_uploads()
        st.dataframe(recent_uploads_df, use_container_width=True, hide_index=True)

        # ---------------------------------------------------------
        # Delete Uploaded Data — safe, easy removal.
        # Only lists batches that currently have live rows merged into
        # 'consumers' (source='upload'). The original system dataset never
        # appears here and can never be deleted through this control.
        # ---------------------------------------------------------
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Delete Uploaded Data</div>", unsafe_allow_html=True)

        deletable_df = get_uploadable_batches()
        if deletable_df.empty:
            st.caption("No uploaded data currently in the system.")
        else:
            st.dataframe(deletable_df, use_container_width=True, hide_index=True)

            del_col1, del_col2 = st.columns([3, 1])
            with del_col1:
                selected_upload_id = st.selectbox(
                    "Select an Upload ID to remove",
                    deletable_df["Upload ID"].tolist(),
                    key="delete_upload_select"
                )
            with del_col2:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                confirm_delete = st.button("Delete This Upload", type="secondary", use_container_width=True)

            if confirm_delete:
                deleted_count = delete_consumers_by_upload(selected_upload_id)
                st.success(f"Removed {deleted_count:,} consumer record(s) from Upload ID #{selected_upload_id}. "
                           f"The Dashboard, KPIs, and Consumer Search will reflect this immediately.")
                st.rerun()
