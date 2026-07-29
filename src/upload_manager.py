"""
Upload Manager Module for ETD-XAI.
Manages user uploaded datasets, stores predictions in uploaded_predictions table,
prevents FLAG data leakage, and computes evaluation metrics when ground truth is supplied.
"""
import io
import json
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

from src.database import get_db, upsert_consumers_from_upload
from src.predictor import predict_sequences, classify

def process_user_uploaded_file(file, user_id: str = "Administrator") -> Dict[str, Any]:
    """
    Processes an uploaded CSV or Excel file.
    - Creates a record in the 'uploads' table.
    - Removes FLAG column before passing data to the model (zero leakage).
    - Runs CNN-LSTM model inference.
    - Stores results in 'uploaded_predictions' table (never overwriting official consumers table).
    - Computes evaluation metrics if FLAG was provided.
    """
    filename = getattr(file, "name", "uploaded_dataset.csv")
    file_lower = filename.lower()
    
    # Read file
    if file_lower.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)
        
    n_rows = len(df)
    upload_date_str = datetime.now().isoformat()
    
    # 1. Create entry in 'uploads' table
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO uploads (file_name, upload_date, number_of_records, status, uploaded_by)
               VALUES (?, ?, ?, ?, ?)""",
            (filename, upload_date_str, n_rows, "Processing", user_id)
        )
        upload_id = cur.lastrowid

    # 2. Column Detection (ID, FLAG, Readings)
    id_cols = [c for c in df.columns if str(c).strip().lower() in ("cons_no", "customer_id", "consumer_id", "id", "meter_id")]
    id_col = id_cols[0] if id_cols else None
    
    flag_cols = [c for c in df.columns if str(c).strip().lower() in ("flag", "label", "target", "theft", "is_theft")]
    flag_col = flag_cols[0] if flag_cols else None
    
    has_flag = flag_col is not None
    
    # Reading columns ONLY (exclude ID and FLAG to prevent data leakage)
    reading_cols = [c for c in df.columns if c not in (id_col, flag_col) and pd.to_numeric(df[c], errors='coerce').notna().mean() >= 0.5]
    
    readings_mat = df[reading_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
    ids = df[id_col].astype(str).tolist() if id_col else [f"CUST_{i+1:06d}" for i in range(n_rows)]
    ground_truth_flags = pd.to_numeric(df[flag_col], errors='coerce').fillna(0).astype(int).tolist() if has_flag else None
    
    # 3. Run CNN-LSTM Model Inference (Pure features, NO FLAG)
    probs = predict_sequences(readings_mat)
    
    # 4. Save results to 'uploaded_predictions' table (unchanged — kept as
    #    the audit/history record of this upload) AND merge into the main
    #    'consumers' table so the Enterprise Dashboard, KPIs, charts, and
    #    Consumer Search immediately reflect the uploaded data.
    db_records = []
    consumer_records = []
    ui_results = []
    
    now_ts = datetime.now().isoformat()
    
    for i in range(n_rows):
        cls = classify(float(probs[i]))
        gt_flag = ground_truth_flags[i] if has_flag else None
        row_readings = readings_mat[i]
        
        db_records.append((
            upload_id,
            ids[i],
            cls["prediction"],
            cls["probability"],
            cls["risk_level"],
            cls["status"],
            gt_flag,
            now_ts
        ))

        # (CONS_NO, FLAG, readings_json, avg_cons, total_cons, zero_days,
        #  probability, prediction, risk_score, risk_level, status)
        consumer_records.append((
            ids[i],
            gt_flag,
            json.dumps(row_readings.tolist()),
            float(np.mean(row_readings)),
            float(np.sum(row_readings)),
            int(np.sum(row_readings == 0.0)),
            cls["probability"],
            cls["prediction"],
            cls["risk_score"],
            cls["risk_level"],
            cls["status"],
        ))
        
        item = {
            "customer_id": ids[i],
            "prediction": cls["prediction"],
            "status": cls["status"],
            "probability": cls["probability"],
            "risk_score": cls["risk_score"],
            "risk_level": cls["risk_level"]
        }
        if has_flag:
            item["ground_truth_flag"] = gt_flag
        ui_results.append(item)

    with get_db() as conn:
        conn.executemany(
            """INSERT INTO uploaded_predictions 
               (upload_id, customer_id, prediction, probability, risk_level, status, flag, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            db_records
        )
        # Update upload status
        conn.execute("UPDATE uploads SET status = 'Completed' WHERE upload_id = ?;", (upload_id,))

    # Merge into 'consumers' (INSERT if CONS_NO is new, UPDATE if it already
    # exists) — tagged source='upload' so it can be found and safely deleted
    # later without ever touching the original system dataset.
    upsert_consumers_from_upload(consumer_records, upload_id)

    res_df = pd.DataFrame(ui_results)
    
    # 5. Compute evaluation metrics if FLAG ground truth is present
    metrics = None
    if has_flag and ground_truth_flags is not None:
        metrics = compute_evaluation_metrics(np.array(ground_truth_flags), np.array([r["prediction"] for r in ui_results]), probs)

    theft_count = int((res_df["prediction"] == 1).sum())
    normal_count = n_rows - theft_count

    return {
        "upload_id": upload_id,
        "filename": filename,
        "total_records": n_rows,
        "theft_cases": theft_count,
        "normal_cases": normal_count,
        "theft_rate_pct": round((theft_count / max(n_rows, 1)) * 100, 2),
        "has_flag": has_flag,
        "metrics": metrics,
        "df_results": res_df
    }

def compute_evaluation_metrics(flags: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    """Calculates accuracy, precision, recall, F1, and ROC-AUC on evaluation dataset."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    out = {
        "accuracy": float(accuracy_score(flags, preds)),
        "precision": float(precision_score(flags, preds, zero_division=0)),
        "recall": float(recall_score(flags, preds, zero_division=0)),
        "f1_score": float(f1_score(flags, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(flags, preds, labels=[0, 1]).tolist()
    }
    try:
        out["roc_auc"] = float(roc_auc_score(flags, probs))
    except Exception:
        out["roc_auc"] = None
    return out

def get_recent_uploads(limit: int = 10) -> pd.DataFrame:
    """Fetches uploaded files metadata from uploads table."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT upload_id as 'Upload ID', file_name as 'File Name', 
                      number_of_records as 'Records', status as 'Status', 
                      upload_date as 'Date', uploaded_by as 'Uploaded By'
               FROM uploads ORDER BY upload_id DESC LIMIT ?""", (limit,)
        ).fetchall()
        
    if not rows:
        return pd.DataFrame(columns=["Upload ID", "File Name", "Records", "Status", "Date", "Uploaded By"])
    return pd.DataFrame([dict(r) for r in rows])

def export_results_to_excel(df: pd.DataFrame) -> bytes:
    """Generates Excel download bytes."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Predictions")
    return buf.getvalue()
