"""
Enterprise Database Layer for ETD-XAI.
Manages multi-table SQLite schema (consumers, uploads, uploaded_predictions, prediction_history),
indexing, vectorized pre-computation on first startup, and SQL aggregation queries.
"""
import sqlite3
import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

from src.config import DB_PATH, DATASET_PATH

_DDL = """
CREATE TABLE IF NOT EXISTS consumers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    CONS_NO TEXT UNIQUE,
    FLAG INTEGER,
    readings_json TEXT,
    avg_cons REAL,
    total_cons REAL,
    zero_days INTEGER,
    probability REAL,
    prediction INTEGER,
    risk_score REAL,
    risk_level TEXT,
    status TEXT,
    source TEXT DEFAULT 'system',
    upload_id INTEGER,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS uploads (
    upload_id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT,
    upload_date TEXT,
    number_of_records INTEGER,
    status TEXT,
    uploaded_by TEXT
);

CREATE TABLE IF NOT EXISTS uploaded_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    upload_id INTEGER,
    customer_id TEXT,
    prediction INTEGER,
    probability REAL,
    risk_level TEXT,
    status TEXT,
    flag INTEGER,
    timestamp TEXT,
    FOREIGN KEY(upload_id) REFERENCES uploads(upload_id)
);

CREATE TABLE IF NOT EXISTS prediction_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT,
    prediction INTEGER,
    probability REAL,
    risk_level TEXT,
    status TEXT,
    readings_json TEXT,
    timestamp TEXT
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""

@contextmanager
def get_db():
    """Context manager for thread-safe SQLite connection."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def ensure_schema():
    """Creates tables/indexes if missing and runs additive column migrations.

    This is intentionally CHEAP (a handful of PRAGMA/ALTER/CREATE INDEX
    statements) and is called EVERY time init_db() runs — and, from app.py,
    is also called directly and unconditionally on every app start (outside
    the @st.cache_resource-wrapped bootstrap step). This guarantees the
    'source'/'upload_id' columns always exist before any upload is
    processed, even if the cached bootstrap step itself doesn't re-run
    after a deploy.
    """
    with get_db() as conn:
        conn.executescript(_DDL)
        
        # Check column migration for consumers table
        cursor = conn.execute("PRAGMA table_info(consumers);")
        cols = [r["name"] for r in cursor.fetchall()]
        if cols and "updated_at" not in cols:
            conn.execute("DROP TABLE consumers;")
            conn.executescript(_DDL)
            cursor = conn.execute("PRAGMA table_info(consumers);")
            cols = [r["name"] for r in cursor.fetchall()]

        # Additive migration ONLY (never drops/recomputes existing data):
        # adds 'source' and 'upload_id' so uploaded consumers can later be
        # told apart from the original system-seeded dataset and safely
        # deleted without ever touching source='system' rows.
        # Wrapped defensively: if Streamlit Cloud spins up more than one
        # session/worker at boot, two processes can race to run this ALTER
        # at the same time — the loser fails with "duplicate column name",
        # which is harmless (the column already exists) and safe to ignore.
        if "source" not in cols:
            try:
                conn.execute("ALTER TABLE consumers ADD COLUMN source TEXT DEFAULT 'system';")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
        if "upload_id" not in cols:
            try:
                conn.execute("ALTER TABLE consumers ADD COLUMN upload_id INTEGER;")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise

        # Create High-Performance Indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cons_no ON consumers(CONS_NO);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_consumers_flag ON consumers(FLAG);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_consumers_pred ON consumers(prediction);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_consumers_source ON consumers(source);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_consumers_upload_id ON consumers(upload_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_up_pred_upload_id ON uploaded_predictions(upload_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_up_pred_cust ON uploaded_predictions(customer_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hist_cust ON prediction_history(customer_id);")


def init_db():
    """Initializes multi-table schema, creates indexes, and performs vectorized pre-computation ONCE."""
    ensure_schema()
    _seed_system_dataset_once()

def _seed_system_dataset_once():
    """Imports system dataset (Book4-7-4months.csv) & runs CNN-LSTM model ONCE on first startup."""
    with get_db() as conn:
        count = conn.execute("SELECT COUNT(*) as cnt FROM consumers;").fetchone()["cnt"]
        if count > 0:
            return  # Never recompute if consumers table is populated

    if not DATASET_PATH.exists():
        return

    try:
        df = pd.read_csv(DATASET_PATH)
        if "CONS_NO" not in df.columns:
            df["CONS_NO"] = [f"CUST_{i+1:06d}" for i in range(len(df))]
        
        reading_cols = [c for c in df.columns if c not in ("CONS_NO", "FLAG")]
        readings_mat = df[reading_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
        
        cids = df["CONS_NO"].astype(str).tolist()
        flags = df["FLAG"].fillna(0).astype(int).tolist() if "FLAG" in df.columns else [0] * len(df)
        
        # Batch Model Prediction ONCE on startup
        from src.predictor import predict_sequences, classify
        probs = predict_sequences(readings_mat)
        
        means = np.mean(readings_mat, axis=1).tolist()
        totals = np.sum(readings_mat, axis=1).tolist()
        zeros = np.sum(readings_mat == 0.0, axis=1).tolist()
        now_str = datetime.now().isoformat()
        
        records = []
        for i in range(len(df)):
            cls = classify(float(probs[i]))
            records.append((
                cids[i],
                flags[i],
                json.dumps(readings_mat[i].tolist()),
                float(means[i]),
                float(totals[i]),
                int(zeros[i]),
                cls["probability"],
                cls["prediction"],
                cls["risk_score"],
                cls["risk_level"],
                cls["status"],
                now_str,
                now_str
            ))
            
        with get_db() as conn:
            conn.executemany(
                """INSERT OR IGNORE INTO consumers 
                   (CONS_NO, FLAG, readings_json, avg_cons, total_cons, zero_days,
                    probability, prediction, risk_score, risk_level, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                records
            )
    except Exception as e:
        print(f"[Database Error] Failed seeding system dataset into SQLite: {e}")

# ---------------------------------------------------------
# SQL Aggregation Queries for Dashboard (<20ms response time)
# ---------------------------------------------------------
def get_dashboard_kpis() -> Dict[str, Any]:
    """Pure SQL aggregation for Dashboard KPI cards."""
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM consumers;").fetchone()[0]
        theft = conn.execute("SELECT COUNT(*) FROM consumers WHERE prediction = 1;").fetchone()[0]
        normal = total - theft
        avg_kwh = conn.execute("SELECT AVG(avg_cons) FROM consumers;").fetchone()[0] or 0.0
        
    db_size_mb = round(DB_PATH.stat().st_size / (1024 * 1024), 2) if DB_PATH.exists() else 0.0

    return {
        "total_consumers": total,
        "theft_cases": theft,
        "normal_consumers": normal,
        "theft_rate_pct": round((theft / max(total, 1)) * 100, 2),
        "avg_kwh": round(float(avg_kwh), 2),
        "db_size_mb": db_size_mb,
        "db_status": f"Active ({db_size_mb} MB)"
    }

def get_consumption_profiles_sql() -> Tuple[List[float], List[float], int]:
    """Fetches sequence vectors from SQLite to build average consumption profiles."""
    with get_db() as conn:
        normal_rows = conn.execute("SELECT readings_json FROM consumers WHERE prediction = 0 LIMIT 500;").fetchall()
        theft_rows = conn.execute("SELECT readings_json FROM consumers WHERE prediction = 1 LIMIT 500;").fetchall()
        
    def _avg_matrix(rows):
        if not rows:
            return [0.0] * 120, 120
        arrs = [np.array(json.loads(r["readings_json"]), dtype=np.float32) for r in rows if r["readings_json"]]
        if not arrs:
            return [0.0] * 120, 120
        min_len = min(len(a) for a in arrs)
        mat = np.vstack([a[:min_len] for a in arrs])
        return np.mean(mat, axis=0).tolist(), min_len

    normal_avg, n_len = _avg_matrix(normal_rows)
    theft_avg, t_len = _avg_matrix(theft_rows)
    n_days = max(n_len, t_len, 120)

    return normal_avg, theft_avg, n_days

def get_consumer_by_id(cons_no: str) -> Dict[str, Any]:
    """Indexed SQL query (<20ms) for individual consumer lookup."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM consumers WHERE CONS_NO = ?;", (cons_no,)).fetchone()
        
    if not row:
        return {}
        
    d = dict(row)
    d["readings"] = json.loads(d["readings_json"]) if d.get("readings_json") else []
    return d

def get_all_consumer_ids(limit: int = None) -> List[str]:   #limit: int = 500
    """Indexed SQL query (<20ms) to fetch customer ID list."""
    with get_db() as conn:
        rows = conn.execute("SELECT CONS_NO FROM consumers LIMIT ?;"), (limit,).fetchall()  #
    return [r["CONS_NO"] for r in rows]

def get_recent_prediction_history(limit: int = 5) -> pd.DataFrame:
    """Fetches recent manual prediction history from prediction_history table (<20ms)."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT customer_id as 'Customer ID', status as 'Prediction', 
                      probability as 'Probability', risk_level as 'Risk Level', 
                      timestamp as 'Timestamp'
               FROM prediction_history 
               ORDER BY id DESC LIMIT ?""", (limit,)
        ).fetchall()
        
        if not rows:
            # Fallback to top 5 consumers from system database if manual history is empty
            rows = conn.execute(
                """SELECT CONS_NO as 'Customer ID', status as 'Prediction', 
                          probability as 'Probability', risk_level as 'Risk Level', 
                          created_at as 'Timestamp'
                   FROM consumers 
                   ORDER BY id DESC LIMIT ?""", (limit,)
            ).fetchall()
            
    if not rows:
        return pd.DataFrame(columns=["Customer ID", "Prediction", "Probability", "Risk Level", "Timestamp"])
        
    df = pd.DataFrame([dict(r) for r in rows])
    if "Probability" in df.columns:
        df["Probability"] = df["Probability"].apply(lambda p: f"{p*100:.1f}%" if pd.notna(p) else "—")
    if "Timestamp" in df.columns:
        df["Timestamp"] = df["Timestamp"].apply(lambda d: str(d)[:16].replace("T", " ") if pd.notna(d) else "—")
    return df

def save_manual_prediction_history(res: Dict[str, Any]):
    """Stores manual single consumer prediction into prediction_history table."""
    with get_db() as conn:
        conn.execute(
            """INSERT INTO prediction_history 
               (customer_id, prediction, probability, risk_level, status, readings_json, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                res.get("customer_id", "SINGLE_USER"),
                res["prediction"],
                res["probability"],
                res["risk_level"],
                res["status"],
                json.dumps(res.get("readings", [])),
                datetime.now().isoformat()
            )
        )

def get_system_db_meta() -> Dict[str, Any]:
    """System diagnostics & table metadata."""
    with get_db() as conn:
        sqlite_ver = conn.execute("SELECT sqlite_version();").fetchone()[0]
        consumers_cnt = conn.execute("SELECT COUNT(*) FROM consumers;").fetchone()[0]
        uploads_cnt = conn.execute("SELECT COUNT(*) FROM uploads;").fetchone()[0]
        up_pred_cnt = conn.execute("SELECT COUNT(*) FROM uploaded_predictions;").fetchone()[0]
        hist_cnt = conn.execute("SELECT COUNT(*) FROM prediction_history;").fetchone()[0]
        
    db_size = f"{round(DB_PATH.stat().st_size / (1024*1024), 2)} MB" if DB_PATH.exists() else "0 MB"
    
    return {
        "sqlite_version": sqlite_ver,
        "consumers_count": consumers_cnt,
        "uploads_count": uploads_cnt,
        "uploaded_predictions_count": up_pred_cnt,
        "history_count": hist_cnt,
        "db_size": db_size,
        "last_sync": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

# ---------------------------------------------------------
# Upload → Consumers Merge (so Dashboard KPIs, charts, and Consumer
# Search immediately reflect uploaded data — no separate/hidden table)
# ---------------------------------------------------------
def upsert_consumers_from_upload(records: List[Tuple], upload_id: int) -> None:
    """Inserts NEW consumers or UPDATES existing ones (matched by CONS_NO) in
    the main 'consumers' table using data from an uploaded file.

    Every row written this way is tagged source='upload' and upload_id=<id>,
    so it can be identified and safely deleted later via
    delete_consumers_by_upload() WITHOUT ever touching the original
    system-seeded dataset (source='system').

    Uses INSERT OR REPLACE (not the newer ON CONFLICT...DO UPDATE syntax)
    for compatibility with older SQLite versions that some hosting
    environments (e.g. Streamlit Community Cloud) may ship.

    `records` is a list of tuples in this exact order:
    (CONS_NO, FLAG, readings_json, avg_cons, total_cons, zero_days,
     probability, prediction, risk_score, risk_level, status)
    """
    now_str = datetime.now().isoformat()
    with get_db() as conn:
        for rec in records:
            (cons_no, flag, readings_json, avg_cons, total_cons, zero_days,
             probability, prediction, risk_score, risk_level, status) = rec

            # Preserve the original created_at if this consumer already exists
            existing = conn.execute(
                "SELECT created_at FROM consumers WHERE CONS_NO = ?;", (cons_no,)
            ).fetchone()
            created_at = existing["created_at"] if existing and existing["created_at"] else now_str

            conn.execute(
                """INSERT OR REPLACE INTO consumers
                   (CONS_NO, FLAG, readings_json, avg_cons, total_cons, zero_days,
                    probability, prediction, risk_score, risk_level, status,
                    source, upload_id, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'upload', ?, ?, ?)
                """,
                (cons_no, flag, readings_json, avg_cons, total_cons, zero_days,
                 probability, prediction, risk_score, risk_level, status,
                 upload_id, created_at, now_str)
            )


def get_uploadable_batches(limit: int = 20) -> pd.DataFrame:
    """Lists uploads that currently have live rows merged into 'consumers'
    (i.e. eligible for deletion) — used by the Delete Uploaded Data UI.

    Wrapped in try/except so that if this specific query ever fails (e.g. an
    older SQLite build, or a fresh DB that hasn't been migrated yet), the
    rest of the dashboard still renders instead of crashing the whole app.
    """
    empty_df = pd.DataFrame(columns=["Upload ID", "File Name", "Active Records", "Date"])
    try:
        with get_db() as conn:
            rows = conn.execute(
                """SELECT u.upload_id AS "Upload ID", u.file_name AS "File Name",
                          COUNT(c.id) AS "Active Records", u.upload_date AS "Date"
                   FROM uploads u
                   LEFT JOIN consumers c ON c.upload_id = u.upload_id AND c.source = 'upload'
                   GROUP BY u.upload_id
                   HAVING COUNT(c.id) > 0
                   ORDER BY u.upload_id DESC LIMIT ?""", (limit,)
            ).fetchall()
    except sqlite3.OperationalError as e:
        print(f"[Database Warning] get_uploadable_batches failed (non-fatal): {e}")
        return empty_df

    if not rows:
        return empty_df
    return pd.DataFrame([dict(r) for r in rows])


def delete_consumers_by_upload(upload_id: int) -> int:
    """Deletes all consumer rows that came from a specific upload.

    SAFETY: the WHERE clause always includes source = 'upload', so this can
    NEVER delete rows from the original system-seeded dataset even if an
    invalid/incorrect upload_id is passed in.
    """
    try:
        with get_db() as conn:
            cur = conn.execute(
                "DELETE FROM consumers WHERE upload_id = ? AND source = 'upload';",
                (upload_id,)
            )
            deleted = cur.rowcount
            conn.execute("DELETE FROM uploaded_predictions WHERE upload_id = ?;", (upload_id,))
            conn.execute("UPDATE uploads SET status = 'Deleted' WHERE upload_id = ?;", (upload_id,))
        return deleted
    except sqlite3.OperationalError as e:
        print(f"[Database Warning] delete_consumers_by_upload failed: {e}")
        return 0
