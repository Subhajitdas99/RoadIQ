import sqlite3
import json
from datetime import datetime

DB_PATH = "roadIQ.db"

def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        timestamp TEXT,
        detections TEXT,
        status TEXT
    )
    """)

    conn.commit()
    conn.close()

def save_report(filename, detections):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO reports (filename, timestamp, detections, status)
    VALUES (?, ?, ?, ?)
    """, (
        filename,
        datetime.utcnow().isoformat(),
        json.dumps(detections),
        "Pending"
    ))

    conn.commit()
    conn.close()

def fetch_reports():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT * FROM reports ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()

    return rows
