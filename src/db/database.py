import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

DB_PATH = Path("roadIQ.db")
INCIDENT_COLUMNS = {
    "status": "TEXT NOT NULL DEFAULT 'reported'",
    "status_updated_at": "TEXT",
    "notes": "TEXT NOT NULL DEFAULT ''",
}


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            timestamp TEXT,
            detections TEXT,
            status TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            risk_score REAL NOT NULL,
            danger_zone TEXT NOT NULL,
            maintenance_eta TEXT NOT NULL,
            maintenance_priority TEXT NOT NULL,
            road_health REAL NOT NULL,
            density REAL NOT NULL,
            alert_level TEXT NOT NULL,
            total_detections INTEGER NOT NULL,
            high_priority_count INTEGER NOT NULL,
            medium_priority_count INTEGER NOT NULL,
            low_priority_count INTEGER NOT NULL,
            inference_time REAL NOT NULL,
            dispatch_score REAL,
            action TEXT,
            decision_level TEXT,
            road_state TEXT,
            status TEXT NOT NULL DEFAULT 'reported',
            status_updated_at TEXT,
            notes TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS incident_detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            incident_id INTEGER NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            x1 REAL NOT NULL,
            y1 REAL NOT NULL,
            x2 REAL NOT NULL,
            y2 REAL NOT NULL,
            severity REAL NOT NULL,
            priority TEXT NOT NULL,
            FOREIGN KEY (incident_id) REFERENCES incidents (id) ON DELETE CASCADE
        )
        """
    )

    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_incidents_created_at
        ON incidents (created_at DESC)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_incidents_geo
        ON incidents (lat, lon)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_incident_detections_incident_id
        ON incident_detections (incident_id)
        """
    )

    _ensure_incident_columns(cur)

    conn.commit()
    conn.close()


def _ensure_incident_columns(cur: sqlite3.Cursor) -> None:
    cur.execute("PRAGMA table_info(incidents)")
    existing_columns = {row[1] for row in cur.fetchall()}

    for column_name, column_def in INCIDENT_COLUMNS.items():
        if column_name not in existing_columns:
            cur.execute(
                f"ALTER TABLE incidents ADD COLUMN {column_name} {column_def}"
            )


def save_incident(
    *,
    filename: str,
    lat: float,
    lon: float,
    risk_score: float,
    danger_zone: str,
    maintenance_eta: str,
    maintenance_priority: str,
    road_health: float,
    density: float,
    alert_level: str,
    total_detections: int,
    high_priority_count: int,
    medium_priority_count: int,
    low_priority_count: int,
    inference_time: float,
    detections: List[Dict[str, Any]],
    decision: Dict[str, Any] | None = None,
    status: str = "reported",
    notes: str = "",
) -> int:
    conn = get_connection()
    cur = conn.cursor()
    created_at = datetime.utcnow().isoformat()
    status_updated_at = created_at

    cur.execute(
        """
        INSERT INTO incidents (
            filename,
            lat,
            lon,
            risk_score,
            danger_zone,
            maintenance_eta,
            maintenance_priority,
            road_health,
            density,
            alert_level,
            total_detections,
            high_priority_count,
            medium_priority_count,
            low_priority_count,
            inference_time,
            dispatch_score,
            action,
            decision_level,
            road_state,
            status,
            status_updated_at,
            notes,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            filename,
            lat,
            lon,
            risk_score,
            danger_zone,
            maintenance_eta,
            maintenance_priority,
            road_health,
            density,
            alert_level,
            total_detections,
            high_priority_count,
            medium_priority_count,
            low_priority_count,
            inference_time,
            None if decision is None else decision.get("dispatch_score"),
            None if decision is None else decision.get("action"),
            None if decision is None else decision.get("decision_level"),
            None if decision is None else decision.get("road_state"),
            status,
            status_updated_at,
            notes,
            created_at,
        ),
    )

    incident_id = cur.lastrowid

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cur.execute(
            """
            INSERT INTO incident_detections (
                incident_id,
                label,
                confidence,
                x1,
                y1,
                x2,
                y2,
                severity,
                priority
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                incident_id,
                det["label"],
                det["confidence"],
                x1,
                y1,
                x2,
                y2,
                det["severity"],
                det["priority"],
            ),
        )

    conn.commit()
    conn.close()
    return incident_id


def _fetch_detections_by_incident_ids(
    conn: sqlite3.Connection, incident_ids: List[int]
) -> Dict[int, List[Dict[str, Any]]]:
    if not incident_ids:
        return {}

    placeholders = ", ".join(["?"] * len(incident_ids))
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT
            id,
            incident_id,
            label,
            confidence,
            x1,
            y1,
            x2,
            y2,
            severity,
            priority
        FROM incident_detections
        WHERE incident_id IN ({placeholders})
        ORDER BY id ASC
        """,
        incident_ids,
    )

    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for row in cur.fetchall():
        incident_id = row["incident_id"]
        grouped.setdefault(incident_id, []).append(
            {
                "id": row["id"],
                "label": row["label"],
                "confidence": row["confidence"],
                "bbox": [row["x1"], row["y1"], row["x2"], row["y2"]],
                "severity": row["severity"],
                "priority": row["priority"],
            }
        )
    return grouped


def fetch_incidents(limit: int = 100) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT *
        FROM incidents
        ORDER BY datetime(created_at) DESC, id DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cur.fetchall()
    detections_by_incident = _fetch_detections_by_incident_ids(
        conn, [row["id"] for row in rows]
    )
    conn.close()

    incidents: List[Dict[str, Any]] = []
    for row in rows:
        incident = dict(row)
        incident["detections"] = detections_by_incident.get(row["id"], [])
        incidents.append(incident)
    return incidents


def fetch_incidents_filtered(
    *,
    limit: int = 100,
    status: str | None = None,
    alert_level: str | None = None,
) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()

    conditions = []
    params: List[Any] = []

    if status:
        conditions.append("lower(status) = lower(?)")
        params.append(status)

    if alert_level:
        conditions.append("upper(alert_level) = upper(?)")
        params.append(alert_level)

    where_sql = ""
    if conditions:
        where_sql = "WHERE " + " AND ".join(conditions)

    params.append(limit)
    cur.execute(
        f"""
        SELECT *
        FROM incidents
        {where_sql}
        ORDER BY datetime(created_at) DESC, id DESC
        LIMIT ?
        """,
        params,
    )

    rows = cur.fetchall()
    detections_by_incident = _fetch_detections_by_incident_ids(
        conn, [row["id"] for row in rows]
    )
    conn.close()

    incidents: List[Dict[str, Any]] = []
    for row in rows:
        incident = dict(row)
        incident["detections"] = detections_by_incident.get(row["id"], [])
        incidents.append(incident)
    return incidents


def fetch_incident_by_id(incident_id: int) -> Dict[str, Any] | None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT *
        FROM incidents
        WHERE id = ?
        """,
        (incident_id,),
    )
    row = cur.fetchone()
    if row is None:
        conn.close()
        return None

    detections_by_incident = _fetch_detections_by_incident_ids(conn, [incident_id])
    conn.close()

    incident = dict(row)
    incident["detections"] = detections_by_incident.get(incident_id, [])
    return incident


def update_incident_status(
    incident_id: int, status: str, notes: str | None = None
) -> Dict[str, Any] | None:
    conn = get_connection()
    cur = conn.cursor()
    status_updated_at = datetime.utcnow().isoformat()

    if notes is None:
        cur.execute(
            """
            UPDATE incidents
            SET status = ?, status_updated_at = ?
            WHERE id = ?
            """,
            (status, status_updated_at, incident_id),
        )
    else:
        cur.execute(
            """
            UPDATE incidents
            SET status = ?, status_updated_at = ?, notes = ?
            WHERE id = ?
            """,
            (status, status_updated_at, notes, incident_id),
        )

    conn.commit()
    conn.close()

    return fetch_incident_by_id(incident_id)


def fetch_geo_points(limit: int = 500) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT lat, lon, risk_score
        FROM incidents
        ORDER BY datetime(created_at) DESC, id DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()

    return [
        {"lat": row["lat"], "lon": row["lon"], "risk": row["risk_score"]}
        for row in rows
    ]


def fetch_analytics_summary() -> Dict[str, Any]:
    return fetch_analytics_summary_filtered(days=None)


def _build_created_at_filter(days: int | None) -> tuple[str, list[Any]]:
    if days is None:
        return "", []
    return "WHERE datetime(created_at) >= datetime('now', ?)", [f"-{days} days"]


def fetch_analytics_summary_filtered(days: int | None = None) -> Dict[str, Any]:
    conn = get_connection()
    cur = conn.cursor()
    where_sql, params = _build_created_at_filter(days)
    cur.execute(
        f"""
        SELECT
            COUNT(*) AS total_incidents,
            COALESCE(AVG(risk_score), 0) AS avg_risk_score,
            COALESCE(AVG(road_health), 0) AS avg_road_health,
            COALESCE(MAX(risk_score), 0) AS max_risk_score,
            COALESCE(SUM(total_detections), 0) AS total_detections
        FROM incidents
        {where_sql}
        """,
        params,
    )
    overall = dict(cur.fetchone())

    where_sql, params = _build_created_at_filter(days)
    cur.execute(
        f"""
        SELECT alert_level, COUNT(*) AS count
        FROM incidents
        {where_sql}
        GROUP BY alert_level
        ORDER BY count DESC
        """,
        params,
    )
    alert_breakdown = [dict(row) for row in cur.fetchall()]

    where_sql, params = _build_created_at_filter(days)
    cur.execute(
        f"""
        SELECT status, COUNT(*) AS count
        FROM incidents
        {where_sql}
        GROUP BY status
        ORDER BY count DESC
        """,
        params,
    )
    status_breakdown = [dict(row) for row in cur.fetchall()]
    conn.close()

    return {
        "total_incidents": overall["total_incidents"],
        "avg_risk_score": round(overall["avg_risk_score"], 2),
        "avg_road_health": round(overall["avg_road_health"], 2),
        "max_risk_score": round(overall["max_risk_score"], 2),
        "total_detections": overall["total_detections"],
        "alert_breakdown": alert_breakdown,
        "status_breakdown": status_breakdown,
    }


def fetch_incident_trends(limit: int = 30, days: int | None = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    where_sql, params = _build_created_at_filter(days)
    params.append(limit)
    cur.execute(
        f"""
        SELECT
            date(created_at) AS day,
            COUNT(*) AS incident_count,
            ROUND(AVG(risk_score), 2) AS avg_risk_score,
            ROUND(AVG(road_health), 2) AS avg_road_health,
            SUM(total_detections) AS total_detections
        FROM incidents
        {where_sql}
        GROUP BY date(created_at)
        ORDER BY day DESC
        LIMIT ?
        """,
        params,
    )
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return list(reversed(rows))


def fetch_damage_type_analytics(days: int | None = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    where_sql = ""
    params: list[Any] = []
    if days is not None:
        where_sql = "WHERE datetime(i.created_at) >= datetime('now', ?)"
        params.append(f"-{days} days")
    cur.execute(
        f"""
        SELECT
            d.label,
            COUNT(*) AS detection_count,
            ROUND(AVG(d.severity), 2) AS avg_severity,
            ROUND(AVG(d.confidence), 4) AS avg_confidence
        FROM incident_detections d
        INNER JOIN incidents i ON i.id = d.incident_id
        {where_sql}
        GROUP BY d.label
        ORDER BY detection_count DESC, label ASC
        """,
        params,
    )
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def fetch_top_risk_locations(limit: int = 5, days: int | None = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    where_sql, params = _build_created_at_filter(days)
    params.append(limit)
    cur.execute(
        f"""
        SELECT
            ROUND(lat, 4) AS lat,
            ROUND(lon, 4) AS lon,
            COUNT(*) AS incident_count,
            ROUND(AVG(risk_score), 2) AS avg_risk_score,
            MAX(alert_level) AS latest_alert_level,
            MAX(created_at) AS latest_seen_at
        FROM incidents
        {where_sql}
        GROUP BY ROUND(lat, 4), ROUND(lon, 4)
        ORDER BY avg_risk_score DESC, incident_count DESC
        LIMIT ?
        """,
        params,
    )
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def fetch_geo_hotspot_points(days: int | None = None, limit: int = 500) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    where_sql, params = _build_created_at_filter(days)
    params.append(limit)
    cur.execute(
        f"""
        SELECT
            lat,
            lon,
            risk_score,
            alert_level,
            total_detections,
            created_at
        FROM incidents
        {where_sql}
        ORDER BY datetime(created_at) DESC, id DESC
        LIMIT ?
        """,
        params,
    )
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows
