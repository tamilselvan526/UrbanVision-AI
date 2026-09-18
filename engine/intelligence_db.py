"""
Intelligence Database Module
=============================
SQLite database for storing:
  1. Watchlist Persons (target memory, reference embeddings, photos)
  2. Person Sightings (timestamp, camera, coordinates, match confidence)
  3. Vehicle & License Plate Records (plate number, vehicle type, direction, speed, timeline, snapshots)
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

DB_PATH = Path(__file__).parent.parent / "intelligence.db"
SNAPSHOTS_DIR = Path(__file__).parent.parent / "snapshots"
SNAPSHOTS_DIR.mkdir(exist_ok=True)
(SNAPSHOTS_DIR / "persons").mkdir(exist_ok=True)
(SNAPSHOTS_DIR / "vehicles").mkdir(exist_ok=True)
(SNAPSHOTS_DIR / "regulars").mkdir(exist_ok=True)


class IntelligenceDB:
    """Thread-safe SQLite client for surveillance intelligence records."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = str(db_path)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._get_conn() as conn:
            cur = conn.cursor()
            # 1. Watchlist Persons
            cur.execute("""
                CREATE TABLE IF NOT EXISTS watchlist_persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    notes TEXT DEFAULT '',
                    photo_filename TEXT DEFAULT '',
                    embedding BLOB NOT NULL,
                    created_at REAL NOT NULL
                )
            """)

            # 2. Person Sightings
            cur.execute("""
                CREATE TABLE IF NOT EXISTS person_sightings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_name TEXT NOT NULL,
                    camera_source TEXT DEFAULT 'Camera 01',
                    timestamp REAL NOT NULL,
                    timestamp_str TEXT NOT NULL,
                    frame_id INTEGER DEFAULT 0,
                    x INTEGER NOT NULL,
                    y INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    snapshot_filename TEXT DEFAULT ''
                )
            """)

            # 3. Vehicle & Plate Records
            cur.execute("""
                CREATE TABLE IF NOT EXISTS vehicle_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_number TEXT NOT NULL,
                    vehicle_type TEXT DEFAULT 'car',
                    camera_source TEXT DEFAULT 'Camera 01',
                    first_seen REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    first_seen_str TEXT NOT NULL,
                    last_seen_str TEXT NOT NULL,
                    trajectory_json TEXT DEFAULT '[]',
                    direction_deg REAL DEFAULT 0.0,
                    avg_speed REAL DEFAULT 0.0,
                    sightings_count INTEGER DEFAULT 1,
                    snapshot_filename TEXT DEFAULT ''
                )
            """)

            # 4. Street Regulars (Recurring Entities & Roles)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS street_regulars (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alias TEXT NOT NULL UNIQUE,
                    custom_name TEXT DEFAULT '',
                    detected_role TEXT DEFAULT 'UNKNOWN',
                    role_confidence REAL DEFAULT 0.0,
                    days_seen_count INTEGER DEFAULT 1,
                    total_appearances INTEGER DEFAULT 1,
                    total_dwell_seconds REAL DEFAULT 0.0,
                    anchor_x INTEGER DEFAULT 0,
                    anchor_y INTEGER DEFAULT 0,
                    active_hours_json TEXT DEFAULT '[]',
                    first_seen REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    first_seen_str TEXT NOT NULL,
                    last_seen_str TEXT NOT NULL,
                    snapshot_filename TEXT DEFAULT '',
                    embedding BLOB NOT NULL
                )
            """)

            # Create indices for fast searches
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sightings_name ON person_sightings(person_name)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_vehicle_plate ON vehicle_records(plate_number)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_regular_alias ON street_regulars(alias)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_regular_role ON street_regulars(detected_role)")
            conn.commit()
            logger.info(f"[IntelligenceDB] Initialized database at {self.db_path}")

    # ── Watchlist Person Methods ───────────────────────────────────────────

    def add_watchlist_person(
        self,
        name: str,
        embedding: np.ndarray,
        notes: str = "",
        photo_filename: str = "",
    ) -> int:
        """Enroll a new person of interest with their visual embedding."""
        emb_bytes = embedding.astype(np.float32).tobytes()
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO watchlist_persons (name, notes, photo_filename, embedding, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (name.strip(), notes.strip(), photo_filename, emb_bytes, time.time()))
            conn.commit()
            return cur.lastrowid

    def get_watchlist(self) -> List[Dict[str, Any]]:
        """Retrieve all enrolled watchlist persons."""
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, name, notes, photo_filename, created_at FROM watchlist_persons ORDER BY id DESC")
            rows = cur.fetchall()
            return [dict(r) for r in rows]

    def get_all_embeddings(self) -> List[Tuple[int, str, np.ndarray]]:
        """Load all enrolled embeddings for real-time live comparison."""
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, name, embedding FROM watchlist_persons")
            results = []
            for row in cur.fetchall():
                emb = np.frombuffer(row["embedding"], dtype=np.float32)
                results.append((row["id"], row["name"], emb))
            return results

    def delete_watchlist_person(self, person_id: int) -> bool:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM watchlist_persons WHERE id = ?", (person_id,))
            conn.commit()
            return cur.rowcount > 0

    # ── Person Sightings ───────────────────────────────────────────────────

    def record_sighting(
        self,
        person_name: str,
        x: int,
        y: int,
        confidence: float,
        camera_source: str = "Camera 01",
        frame_id: int = 0,
        snapshot_filename: str = "",
    ):
        """Record when an enrolled person is identified in a feed."""
        now = time.time()
        now_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO person_sightings
                (person_name, camera_source, timestamp, timestamp_str, frame_id, x, y, confidence, snapshot_filename)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (person_name, camera_source, now, now_str, frame_id, x, y, round(confidence, 2), snapshot_filename))
            conn.commit()

    def get_sightings_for_person(self, person_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT * FROM person_sightings
                WHERE person_name LIKE ?
                ORDER BY timestamp DESC LIMIT ?
            """, (f"%{person_name}%", limit))
            return [dict(r) for r in cur.fetchall()]

    def get_recent_sightings(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM person_sightings ORDER BY timestamp DESC LIMIT ?", (limit,))
            return [dict(r) for r in cur.fetchall()]

    # ── Vehicle & License Plate Records ────────────────────────────────────

    def record_or_update_vehicle(
        self,
        plate_number: str,
        vehicle_type: str = "car",
        camera_source: str = "Camera 01",
        point: Optional[Tuple[int, int]] = None,
        direction_deg: float = 0.0,
        speed: float = 0.0,
        snapshot_filename: str = "",
    ):
        """Record a license plate sighting or update existing vehicle trail."""
        plate = plate_number.upper().strip()
        if not plate or len(plate) < 3:
            return

        now = time.time()
        now_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))

        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM vehicle_records WHERE plate_number = ? ORDER BY id DESC LIMIT 1", (plate,))
            existing = cur.fetchone()

            if existing:
                # Update recent vehicle record if seen within last 10 minutes
                if now - existing["last_seen"] < 600:
                    try:
                        traj = json.loads(existing["trajectory_json"] or "[]")
                    except Exception:
                        traj = []
                    if point:
                        traj.append(point)
                    if len(traj) > 60:
                        traj = traj[-60:]

                    snap = snapshot_filename or existing["snapshot_filename"]
                    cur.execute("""
                        UPDATE vehicle_records
                        SET last_seen = ?, last_seen_str = ?, trajectory_json = ?,
                            direction_deg = ?, avg_speed = ?, sightings_count = sightings_count + 1,
                            snapshot_filename = ?
                        WHERE id = ?
                    """, (now, now_str, json.dumps(traj), direction_deg, speed, snap, existing["id"]))
                    conn.commit()
                    return

            # Insert fresh vehicle record
            traj = [point] if point else []
            cur.execute("""
                INSERT INTO vehicle_records
                (plate_number, vehicle_type, camera_source, first_seen, last_seen, first_seen_str,
                 last_seen_str, trajectory_json, direction_deg, avg_speed, sightings_count, snapshot_filename)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            """, (plate, vehicle_type, camera_source, now, now, now_str, now_str, json.dumps(traj),
                  direction_deg, speed, snapshot_filename))
            conn.commit()

    def search_vehicles(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search vehicles by partial or exact plate number."""
        query = query.upper().strip()
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT * FROM vehicle_records
                WHERE plate_number LIKE ?
                ORDER BY last_seen DESC LIMIT ?
            """, (f"%{query}%", limit))
            rows = [dict(r) for r in cur.fetchall()]
            for r in rows:
                try:
                    r["trajectory"] = json.loads(r.get("trajectory_json", "[]"))
                except Exception:
                    r["trajectory"] = []
            return rows

    def get_recent_vehicles(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get latest vehicle sightings."""
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM vehicle_records ORDER BY last_seen DESC LIMIT ?", (limit,))
            rows = [dict(r) for r in cur.fetchall()]
            for r in rows:
                try:
                    r["trajectory"] = json.loads(r.get("trajectory_json", "[]"))
                except Exception:
                    r["trajectory"] = []
            return rows

    # ── Street Regulars Methods ────────────────────────────────────────────

    def add_or_update_regular(
        self,
        alias: str,
        embedding: np.ndarray,
        detected_role: str = "UNKNOWN",
        role_confidence: float = 0.0,
        dwell_delta: float = 0.0,
        anchor: Tuple[int, int] = (0, 0),
        active_hour: Optional[int] = None,
        snapshot_filename: Optional[str] = None,
        custom_name: Optional[str] = None,
    ) -> int:
        """Add a new street regular or update existing regular's stats and moving average embedding."""
        now = time.time()
        now_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
        emb_blob = embedding.astype(np.float32).tobytes()

        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM street_regulars WHERE alias = ?", (alias,))
            existing = cur.fetchone()

            if existing:
                # Parse active hours
                try:
                    active_hours = json.loads(existing["active_hours_json"] or "[]")
                except Exception:
                    active_hours = []
                if active_hour is not None and active_hour not in active_hours:
                    active_hours.append(active_hour)
                    active_hours.sort()

                # Moving average embedding update: new_emb = 0.8 * old + 0.2 * new
                old_emb = np.frombuffer(existing["embedding"], dtype=np.float32)
                updated_emb = 0.8 * old_emb + 0.2 * embedding.astype(np.float32)
                norm = np.linalg.norm(updated_emb)
                if norm > 1e-6:
                    updated_emb /= norm
                updated_blob = updated_emb.astype(np.float32).tobytes()

                # Check if calendar day differs for days_seen_count
                old_last_seen = existing["last_seen"]
                old_day = time.strftime("%Y-%m-%d", time.localtime(old_last_seen))
                curr_day = time.strftime("%Y-%m-%d", time.localtime(now))
                day_increment = 1 if curr_day != old_day else 0

                snap = snapshot_filename or existing["snapshot_filename"]
                c_name = custom_name if custom_name is not None else existing["custom_name"]

                # Anchor point moving average
                new_anchor_x = int(0.7 * existing["anchor_x"] + 0.3 * anchor[0]) if existing["anchor_x"] > 0 else anchor[0]
                new_anchor_y = int(0.7 * existing["anchor_y"] + 0.3 * anchor[1]) if existing["anchor_y"] > 0 else anchor[1]

                cur.execute("""
                    UPDATE street_regulars
                    SET custom_name = ?, detected_role = ?, role_confidence = ?,
                        days_seen_count = days_seen_count + ?,
                        total_appearances = total_appearances + 1,
                        total_dwell_seconds = total_dwell_seconds + ?,
                        anchor_x = ?, anchor_y = ?,
                        active_hours_json = ?,
                        last_seen = ?, last_seen_str = ?,
                        snapshot_filename = ?, embedding = ?
                    WHERE id = ?
                """, (
                    c_name, detected_role, role_confidence,
                    day_increment, dwell_delta,
                    new_anchor_x, new_anchor_y,
                    json.dumps(active_hours),
                    now, now_str, snap, updated_blob, existing["id"]
                ))
                conn.commit()
                return existing["id"]
            else:
                active_hours = [active_hour] if active_hour is not None else []
                cur.execute("""
                    INSERT INTO street_regulars
                    (alias, custom_name, detected_role, role_confidence, days_seen_count,
                     total_appearances, total_dwell_seconds, anchor_x, anchor_y,
                     active_hours_json, first_seen, last_seen, first_seen_str, last_seen_str,
                     snapshot_filename, embedding)
                    VALUES (?, ?, ?, ?, 1, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alias, custom_name or "", detected_role, role_confidence,
                    dwell_delta, anchor[0], anchor[1],
                    json.dumps(active_hours), now, now, now_str, now_str,
                    snapshot_filename or "", emb_blob
                ))
                conn.commit()
                return cur.lastrowid

    def get_all_regulars(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch all discovered regulars."""
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, alias, custom_name, detected_role, role_confidence,
                       days_seen_count, total_appearances, total_dwell_seconds,
                       anchor_x, anchor_y, active_hours_json,
                       first_seen, last_seen, first_seen_str, last_seen_str,
                       snapshot_filename
                FROM street_regulars
                ORDER BY total_appearances DESC, last_seen DESC
                LIMIT ?
            """, (limit,))
            rows = [dict(r) for r in cur.fetchall()]
            for r in rows:
                try:
                    r["active_hours"] = json.loads(r.get("active_hours_json", "[]"))
                except Exception:
                    r["active_hours"] = []
            return rows

    def get_regular_by_id(self, regular_id: int) -> Optional[Dict[str, Any]]:
        """Fetch regular by ID."""
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, alias, custom_name, detected_role, role_confidence,
                       days_seen_count, total_appearances, total_dwell_seconds,
                       anchor_x, anchor_y, active_hours_json,
                       first_seen, last_seen, first_seen_str, last_seen_str,
                       snapshot_filename
                FROM street_regulars WHERE id = ?
            """, (regular_id,))
            row = cur.fetchone()
            if not row:
                return None
            res = dict(row)
            try:
                res["active_hours"] = json.loads(res.get("active_hours_json", "[]"))
            except Exception:
                res["active_hours"] = []
            return res

    def update_regular_custom_name(self, regular_id: int, custom_name: str) -> bool:
        """Update user-assigned name for a regular."""
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("UPDATE street_regulars SET custom_name = ? WHERE id = ?", (custom_name.strip(), regular_id))
            conn.commit()
            return cur.rowcount > 0

    def delete_regular(self, regular_id: int) -> bool:
        """Delete regular by ID."""
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM street_regulars WHERE id = ?", (regular_id,))
            conn.commit()
            return cur.rowcount > 0

    def load_all_regular_embeddings(self) -> List[Dict[str, Any]]:
        """Load all regular embeddings into memory for fast cosine similarity matching."""
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, alias, custom_name, detected_role, role_confidence, embedding FROM street_regulars")
            records = []
            for r in cur.fetchall():
                emb = np.frombuffer(r["embedding"], dtype=np.float32)
                records.append({
                    "id": r["id"],
                    "alias": r["alias"],
                    "custom_name": r["custom_name"],
                    "detected_role": r["detected_role"],
                    "role_confidence": r["role_confidence"],
                    "embedding": emb,
                })
            return records


# Global singleton instance
intelligence_db = IntelligenceDB()
