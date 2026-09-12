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

            # Create indices for fast searches
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sightings_name ON person_sightings(person_name)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_vehicle_plate ON vehicle_records(plate_number)")
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


# Global singleton instance
intelligence_db = IntelligenceDB()
