"""
Street Entity Profiler & Routine Activity Discovery Engine
===========================================================
Automatically discovers recurring street regulars (vendors, shopkeepers, commuters,
delivery workers, loiterers) without manual pre-enrollment in dense, overcrowded urban streets.
Features:
  - Upper-body weighted Re-ID embedding clustering (immune to lower-body crowd occlusions).
  - Spatial dwell time, anchor stationarity, and kinematic pose gesture analysis.
  - Autonomous role/job classification:
      STREET_VENDOR, SHOPKEEPER, COMMUTER, DELIVERY_WORKER, SUSPICIOUS_LOITERER.
  - Auto-promotion from candidate to verified regular and persistence into SQLite DB.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from engine.intelligence_db import intelligence_db, SNAPSHOTS_DIR
from engine.person_reid import person_reid
from engine.pose_estimator import PoseResult

REGULARS_SNAPSHOT_DIR = SNAPSHOTS_DIR / "regulars"
REGULARS_SNAPSHOT_DIR.mkdir(exist_ok=True)

# Cosine similarity threshold to match an existing regular
SIMILARITY_REGULAR_MATCH = 0.74

# Minimum frames a track must persist before qualifying as a candidate regular
MIN_FRAMES_TO_QUALIFY = 40

# Radius in pixels within which an entity is considered stationary at their anchor
STATIONARY_RADIUS_PX = 85.0


@dataclass
class TrackSession:
    """Active surveillance session for a single tracked person."""
    track_id: int
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    frame_count: int = 0
    positions: deque = field(default_factory=lambda: deque(maxlen=120))  # (x, y, time)
    anchor_x: int = 0
    anchor_y: int = 0
    stationary_frames: int = 0
    upper_gesture_frames: int = 0
    last_wrist_pos: Optional[Tuple[float, float]] = None
    embedding_accumulator: Optional[np.ndarray] = None
    embedding_samples: int = 0
    best_crop: Optional[np.ndarray] = None
    best_crop_score: float = 0.0

    # Discovered regular association
    regular_id: Optional[int] = None
    regular_alias: Optional[str] = None
    custom_name: Optional[str] = None
    classified_role: str = "UNKNOWN"
    role_confidence: float = 0.0
    synced_to_db: bool = False
    last_db_sync: float = 0.0


class StreetEntityProfiler:
    """
    Automated discovery, spatial routine profiling, and role inference engine.
    """

    def __init__(self, match_threshold: float = SIMILARITY_REGULAR_MATCH):
        self.match_threshold = match_threshold
        # In-memory regular gallery: list of dicts {id, alias, custom_name, detected_role, role_confidence, embedding}
        self._gallery: List[Dict[str, Any]] = []
        # Active track sessions: track_id -> TrackSession
        self.active_sessions: Dict[int, TrackSession] = {}
        self.reload_regulars_from_db()

    def reload_regulars_from_db(self):
        """Load all registered regulars and their embeddings from DB."""
        self._gallery = intelligence_db.load_all_regular_embeddings()
        logger.info(f"[StreetProfiler] Loaded {len(self._gallery)} street regular profile(s) from database.")

    def _match_embedding(self, emb: np.ndarray) -> Optional[Tuple[Dict[str, Any], float]]:
        """Find best matching regular in gallery above threshold."""
        if not self._gallery or emb is None or len(emb) == 0:
            return None

        best_reg = None
        best_sim = -1.0

        for reg in self._gallery:
            target_emb = reg["embedding"]
            sim = float(np.dot(emb, target_emb))
            if sim > best_sim:
                best_sim = sim
                best_reg = reg

        if best_sim >= self.match_threshold and best_reg is not None:
            return best_reg, best_sim
        return None

    def process_person_detection(
        self,
        track_id: int,
        bbox: List[int],
        frame_bgr: np.ndarray,
        pose_res: Optional[PoseResult] = None,
        is_mannequin: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a detected person in the current frame:
        1. Update track session kinematic metrics.
        2. Extract upper-body weighted Re-ID embedding.
        3. Match against known regulars gallery or cluster into candidate.
        4. Infer role based on dwell, stationarity, speed, and pose gestures.
        5. Return regular metadata for display badge.
        """
        now = time.time()
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)

        # Get or initialize track session
        if track_id not in self.active_sessions:
            session = TrackSession(
                track_id=track_id,
                first_seen=now,
                last_seen=now,
                anchor_x=cx,
                anchor_y=cy,
            )
            self.active_sessions[track_id] = session
        else:
            session = self.active_sessions[track_id]

        session.last_seen = now
        session.frame_count += 1
        session.positions.append((cx, cy, now))

        # Check stationarity against initial anchor
        dist_from_anchor = math.hypot(cx - session.anchor_x, cy - session.anchor_y)
        if dist_from_anchor <= STATIONARY_RADIUS_PX:
            session.stationary_frames += 1
        else:
            # Shift anchor slowly if entity moved and stayed in new spot
            session.anchor_x = int(0.95 * session.anchor_x + 0.05 * cx)
            session.anchor_y = int(0.95 * session.anchor_y + 0.05 * cy)

        # Kinematic / Gesture analysis using Pose
        if pose_res and pose_res.keypoints is not None:
            # Check wrist activity (keypoints 9=left wrist, 10=right wrist)
            kpts = pose_res.keypoints
            if len(kpts) > 10:
                rw = kpts[10]  # [x, y, conf]
                if rw[2] > 0.4:
                    if session.last_wrist_pos is not None:
                        w_dist = math.hypot(rw[0] - session.last_wrist_pos[0], rw[1] - session.last_wrist_pos[1])
                        # Repetitive hand motion (e.g. preparing food, packaging, counting)
                        if w_dist > 8.0:
                            session.upper_gesture_frames += 1
                    session.last_wrist_pos = (rw[0], rw[1])

        # Extract embedding every 6 frames or on first frame
        crop = None
        if session.frame_count == 1 or session.frame_count % 6 == 0:
            ih, iw = frame_bgr.shape[:2]
            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(iw, x2), min(ih, y2)
            if cx2 > cx1 + 10 and cy2 > cy1 + 20:
                crop = frame_bgr[cy1:cy2, cx1:cx2]
                emb = person_reid.extract_embedding(crop, crowd_mode=True)

                if session.embedding_accumulator is None:
                    session.embedding_accumulator = emb.copy()
                    session.embedding_samples = 1
                else:
                    session.embedding_accumulator += emb
                    session.embedding_samples += 1

                # Update best crop candidate (prefer larger, un-clipped crops)
                crop_score = (cx2 - cx1) * (cy2 - cy1)
                if crop_score > session.best_crop_score:
                    session.best_crop = crop.copy()
                    session.best_crop_score = crop_score

        # If not yet identified with a regular, try matching gallery
        current_emb = None
        if session.embedding_accumulator is not None and session.embedding_samples > 0:
            current_emb = session.embedding_accumulator / float(session.embedding_samples)
            norm = np.linalg.norm(current_emb)
            if norm > 1e-6:
                current_emb /= norm

        if session.regular_alias is None and current_emb is not None:
            match_res = self._match_embedding(current_emb)
            if match_res is not None:
                matched_reg, sim = match_res
                session.regular_id = matched_reg["id"]
                session.regular_alias = matched_reg["alias"]
                session.custom_name = matched_reg["custom_name"]
                session.classified_role = matched_reg["detected_role"]
                session.role_confidence = matched_reg["role_confidence"]
                logger.info(
                    f"[StreetProfiler] Track #{track_id} recognized as Regular '{session.regular_alias}' "
                    f"({session.custom_name or 'No Custom Name'}) with sim={sim:.3f}"
                )

        # Role Activity Classifier
        if is_mannequin:
            session.classified_role = "DISPLAY_MANNEQUIN"
            session.role_confidence = 0.95
        else:
            self._classify_role(session)

        # Autonomous Registration & Promotion:
        # If candidate has persisted > MIN_FRAMES_TO_QUALIFY and is still unknown (and NOT a mannequin!), register as a new Regular!
        if (
            not is_mannequin
            and session.regular_alias is None
            and session.frame_count >= MIN_FRAMES_TO_QUALIFY
            and current_emb is not None
        ):
            self._register_new_regular(session, current_emb)

        # Periodic DB Sync every 5 seconds
        if not is_mannequin and session.regular_id is not None and (now - session.last_db_sync > 5.0):
            self._sync_session_to_db(session, current_emb)

        # Return status for badge overlay
        display_name = session.custom_name if session.custom_name else session.regular_alias
        return {
            "is_regular": (session.regular_alias is not None) and not is_mannequin,
            "regular_id": session.regular_id if not is_mannequin else None,
            "regular_alias": session.regular_alias if not is_mannequin else None,
            "custom_name": session.custom_name if not is_mannequin else None,
            "display_name": display_name if not is_mannequin else "DISPLAY_MANNEQUIN",
            "role": session.classified_role,
            "role_confidence": round(session.role_confidence, 2),
            "dwell_seconds": round(now - session.first_seen, 1),
            "stationarity_ratio": round(session.stationary_frames / max(1, session.frame_count), 2),
        }

    def _classify_role(self, session: TrackSession):
        """
        Classify entity role based on spatio-temporal dynamics:
        - STREET_VENDOR: High stationarity (>60%), high dwell (>12s), upper gesture activity.
        - SHOPKEEPER: High stationarity (>75%), high dwell (>15s), minimal whole-body transit.
        - COMMUTER: Low dwell (<30s), high linear speed (>16 px/s), low stationarity.
        - DELIVERY_WORKER: Fast transit speed (>20 px/s), frequent movement between locations.
        - SUSPICIOUS_LOITERER: Moderate/high dwell (>25s), low stationarity (<45%), erratic wandering.
        """
        dwell = time.time() - session.first_seen
        stat_ratio = session.stationary_frames / max(1, session.frame_count)

        # Calculate speed (px/sec) across positions
        speed = 0.0
        if len(session.positions) >= 2:
            p_first = session.positions[0]
            p_last = session.positions[-1]
            dt = p_last[2] - p_first[2]
            if dt > 0.5:
                dx = p_last[0] - p_first[0]
                dy = p_last[1] - p_first[1]
                speed = math.hypot(dx, dy) / dt

        # Vendor vs Shopkeeper
        if dwell >= 10.0 and stat_ratio >= 0.60:
            if session.upper_gesture_frames >= 4:
                session.classified_role = "STREET_VENDOR"
                session.role_confidence = min(0.95, 0.65 + (stat_ratio * 0.2) + (session.upper_gesture_frames * 0.02))
            else:
                session.classified_role = "SHOPKEEPER"
                session.role_confidence = min(0.90, 0.60 + (stat_ratio * 0.3))
        # Commuter vs Delivery Worker
        elif speed > 18.0 and dwell < 35.0:
            if speed > 28.0:
                session.classified_role = "DELIVERY_WORKER"
                session.role_confidence = min(0.88, 0.55 + (speed / 100.0))
            else:
                session.classified_role = "COMMUTER"
                session.role_confidence = min(0.92, 0.60 + (speed / 80.0))
        # Suspicious Loiterer
        elif dwell >= 25.0 and stat_ratio < 0.45:
            session.classified_role = "SUSPICIOUS_LOITERER"
            session.role_confidence = min(0.85, 0.50 + ((1.0 - stat_ratio) * 0.35))
        elif session.frame_count >= MIN_FRAMES_TO_QUALIFY:
            if session.classified_role == "UNKNOWN":
                session.classified_role = "STREET_REGULAR"
                session.role_confidence = 0.60

    def _register_new_regular(self, session: TrackSession, emb: np.ndarray):
        """Auto-register a newly discovered recurring entity."""
        # Generate unique alias based on role
        alias_prefix = {
            "STREET_VENDOR": "VND",
            "SHOPKEEPER": "SHP",
            "COMMUTER": "COM",
            "DELIVERY_WORKER": "DLV",
            "SUSPICIOUS_LOITERER": "LOIT",
        }.get(session.classified_role, "REG")

        rand_code = f"{int(time.time()) % 10000:04d}_{session.track_id:02d}"
        alias = f"{alias_prefix}_{rand_code}"

        # Save snapshot
        snapshot_filename = ""
        if session.best_crop is not None:
            snapshot_filename = f"regular_{alias}.jpg"
            snap_path = REGULARS_SNAPSHOT_DIR / snapshot_filename
            cv2.imwrite(str(snap_path), session.best_crop)

        active_hour = time.localtime().tm_hour
        dwell_seconds = time.time() - session.first_seen

        reg_id = intelligence_db.add_or_update_regular(
            alias=alias,
            embedding=emb,
            detected_role=session.classified_role,
            role_confidence=session.role_confidence,
            dwell_delta=dwell_seconds,
            anchor=(session.anchor_x, session.anchor_y),
            active_hour=active_hour,
            snapshot_filename=snapshot_filename,
        )

        session.regular_id = reg_id
        session.regular_alias = alias
        session.synced_to_db = True
        session.last_db_sync = time.time()

        # Update in-memory gallery
        self._gallery.append({
            "id": reg_id,
            "alias": alias,
            "custom_name": "",
            "detected_role": session.classified_role,
            "role_confidence": session.role_confidence,
            "embedding": emb,
        })
        logger.success(
            f"[StreetProfiler] Discovered & Auto-Enrolled New Street Regular '{alias}' "
            f"Role: {session.classified_role} (Conf: {session.role_confidence:.2f})"
        )

    def _sync_session_to_db(self, session: TrackSession, emb: Optional[np.ndarray]):
        """Update existing regular's database records with latest stats and centroid."""
        if not session.regular_alias or emb is None:
            return

        now = time.time()
        dwell_seconds = now - session.first_seen
        active_hour = time.localtime().tm_hour

        snapshot_filename = None
        if session.best_crop is not None:
            snapshot_filename = f"regular_{session.regular_alias}.jpg"
            snap_path = REGULARS_SNAPSHOT_DIR / snapshot_filename
            cv2.imwrite(str(snap_path), session.best_crop)

        intelligence_db.add_or_update_regular(
            alias=session.regular_alias,
            embedding=emb,
            detected_role=session.classified_role,
            role_confidence=session.role_confidence,
            dwell_delta=dwell_seconds,
            anchor=(session.anchor_x, session.anchor_y),
            active_hour=active_hour,
            snapshot_filename=snapshot_filename,
            custom_name=session.custom_name,
        )
        session.last_db_sync = now

    def cleanup_inactive_tracks(self, max_idle_seconds: float = 8.0):
        """Purge tracks that have left the scene."""
        now = time.time()
        to_del = [tid for tid, s in self.active_sessions.items() if now - s.last_seen > max_idle_seconds]
        for tid in to_del:
            del self.active_sessions[tid]


# Global singleton instance
street_entity_profiler = StreetEntityProfiler()
