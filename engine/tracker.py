"""
Algorithm 2: ByteTrack Multi-Object Movement Tracker
=====================================================
Implements ByteTrack-style tracking using Ultralytics' built-in tracker
(wraps BoT-SORT / ByteTrack under the hood) with Kalman Filter + Hungarian
Algorithm for data association.

Responsibilities:
  - Assigns persistent IDs to each person and weapon across frames
  - Maintains per-ID trajectory history (breadcrumb path)
  - Computes kinematic features: velocity, acceleration, direction vector
  - Detects sudden movements (potential aggression/running indicators)
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from ultralytics import YOLO

from engine.detector import Detection


# ─── Configuration ─────────────────────────────────────────────────────────

TRAIL_MAX_LEN = 60          # How many past positions to keep per track
VELOCITY_ALPHA = 0.3        # EMA smoothing factor for velocity
SUDDEN_MOVE_THRESH = 25.0   # Pixels/frame threshold for "sudden movement"


# ─── Data Structures ───────────────────────────────────────────────────────

@dataclass
class TrackState:
    """Per-track state maintained across frames."""
    track_id: int
    class_name: str
    is_weapon: bool
    is_person: bool
    trail: deque = field(default_factory=lambda: deque(maxlen=TRAIL_MAX_LEN))
    velocity: Tuple[float, float] = (0.0, 0.0)   # (vx, vy) pixels/frame
    speed: float = 0.0                             # scalar px/frame
    acceleration: float = 0.0
    direction_deg: float = 0.0                     # 0=right, 90=up, etc.
    sudden_move: bool = False
    last_seen: float = field(default_factory=time.time)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    confidence: float = 0.0

    def update_kinematics(self, new_center: Tuple[int, int]):
        """Compute velocity, acceleration, direction from new position."""
        if len(self.trail) >= 1:
            prev = self.trail[-1]
            dx = new_center[0] - prev[0]
            dy = new_center[1] - prev[1]
            raw_speed = math.sqrt(dx * dx + dy * dy)

            # Exponential moving average smoothing
            prev_vx, prev_vy = self.velocity
            new_vx = VELOCITY_ALPHA * dx + (1 - VELOCITY_ALPHA) * prev_vx
            new_vy = VELOCITY_ALPHA * dy + (1 - VELOCITY_ALPHA) * prev_vy
            self.velocity = (new_vx, new_vy)

            prev_speed = self.speed
            self.speed = VELOCITY_ALPHA * raw_speed + (1 - VELOCITY_ALPHA) * prev_speed
            self.acceleration = self.speed - prev_speed

            if abs(dx) > 1e-3 or abs(dy) > 1e-3:
                self.direction_deg = math.degrees(math.atan2(-dy, dx)) % 360

            self.sudden_move = raw_speed > SUDDEN_MOVE_THRESH

        self.trail.append(new_center)
        self.last_seen = time.time()


# ─── Tracker Class ─────────────────────────────────────────────────────────

class MultiObjectTracker:
    """
    Algorithm 2 — ByteTrack Multi-Object Movement Tracker
    
    Uses Ultralytics' built-in ByteTrack tracker (Kalman Filter + Hungarian
    matching) with enhanced kinematic state management:
      - Trajectory trails for person and weapon movement paths
      - Per-track velocity and acceleration computation
      - Sudden movement detection for threat flagging
    """

    def __init__(
        self,
        tracker_config: str = "bytetrack.yaml",
        max_age: int = 30,
        device: Optional[str] = None,
    ):
        self.tracker_config = tracker_config
        self.max_age = max_age
        self.device = device

        # State store: track_id -> TrackState
        self._tracks: Dict[int, TrackState] = {}
        # Track ID counter for weapon pseudo-tracks (manual assignment fallback)
        self._next_weapon_id = 9000

        logger.success(f"[Tracker] Algorithm 2 (ByteTrack) initialized. Config: {tracker_config}")

    def update(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        person_model: YOLO,
    ) -> List[Detection]:
        """
        Run ByteTrack on person detections, assign track IDs, then associate
        weapons to the same frame via proximity IoU matching.

        Args:
            frame:       Current BGR video frame.
            detections:  Algorithm 1 detections (persons + weapons).
            person_model: YOLO model used for tracking (with .track() method).

        Returns:
            Updated detections with track_id filled in.
        """
        tracked_objects: List[Detection] = []
        vehicles_and_persons = [d for d in detections if d.is_person or d.is_vehicle]
        weapons = [d for d in detections if d.is_weapon]

        # Vehicle class ID map
        vehicle_class_names = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

        # ── Run ByteTrack on persons & vehicles via ultralytics .track() ────
        if vehicles_and_persons:
            track_results = person_model.track(
                source=frame,
                persist=True,
                conf=0.30,
                classes=[0, 2, 3, 5, 7],
                tracker=self.tracker_config,
                verbose=False,
                device=self.device,
            )

            for result in track_results:
                if result.boxes is None or result.boxes.id is None:
                    continue
                for box in result.boxes:
                    tid = int(box.id[0].item())
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)

                    is_p = cls_id == 0
                    is_v = cls_id in vehicle_class_names
                    cname = "person" if is_p else vehicle_class_names.get(cls_id, "vehicle")

                    det = Detection(
                        track_id=tid,
                        class_id=cls_id,
                        class_name=cname,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        is_person=is_p,
                        is_vehicle=is_v,
                    )

                    # Update or create track state
                    if tid not in self._tracks:
                        self._tracks[tid] = TrackState(
                            track_id=tid,
                            class_name=cname,
                            is_weapon=False,
                            is_person=is_p,
                        )
                    ts = self._tracks[tid]
                    ts.bbox = (x1, y1, x2, y2)
                    ts.confidence = conf
                    ts.class_name = cname
                    ts.update_kinematics(center)

                    tracked_objects.append(det)

        # ── Assign IDs to weapons via nearest-person-hand association ──────
        tracked_weapons: List[Detection] = []
        for w in weapons:
            wx, wy = w.center
            best_id = self._find_or_create_weapon_id(wx, wy)
            w.track_id = best_id
            
            if best_id not in self._tracks:
                self._tracks[best_id] = TrackState(
                    track_id=best_id,
                    class_name=w.class_name,
                    is_weapon=True,
                    is_person=False,
                )
            ts = self._tracks[best_id]
            ts.bbox = w.bbox
            ts.confidence = w.confidence
            ts.class_name = w.class_name
            ts.update_kinematics(w.center)
            tracked_weapons.append(w)

        # ── Prune stale tracks ─────────────────────────────────────────────
        stale_threshold = time.time() - (self.max_age / 30.0)
        stale = [tid for tid, ts in self._tracks.items() if ts.last_seen < stale_threshold]
        for tid in stale:
            del self._tracks[tid]

        return tracked_objects + tracked_weapons

    def _find_or_create_weapon_id(self, wx: int, wy: int) -> int:
        """
        Find existing weapon track near (wx, wy), or create a new one.
        Uses Euclidean distance threshold (50 px) to associate weapon detections.
        """
        best_id = None
        best_dist = 50.0  # px threshold

        for tid, ts in self._tracks.items():
            if not ts.is_weapon:
                continue
            if ts.trail:
                tx, ty = ts.trail[-1]
                dist = math.sqrt((wx - tx) ** 2 + (wy - ty) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid

        if best_id is None:
            best_id = self._next_weapon_id
            self._next_weapon_id += 1

        return best_id

    def get_track(self, track_id: int) -> Optional[TrackState]:
        return self._tracks.get(track_id)

    def get_all_tracks(self) -> Dict[int, TrackState]:
        return dict(self._tracks)

    def draw_trails(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Draw motion trail breadcrumbs for each tracked entity.
        - Persons: cyan gradient trail
        - Weapons: red gradient trail
        """
        vis = frame.copy()

        for det in detections:
            if det.track_id < 0:
                continue
            ts = self._tracks.get(det.track_id)
            if ts is None or len(ts.trail) < 2:
                continue

            trail = list(ts.trail)
            n = len(trail)
            for i in range(1, n):
                alpha = i / n  # 0 = oldest (faint), 1 = newest (bright)
                if det.is_weapon:
                    color = (0, int(50 * alpha), int(255 * alpha))  # Red
                else:
                    color = (int(255 * alpha), int(200 * alpha), 0)  # Cyan-yellow

                thickness = max(1, int(3 * alpha))
                cv2.line(vis, trail[i - 1], trail[i], color, thickness, cv2.LINE_AA)

            # Draw direction arrow at current position
            if n >= 2:
                vx, vy = ts.velocity
                if ts.speed > 2.0:
                    cx, cy = trail[-1]
                    arrow_scale = min(ts.speed * 1.5, 40)
                    end = (int(cx + vx * arrow_scale / max(ts.speed, 1)),
                           int(cy + vy * arrow_scale / max(ts.speed, 1)))
                    arrow_color = (0, 0, 200) if det.is_weapon else (200, 200, 0)
                    cv2.arrowedLine(vis, (cx, cy), end, arrow_color, 2, cv2.LINE_AA, tipLength=0.4)

        return vis

    def draw_kinematics(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Overlay speed/direction HUD text near each tracked object."""
        vis = frame.copy()
        for det in detections:
            if det.track_id < 0:
                continue
            ts = self._tracks.get(det.track_id)
            if ts is None:
                continue
            x1, y1, _, _ = det.bbox
            spd_label = f"Spd:{ts.speed:.1f}px/f"
            if ts.sudden_move:
                spd_label += " !"
            color = (0, 0, 255) if ts.sudden_move else (200, 200, 200)
            cv2.putText(
                vis, spd_label, (x1, y1 - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA
            )
        return vis
