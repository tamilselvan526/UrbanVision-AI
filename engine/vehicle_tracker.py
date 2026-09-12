"""
Vehicle Tracker & License Plate Association Module
===================================================
Maintains state for all tracked vehicles (cars, motorcycles, buses, trucks),
associates detected license plates with persistent vehicle tracks, records
movement vectors, and logs events to the Intelligence Database.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from engine.detector import Detection
from engine.intelligence_db import intelligence_db, SNAPSHOTS_DIR
from engine.plate_recognizer import plate_recognizer


@dataclass
class VehicleState:
    """State for a single tracked vehicle."""
    track_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    trail: deque = field(default_factory=lambda: deque(maxlen=60))
    plate_number: str = ""
    plate_confidence: float = 0.0
    plate_box: Optional[Tuple[int, int, int, int]] = None
    direction_deg: float = 0.0
    speed: float = 0.0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    snapshot_filename: str = ""
    frames_since_plate_check: int = 0
    plate_confirmed: bool = False

    def update_kinematics(self, new_center: Tuple[int, int]):
        if len(self.trail) >= 1:
            prev = self.trail[-1]
            dx = new_center[0] - prev[0]
            dy = new_center[1] - prev[1]
            self.speed = math.sqrt(dx * dx + dy * dy)
            if abs(dx) > 1e-2 or abs(dy) > 1e-2:
                self.direction_deg = math.degrees(math.atan2(-dy, dx)) % 360
        self.trail.append(new_center)
        self.last_seen = time.time()


class VehicleTracker:
    """
    Coordinates vehicle tracking and license plate OCR logging across frames.
    """

    def __init__(self, check_interval: int = 6):
        self.check_interval = check_interval
        self._vehicles: Dict[int, VehicleState] = {}
        logger.info("[VehicleTracker] Initialized vehicle tracking & plate association.")

    def update(
        self,
        frame: np.ndarray,
        vehicle_detections: List[Detection],
        camera_source: str = "Camera 01",
    ) -> List[VehicleState]:
        """
        Update vehicle trajectories and run plate recognition if needed.
        """
        active_ids = set()

        for det in vehicle_detections:
            tid = det.track_id
            if tid < 0:
                continue
            active_ids.add(tid)

            if tid not in self._vehicles:
                self._vehicles[tid] = VehicleState(
                    track_id=tid,
                    class_name=det.class_name,
                    bbox=det.bbox,
                    center=det.center,
                )

            v = self._vehicles[tid]
            v.bbox = det.bbox
            v.center = det.center
            v.update_kinematics(det.center)
            v.frames_since_plate_check += 1

            # Run plate OCR periodically if not yet confirmed
            if not v.plate_confirmed or v.frames_since_plate_check >= self.check_interval:
                v.frames_since_plate_check = 0
                x1, y1, x2, y2 = det.bbox
                h, w = frame.shape[:2]
                pad = 10
                crop = frame[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]

                if crop.size > 0:
                    plate_res = plate_recognizer.recognize_plate(crop)
                    if plate_res:
                        plate_text, conf, pbox = plate_res
                        v.plate_number = plate_text
                        v.plate_confidence = conf
                        v.plate_box = (x1 + pbox[0], y1 + pbox[1], pbox[2], pbox[3])
                        v.plate_confirmed = True

                        # Save vehicle snapshot
                        snap_name = f"veh_{plate_text}_{tid}_{int(time.time())}.jpg"
                        snap_path = SNAPSHOTS_DIR / "vehicles" / snap_name
                        cv2.imwrite(str(snap_path), crop)
                        v.snapshot_filename = snap_name

                        # Update database
                        intelligence_db.record_or_update_vehicle(
                            plate_number=plate_text,
                            vehicle_type=det.class_name,
                            camera_source=camera_source,
                            point=det.center,
                            direction_deg=v.direction_deg,
                            speed=v.speed,
                            snapshot_filename=snap_name,
                        )

        # Prune stale vehicle tracks (older than 10 seconds)
        now = time.time()
        stale = [tid for tid, v in self._vehicles.items() if (now - v.last_seen) > 10.0]
        for tid in stale:
            del self._vehicles[tid]

        return list(self._vehicles.values())

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw vehicle boxes, plate banners, and movement trails."""
        vis = frame.copy()

        for v in self._vehicles.values():
            x1, y1, x2, y2 = v.bbox

            # Muted amber / khaki vehicle border
            color = (50, 160, 200) if v.plate_number else (40, 120, 150)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Draw vehicle motion trail
            trail = list(v.trail)
            if len(trail) >= 2:
                for i in range(1, len(trail)):
                    cv2.line(vis, trail[i - 1], trail[i], (30, 130, 180), 2, cv2.LINE_AA)

            # Direction arrow
            if v.speed > 2.0 and trail:
                rad = math.radians(v.direction_deg)
                end = (int(v.center[0] + math.cos(rad) * 35), int(v.center[1] - math.sin(rad) * 35))
                cv2.arrowedLine(vis, v.center, end, (20, 160, 220), 2, cv2.LINE_AA, tipLength=0.35)

            # Label banner
            label = f"{v.class_name.upper()} #{v.track_id}"
            if v.plate_number:
                label += f" | PLATE: {v.plate_number}"
            else:
                label += f" | SPD: {v.speed:.1f}px"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 6, y1), (20, 20, 20), -1)
            cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, 1)
            cv2.putText(vis, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

            # Highlight detected plate box inside vehicle
            if v.plate_box:
                px, py, pw, ph = v.plate_box
                cv2.rectangle(vis, (px, py), (px + pw, py + ph), (0, 220, 220), 2)

        return vis


# Global singleton
vehicle_tracker = VehicleTracker()
