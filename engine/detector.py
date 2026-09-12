"""
Algorithm 1: YOLOv8 Weapon & Person Detection
=============================================
Uses Ultralytics YOLOv8/YOLO11 to detect:
  - Persons (COCO class 0)
  - Weapons: handguns, knives, rifles, and other dangerous items

Model weights auto-downloaded on first run.
Custom weapon-trained weights can be specified via WEAPON_MODEL_PATH env var.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from loguru import logger
from ultralytics import YOLO


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class Detection:
    """Single detection result from Algorithm 1."""
    track_id: int = -1               # Filled by Algorithm 2 (Tracker)
    class_id: int = 0
    class_name: str = ""
    confidence: float = 0.0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)   # x1, y1, x2, y2
    is_weapon: bool = False
    is_person: bool = False
    is_vehicle: bool = False
    target_match_name: Optional[str] = None  # Watchlist Person Re-ID match
    target_match_conf: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)
        cname = self.class_name.lower()
        self.is_person = cname == "person"
        self.is_weapon = cname in WEAPON_CLASS_NAMES
        self.is_vehicle = cname in VEHICLE_CLASS_NAMES


VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
VEHICLE_CLASS_NAMES = {"car", "motorcycle", "bus", "truck", "vehicle"}


# Weapon class names (COCO + custom weapon dataset labels)
WEAPON_CLASS_NAMES = {
    "knife", "gun", "pistol", "handgun", "rifle", "firearm",
    "revolver", "shotgun", "sword", "machete", "weapon", "blade",
    # Improvised weapons
    "bottle", "baseball bat", "bat", "scissors", "fork", "hammer",
    "wrench", "umbrella", "stick", "rod", "pen", "pencil", "pipe",
}

# COCO class 0 = person; we also detect common objects that can serve as
# weapons using a secondary custom model if available.
PERSON_CLASS_ID = 0

# Default confidence thresholds
DEFAULT_PERSON_CONF = 0.40
DEFAULT_WEAPON_CONF = 0.30  # Lower threshold — catch more potential weapons

# IMPROVISED weapon COCO class IDs (objects held in hand that can harm)
# These are standard COCO dataset class IDs
IMPROVISED_WEAPON_COCO_IDS = {
    25: "umbrella",      # Can be used as a weapon
    33: "baseball bat",  # Clear weapon
    34: "baseball glove",
    35: "skateboard",
    39: "bottle",        # Glass bottle = weapon
    40: "wine glass",    # Glass = weapon
    42: "fork",          # Sharp object
    43: "knife",         # Weapon
    44: "spoon",
    76: "scissors",      # Sharp weapon
    84: "book",          # Can be used in altercations
}

# Colors for visualization
COLOR_PERSON = (0, 200, 255)    # Cyan-ish
COLOR_WEAPON = (0, 0, 255)      # Red
COLOR_UNKNOWN = (180, 180, 180) # Gray


class WeaponDetector:
    """
    Algorithm 1 — YOLOv8 Deep Neural Network Detector
    
    Detects persons and weapons in a video frame using two YOLO models:
      - Primary: yolov8n.pt (fast, COCO - detects persons + common objects)
      - Weapon:  yolov8n-weapon.pt / custom (if available, else fallback to 
                 filtering COCO knife class)
    
    The detector runs on CUDA (RTX 4050) when available.
    """

    def __init__(
        self,
        person_model_path: str = "yolov8n.pt",
        weapon_model_path: Optional[str] = None,
        person_conf: float = DEFAULT_PERSON_CONF,
        weapon_conf: float = DEFAULT_WEAPON_CONF,
        device: Optional[str] = None,
        imgsz: int = 640,
    ):
        # Auto-select device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.imgsz = imgsz
        self.person_conf = person_conf
        self.weapon_conf = weapon_conf

        logger.info(f"[Detector] Loading person model: {person_model_path} on {device}")
        self.person_model = YOLO(person_model_path)
        self.person_model.to(device)

        # Weapon model: use environment override, constructor arg, or fallback
        _weapon_path = (
            os.environ.get("WEAPON_MODEL_PATH")
            or weapon_model_path
        )
        if _weapon_path and os.path.isfile(_weapon_path):
            logger.info(f"[Detector] Loading weapon model: {_weapon_path}")
            self.weapon_model = YOLO(_weapon_path)
            self.weapon_model.to(device)
            self._has_weapon_model = True
        else:
            logger.warning(
                "[Detector] No dedicated weapon model found. "
                "Using COCO knife/scissors detection + heuristic extension."
            )
            self.weapon_model = None
            self._has_weapon_model = False

        # Map COCO IDs that overlap with weapon categories
        # knife=43, scissors=76, baseball bat=33, bottle=39, wine glass=40,
        # fork=42, umbrella=25, skateboard=35
        self._coco_weapon_ids = IMPROVISED_WEAPON_COCO_IDS

        logger.success(f"[Detector] Ready. CUDA={torch.cuda.is_available()}, Device={device}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run Algorithm 1 detection on a single BGR frame.
        
        Returns:
            List of Detection objects (persons + weapons).
        """
        detections: List[Detection] = []

        # ── Person & Vehicle detection via primary YOLO model ──────────────
        # Classes: 0 (person), 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
        primary_classes = [PERSON_CLASS_ID] + list(VEHICLE_CLASS_IDS.keys())
        person_results = self.person_model.predict(
            source=frame,
            conf=self.person_conf,
            classes=primary_classes,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
        )

        for result in person_results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if cls_id == PERSON_CLASS_ID:
                    det = Detection(
                        class_id=cls_id,
                        class_name="person",
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        is_person=True,
                    )
                else:
                    vname = VEHICLE_CLASS_IDS.get(cls_id, "vehicle")
                    det = Detection(
                        class_id=cls_id,
                        class_name=vname,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        is_vehicle=True,
                    )
                detections.append(det)

        # ── Weapon detection ───────────────────────────────────────────────
        if self._has_weapon_model:
            weapon_results = self.weapon_model.predict(
                source=frame,
                conf=self.weapon_conf,
                imgsz=self.imgsz,
                verbose=False,
                device=self.device,
            )
            for result in weapon_results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cls_name = (
                        result.names.get(cls_id, "weapon")
                        if result.names
                        else "weapon"
                    )
                    det = Detection(
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        is_weapon=True,
                    )
                    detections.append(det)
        else:
            # Fallback: detect knife/scissors from COCO person model
            general_results = self.person_model.predict(
                source=frame,
                conf=self.weapon_conf,
                classes=list(self._coco_weapon_ids.keys()),
                imgsz=self.imgsz,
                verbose=False,
                device=self.device,
            )
            for result in general_results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cls_name = self._coco_weapon_ids.get(cls_id, "weapon")
                    det = Detection(
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        is_weapon=True,
                    )
                    detections.append(det)

        return detections

    def draw(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes and labels for all detections."""
        vis = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            if det.target_match_name:
                color = (20, 20, 220)  # Intense alert red
                label = f"TARGET: {det.target_match_name.upper()} [{det.target_match_conf:.0%}]"
            elif det.is_weapon:
                color = COLOR_WEAPON
                label = f"{det.class_name.upper()} [{det.confidence:.0%}]"
            elif det.is_vehicle:
                color = (50, 150, 200)  # Amber / ochre for vehicles
                label = f"{det.class_name.upper()} [{det.confidence:.0%}]"
            else:
                color = COLOR_PERSON
                label = f"{det.class_name.upper()} [{det.confidence:.0%}]"

            if det.track_id >= 0:
                label = f"ID#{det.track_id} {label}"

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3 if det.target_match_name else 2)
            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                vis, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA
            )
        return vis
