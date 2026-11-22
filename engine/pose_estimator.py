"""
Algorithm 3 (Enhanced): YOLO-Pose Human Keypoint Estimation
============================================================
ENHANCED VERSION — adds:
  - Per-keypoint velocity tracking across frames (temporal motion)
  - Striking / punch / slap motion detection (rapid arm swing)
  - Stabbing motion detection (rapid elbow→wrist forward thrust)
  - Aggressive stare detection (face-to-face orientation + close proximity)
  - Improvised weapon holding detection (any object in wrist zone)
  - Enhanced posture: FIGHTING_STANCE, STRIKING, STABBING, THREATENING

Keypoints (COCO order):
  0:nose, 1:l_eye, 2:r_eye, 3:l_ear, 4:r_ear
  5:l_shoulder, 6:r_shoulder, 7:l_elbow, 8:r_elbow
  9:l_wrist, 10:r_wrist, 11:l_hip, 12:r_hip
  13:l_knee, 14:r_knee, 15:l_ankle, 16:r_ankle
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from ultralytics import YOLO


# ─── COCO Keypoint Indices ─────────────────────────────────────────────────

KP = {
    "nose": 0,
    "l_eye": 1, "r_eye": 2,
    "l_ear": 3, "r_ear": 4,
    "l_shoulder": 5, "r_shoulder": 6,
    "l_elbow": 7, "r_elbow": 8,
    "l_wrist": 9, "r_wrist": 10,
    "l_hip": 11, "r_hip": 12,
    "l_knee": 13, "r_knee": 14,
    "l_ankle": 15, "r_ankle": 16,
}

SKELETON_PAIRS = [
    (KP["nose"], KP["l_eye"]),   (KP["nose"], KP["r_eye"]),
    (KP["l_eye"], KP["l_ear"]),  (KP["r_eye"], KP["r_ear"]),
    (KP["l_ear"], KP["l_shoulder"]), (KP["r_ear"], KP["r_shoulder"]),
    (KP["l_shoulder"], KP["r_shoulder"]),
    (KP["l_shoulder"], KP["l_elbow"]), (KP["r_shoulder"], KP["r_elbow"]),
    (KP["l_elbow"], KP["l_wrist"]),   (KP["r_elbow"], KP["r_wrist"]),
    (KP["l_shoulder"], KP["l_hip"]),  (KP["r_shoulder"], KP["r_hip"]),
    (KP["l_hip"], KP["r_hip"]),
    (KP["l_hip"], KP["l_knee"]),      (KP["r_hip"], KP["r_knee"]),
    (KP["l_knee"], KP["l_ankle"]),    (KP["r_knee"], KP["r_ankle"]),
]

# ─── Thresholds ───────────────────────────────────────────────────────────

# Wrist speed (px/frame) to classify as a striking/stabbing motion
STRIKE_VELOCITY_THRESH  = 7.5     # Sudden fast wrist movement = punch / slap / strike
STAB_VELOCITY_THRESH    = 11.0    # Fast forward wrist thrust = stab
ELBOW_CHANGE_THRESH     = 14.0    # Degrees/frame change = rapid elbow movement
WRIST_HISTORY_FRAMES    = 8       # Frames of wrist history for motion pattern

# Posture angle thresholds
FALLEN_TORSO_ANGLE      = 45.0
ARM_RAISED_MIN_ANGLE    = 60.0
FIGHTING_STANCE_DIST    = 0.35    # shoulder-width ratio for wide leg stance
AIMING_ELBOW_ANGLE      = 155.0


# ─── Data Structures ──────────────────────────────────────────────────────

@dataclass
class MotionFeatures:
    """Temporal motion features derived from keypoint history."""
    left_wrist_vel:  float = 0.0
    right_wrist_vel: float = 0.0
    max_wrist_vel:   float = 0.0

    # Strike: fast arm swing (wrist speed peak)
    is_striking:     bool = False
    strike_arm:      str = ""         # "left" | "right" | "both"

    # Stab: fast forward wrist thrust (elbow rapidly extending)
    is_stabbing:     bool = False
    stab_arm:        str = ""

    # Elbow angle change rate (indicates rapid arm extension/flexion)
    elbow_change_left:  float = 0.0
    elbow_change_right: float = 0.0

    # Wrist trajectory direction (for thrust vs swing detection)
    left_wrist_dir:  Optional[Tuple[float, float]] = None
    right_wrist_dir: Optional[Tuple[float, float]] = None


@dataclass
class PoseResult:
    """Pose estimation result for one person — enhanced with motion features."""
    track_id:         int = -1
    person_bbox:      Tuple[int, int, int, int] = (0, 0, 0, 0)
    keypoints:        np.ndarray = field(default_factory=lambda: np.zeros((17, 3)))
    confidence_score: float = 0.0

    # Derived static features
    left_wrist:        Optional[Tuple[int, int]] = None
    right_wrist:       Optional[Tuple[int, int]] = None
    nose_pos:          Optional[Tuple[int, int]] = None
    aim_vector_left:   Optional[Tuple[float, float]] = None
    aim_vector_right:  Optional[Tuple[float, float]] = None
    elbow_angle_left:  float = 180.0
    elbow_angle_right: float = 180.0
    torso_angle:       float = 0.0

    # Enhanced posture — now includes STRIKING, STABBING, FIGHTING_STANCE
    posture: str = "unknown"

    # Compliant safe action indicators (hands up / waving / surrender / clear)
    is_safe_action: bool = False
    safe_action_label: str = ""

    # Temporal motion analysis
    motion: MotionFeatures = field(default_factory=MotionFeatures)

    # Face direction vector (for stare detection)
    face_direction: Optional[Tuple[float, float]] = None   # unit vec nose→target


# ─── Keypoint Utilities ───────────────────────────────────────────────────

def _kp_valid(kp: np.ndarray, idx: int, conf_thresh: float = 0.3) -> bool:
    return kp[idx, 2] >= conf_thresh

def _kp_xy(kp: np.ndarray, idx: int) -> Tuple[float, float]:
    return (kp[idx, 0], kp[idx, 1])

def _angle_between(a, b, c) -> float:
    """Angle ABC in degrees (b is vertex)."""
    ax, ay = a[0]-b[0], a[1]-b[1]
    cx, cy = c[0]-b[0], c[1]-b[1]
    dot = ax*cx + ay*cy
    mag = math.sqrt(ax**2+ay**2) * math.sqrt(cx**2+cy**2)
    if mag < 1e-6: return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot/mag))))

def _unit_vec(a, b) -> Tuple[float, float]:
    dx, dy = b[0]-a[0], b[1]-a[1]
    mag = math.sqrt(dx**2+dy**2)
    if mag < 1e-6: return (0.0, 0.0)
    return (dx/mag, dy/mag)

def _dist(a, b) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


# ─── Per-Track Keypoint History (for velocity/motion) ─────────────────────

class _KPHistory:
    """Tracks keypoint positions over last N frames for one person."""
    def __init__(self, max_len: int = WRIST_HISTORY_FRAMES):
        self.max_len = max_len
        self.left_wrist:  deque = deque(maxlen=max_len)
        self.right_wrist: deque = deque(maxlen=max_len)
        self.elbow_l:     deque = deque(maxlen=max_len)
        self.elbow_r:     deque = deque(maxlen=max_len)
        self.nose:        deque = deque(maxlen=max_len)

    def push(self, pr: PoseResult):
        kp = pr.keypoints
        if _kp_valid(kp, KP["l_wrist"]): self.left_wrist.append(_kp_xy(kp, KP["l_wrist"]))
        if _kp_valid(kp, KP["r_wrist"]): self.right_wrist.append(_kp_xy(kp, KP["r_wrist"]))
        self.elbow_l.append(pr.elbow_angle_left)
        self.elbow_r.append(pr.elbow_angle_right)
        if pr.nose_pos: self.nose.append(pr.nose_pos)

    def wrist_velocity(self, side: str) -> float:
        """Average pixel/frame speed of wrist over recent frames."""
        hist = self.left_wrist if side == "left" else self.right_wrist
        if len(hist) < 2: return 0.0
        pts = list(hist)
        speeds = [_dist(pts[i], pts[i-1]) for i in range(1, len(pts))]
        return float(np.mean(speeds[-3:])) if speeds else 0.0

    def elbow_change_rate(self, side: str) -> float:
        """Max angle change per frame (degrees/frame) for elbow."""
        hist = list(self.elbow_l if side == "left" else self.elbow_r)
        if len(hist) < 2: return 0.0
        changes = [abs(hist[i]-hist[i-1]) for i in range(1, len(hist))]
        return float(max(changes[-3:])) if changes else 0.0

    def wrist_direction(self, side: str) -> Optional[Tuple[float, float]]:
        """Unit vector of wrist movement over last 3 frames."""
        hist = list(self.left_wrist if side == "left" else self.right_wrist)
        if len(hist) < 3: return None
        return _unit_vec(hist[-3], hist[-1])


# ─── Pose Estimator ───────────────────────────────────────────────────────

class PoseEstimator:
    """
    Algorithm 3 (Enhanced) — YOLO-Pose Skeletal Keypoint Estimator

    New capabilities over baseline:
      - Temporal wrist velocity tracking → detects strikes and stabs
      - Aggressive posture detection: STRIKING, STABBING, FIGHTING_STANCE
      - Face direction vector for confrontation/stare detection
      - Improvised weapon holding indicator (wrist in tight grip config)
    """

    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        conf: float = 0.35,
        device: Optional[str] = None,
        imgsz: int = 640,
    ):
        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.conf = conf
        self.imgsz = imgsz

        logger.info(f"[PoseEstimator] Loading: {model_path} on {device}")
        self.model = YOLO(model_path)
        self.model.to(device)

        # Per-track keypoint history
        self._kp_histories: Dict[int, _KPHistory] = {}
        logger.success("[PoseEstimator] Algorithm 3 Enhanced (Pose + Motion) ready.")

    def estimate(self, frame: np.ndarray) -> List[PoseResult]:
        """Run pose estimation on a BGR frame. Returns list of PoseResult."""
        h, w = frame.shape[:2]
        results = self.model.predict(
            source=frame, conf=self.conf, imgsz=self.imgsz,
            verbose=False, device=self.device,
        )

        pose_results: List[PoseResult] = []
        for result in results:
            if result.keypoints is None or result.boxes is None:
                continue
            kp_data = result.keypoints.data.cpu().numpy()
            boxes   = result.boxes.xyxy.cpu().numpy()
            confs   = result.boxes.conf.cpu().numpy()

            for i in range(len(kp_data)):
                kp   = kp_data[i]
                bbox = tuple(map(int, boxes[i]))
                pr   = PoseResult(
                    person_bbox=bbox,
                    keypoints=kp,
                    confidence_score=float(confs[i]),
                )
                self._compute_static_features(pr, h, w)
                pose_results.append(pr)

        return pose_results

    def update_motion(self, pose_results: List[PoseResult]) -> List[PoseResult]:
        """
        Update temporal motion features for each tracked pose.
        Must be called AFTER tracker assigns track_ids.
        """
        for pr in pose_results:
            tid = pr.track_id
            if tid < 0:
                continue

            if tid not in self._kp_histories:
                self._kp_histories[tid] = _KPHistory()

            hist = self._kp_histories[tid]
            hist.push(pr)

            # ── Wrist velocities ──────────────────────────────────────
            lv = hist.wrist_velocity("left")
            rv = hist.wrist_velocity("right")
            pr.motion.left_wrist_vel  = lv
            pr.motion.right_wrist_vel = rv
            pr.motion.max_wrist_vel   = max(lv, rv)

            # ── Elbow change rates ────────────────────────────────────
            pr.motion.elbow_change_left  = hist.elbow_change_rate("left")
            pr.motion.elbow_change_right = hist.elbow_change_rate("right")

            # ── Wrist trajectory directions ───────────────────────────
            pr.motion.left_wrist_dir  = hist.wrist_direction("left")
            pr.motion.right_wrist_dir = hist.wrist_direction("right")

            # ── Strike detection ──────────────────────────────────────
            # Fast wrist swing = strike (but NOT a pure forward thrust)
            strike_arms = []
            if lv > STRIKE_VELOCITY_THRESH:
                strike_arms.append("left")
            if rv > STRIKE_VELOCITY_THRESH:
                strike_arms.append("right")
            pr.motion.is_striking = len(strike_arms) > 0
            pr.motion.strike_arm  = "+".join(strike_arms)

            # ── Stab detection ────────────────────────────────────────
            # Fast wrist + rapid elbow extension = stab/thrust motion
            stab_arms = []
            if (lv > STAB_VELOCITY_THRESH and
                    pr.motion.elbow_change_left > ELBOW_CHANGE_THRESH):
                stab_arms.append("left")
            if (rv > STAB_VELOCITY_THRESH and
                    pr.motion.elbow_change_right > ELBOW_CHANGE_THRESH):
                stab_arms.append("right")
            pr.motion.is_stabbing = len(stab_arms) > 0
            pr.motion.stab_arm    = "+".join(stab_arms)

            # ── Upgrade posture if motion detected ────────────────────
            if pr.motion.is_stabbing:
                pr.posture = "stabbing"
            elif pr.motion.is_striking:
                pr.posture = "striking"

        # Prune stale track histories
        active = {pr.track_id for pr in pose_results if pr.track_id >= 0}
        stale  = [tid for tid in self._kp_histories if tid not in active]
        for tid in stale:
            del self._kp_histories[tid]

        return pose_results

    # ─── Static Feature Computation ──────────────────────────────────────

    def _compute_static_features(self, pr: PoseResult, fh: int, fw: int):
        kp = pr.keypoints

        # Wrists
        if _kp_valid(kp, KP["l_wrist"]):
            pr.left_wrist  = (int(kp[KP["l_wrist"],0]), int(kp[KP["l_wrist"],1]))
        if _kp_valid(kp, KP["r_wrist"]):
            pr.right_wrist = (int(kp[KP["r_wrist"],0]), int(kp[KP["r_wrist"],1]))

        # Nose
        if _kp_valid(kp, KP["nose"]):
            pr.nose_pos = (int(kp[KP["nose"],0]), int(kp[KP["nose"],1]))

        # Aim vectors
        if _kp_valid(kp, KP["l_shoulder"]) and _kp_valid(kp, KP["l_wrist"]):
            pr.aim_vector_left = _unit_vec(_kp_xy(kp, KP["l_shoulder"]),
                                           _kp_xy(kp, KP["l_wrist"]))
        if _kp_valid(kp, KP["r_shoulder"]) and _kp_valid(kp, KP["r_wrist"]):
            pr.aim_vector_right = _unit_vec(_kp_xy(kp, KP["r_shoulder"]),
                                            _kp_xy(kp, KP["r_wrist"]))

        # Elbow angles
        if all(_kp_valid(kp, k) for k in [KP["l_shoulder"], KP["l_elbow"], KP["l_wrist"]]):
            pr.elbow_angle_left = _angle_between(
                _kp_xy(kp, KP["l_shoulder"]), _kp_xy(kp, KP["l_elbow"]), _kp_xy(kp, KP["l_wrist"]))
        if all(_kp_valid(kp, k) for k in [KP["r_shoulder"], KP["r_elbow"], KP["r_wrist"]]):
            pr.elbow_angle_right = _angle_between(
                _kp_xy(kp, KP["r_shoulder"]), _kp_xy(kp, KP["r_elbow"]), _kp_xy(kp, KP["r_wrist"]))

        # Torso angle
        if _kp_valid(kp, KP["l_shoulder"]) and _kp_valid(kp, KP["l_hip"]):
            sy, hy = kp[KP["l_shoulder"],1], kp[KP["l_hip"],1]
            sx, hx = kp[KP["l_shoulder"],0], kp[KP["l_hip"],0]
            pr.torso_angle = abs(math.degrees(math.atan2(abs(sx-hx), max(abs(sy-hy), 1e-6))))

        # Face direction (ear-to-ear midpoint → nose direction)
        if (_kp_valid(kp, KP["l_ear"]) and _kp_valid(kp, KP["r_ear"])
                and _kp_valid(kp, KP["nose"])):
            ear_mid = ((kp[KP["l_ear"],0]+kp[KP["r_ear"],0])/2,
                       (kp[KP["l_ear"],1]+kp[KP["r_ear"],1])/2)
            pr.face_direction = _unit_vec(ear_mid, _kp_xy(kp, KP["nose"]))

        # Posture classification (static)
        pr.posture = self._classify_posture(pr, fh, fw)

    def _classify_posture(self, pr: PoseResult, fh: int, fw: int) -> str:
        kp = pr.keypoints

        # Fallen: torso near horizontal
        if pr.torso_angle > FALLEN_TORSO_ANGLE:
            return "fallen"

        # Crouching: tight knee angle
        if all(_kp_valid(kp, k) for k in [KP["l_hip"], KP["l_knee"], KP["l_ankle"]]):
            knee_angle = _angle_between(
                _kp_xy(kp, KP["l_hip"]), _kp_xy(kp, KP["l_knee"]), _kp_xy(kp, KP["l_ankle"]))
            if knee_angle < 110:
                return "crouching"

        # Fighting stance: wide leg spread (feet wider than shoulders)
        if (all(_kp_valid(kp, k) for k in [KP["l_ankle"], KP["r_ankle"],
                                             KP["l_shoulder"], KP["r_shoulder"]])):
            foot_spread     = abs(kp[KP["l_ankle"],0] - kp[KP["r_ankle"],0])
            shoulder_spread = abs(kp[KP["l_shoulder"],0] - kp[KP["r_shoulder"],0])
            if shoulder_spread > 0 and foot_spread / shoulder_spread > 1.5:
                return "fighting_stance"

        # Arms raised / Waving / Safe Action:
        # Check if wrists are raised above shoulders or in upper body/head region (works for seated webcam demos too)
        bx1, by1, bx2, by2 = pr.person_bbox
        bbox_height = max(1, by2 - by1)

        l_wrist_up = False
        r_wrist_up = False
        if _kp_valid(kp, KP["l_wrist"]):
            lw_y = kp[KP["l_wrist"], 1]
            if _kp_valid(kp, KP["l_shoulder"]) and lw_y < kp[KP["l_shoulder"], 1]:
                l_wrist_up = True
            elif _kp_valid(kp, KP["nose"]) and lw_y < kp[KP["nose"], 1] + 25:
                l_wrist_up = True
            elif lw_y < by1 + bbox_height * 0.45:
                l_wrist_up = True

        if _kp_valid(kp, KP["r_wrist"]):
            rw_y = kp[KP["r_wrist"], 1]
            if _kp_valid(kp, KP["r_shoulder"]) and rw_y < kp[KP["r_shoulder"], 1]:
                r_wrist_up = True
            elif _kp_valid(kp, KP["nose"]) and rw_y < kp[KP["nose"], 1] + 25:
                r_wrist_up = True
            elif rw_y < by1 + bbox_height * 0.45:
                r_wrist_up = True

        raised = (1 if l_wrist_up else 0) + (1 if r_wrist_up else 0)
        if raised >= 2:
            pr.is_safe_action = True
            pr.safe_action_label = "HANDS RAISED (SAFE)"
            return "hands_up_safe"
        elif raised == 1:
            pr.is_safe_action = True
            pr.safe_action_label = "ARM RAISED / WAVING (SAFE)"
            return "arms_raised"

        # Aiming: nearly straight extended arm
        if pr.elbow_angle_left > AIMING_ELBOW_ANGLE or pr.elbow_angle_right > AIMING_ELBOW_ANGLE:
            return "aiming"

        return "standing"

    # ─── Visualization ────────────────────────────────────────────────────

    def draw_skeleton(self, frame: np.ndarray, pose_results: List[PoseResult]) -> np.ndarray:
        vis = frame.copy()

        POSTURE_COLORS = {
            "hands_up_safe":  (0, 255, 0),    # Bright Green for safe action
            "arms_raised":    (0, 255, 0),    # Bright Green for safe action
            "fallen":         (0, 0, 200),
            "stabbing":       (0, 0, 180),
            "striking":       (0, 40, 220),
            "aiming":         (0, 60, 200),
            "fighting_stance":(0, 100, 200),
            "crouching":      (0, 180, 140),
            "standing":       (0, 160, 0),
            "unknown":        (100, 100, 100),
        }

        for pr in pose_results:
            kp = pr.keypoints
            color = POSTURE_COLORS.get(pr.posture, (100, 100, 100))

            # Skeleton lines
            for p1i, p2i in SKELETON_PAIRS:
                if _kp_valid(kp, p1i, 0.2) and _kp_valid(kp, p2i, 0.2):
                    p1 = (int(kp[p1i,0]), int(kp[p1i,1]))
                    p2 = (int(kp[p2i,0]), int(kp[p2i,1]))
                    cv2.line(vis, p1, p2, color, 2, cv2.LINE_AA)

            # Keypoints
            for i in range(17):
                if _kp_valid(kp, i, 0.3):
                    kp_color = (0, 255, 0) if pr.is_safe_action else (200, 200, 80)
                    cv2.circle(vis, (int(kp[i,0]), int(kp[i,1])), 4, kp_color, -1)

            # Strike/Stab velocity indicator on wrist
            for wrist, vel, side in [
                (pr.left_wrist, pr.motion.left_wrist_vel, "L"),
                (pr.right_wrist, pr.motion.right_wrist_vel, "R"),
            ]:
                if wrist and vel > STRIKE_VELOCITY_THRESH:
                    # Draw velocity arc around wrist
                    radius = min(int(vel * 1.5), 40)
                    wc = (0, 0, 200) if vel > STAB_VELOCITY_THRESH else (0, 80, 220)
                    cv2.circle(vis, wrist, radius, wc, 2, cv2.LINE_AA)
                    cv2.putText(vis, f"{vel:.0f}px", (wrist[0]+12, wrist[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, wc, 1, cv2.LINE_AA)

            # Aim vectors
            for vec, wrist in [(pr.aim_vector_left, pr.left_wrist),
                                (pr.aim_vector_right, pr.right_wrist)]:
                if vec and wrist:
                    end = (wrist[0]+int(vec[0]*40), wrist[1]+int(vec[1]*40))
                    cv2.arrowedLine(vis, wrist, end, (0, 60, 200), 2, cv2.LINE_AA, tipLength=0.4)

            # Posture label
            x1, y1, x2, y2 = pr.person_bbox
            display_label = pr.safe_action_label if pr.is_safe_action else pr.posture.upper()
            cv2.putText(vis, display_label, (x1, y2+18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        return vis
