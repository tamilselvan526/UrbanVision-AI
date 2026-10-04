"""
Human Liveness & Fake Entity / Mannequin Detection Module
==========================================================
Distinguishes real living humans from store display mannequins, clothing dummies,
and cardboard cutouts in crowded retail and street environments.

Techniques:
  1. Biological Micro-Kinematics & Involuntary Motion:
     - Real humans (even when standing still) exhibit involuntary postural micro-sway
       and respiration motion (0.2 - 2.5 Hz).
     - Plastic/fiberglass mannequins exhibit near-zero relative internal motion.
  2. Global Background Motion Compensation:
     - Compensates for camera pan/tilt/jitter so camera movement is not mistaken
       for biological motion.
  3. Keypoint Micro-Variance:
     - Checks temporal fluctuation of joint angles and limb keypoints.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from engine.pose_estimator import PoseResult

# Minimum frames a track must be observed before qualifying as a confirmed mannequin
MIN_EVAL_FRAMES = 25

# Liveness thresholds
LIVENESS_MANNEQUIN_THRESHOLD = 0.25
LIVENESS_HUMAN_THRESHOLD = 0.42


@dataclass
class LivenessTrackState:
    """Temporal tracking state for liveness analysis of a single person track."""
    track_id: int
    first_seen: float = field(default_factory=time.time)
    frame_count: int = 0
    liveness_score: float = 0.70  # Initial neutral-high prior
    is_mannequin: bool = False
    status: str = "EVALUATING"   # "EVALUATING", "REAL_HUMAN", "MANNEQUIN"
    
    # Rolling buffer of aligned normalized grayscale upper-body crops (64x64)
    crop_buffer: deque = field(default_factory=lambda: deque(maxlen=20))
    # Rolling buffer of center coordinates (x, y, time)
    center_history: deque = field(default_factory=lambda: deque(maxlen=30))
    # Rolling buffer of pose joint angles (if available)
    pose_angles: deque = field(default_factory=lambda: deque(maxlen=20))
    
    # Internal micro-motion energy moving average
    micro_energy_avg: float = 0.05
    last_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)


class HumanLivenessDetector:
    """
    Evaluates real human vs mannequin / dummy liveness per tracked person.
    """

    def __init__(
        self,
        mannequin_threshold: float = LIVENESS_MANNEQUIN_THRESHOLD,
        human_threshold: float = LIVENESS_HUMAN_THRESHOLD,
    ):
        self.mannequin_threshold = mannequin_threshold
        self.human_threshold = human_threshold
        self.tracks: Dict[int, LivenessTrackState] = {}
        
        # Camera background motion compensation state
        self._prev_frame_gray: Optional[np.ndarray] = None
        self._camera_dx: float = 0.0
        self._camera_dy: float = 0.0

    def update_frame(self, frame_bgr: np.ndarray):
        """Estimate global camera shift (jitter/panning) between consecutive frames."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        
        # Downscale for fast optical motion compensation
        small = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)
        
        if self._prev_frame_gray is not None:
            # Phase correlation / translation estimation
            shift, response = cv2.phaseCorrelate(
                self._prev_frame_gray.astype(np.float32),
                small.astype(np.float32)
            )
            # Scale shift back to original coordinates
            self._camera_dx = shift[0] * (w / 160.0)
            self._camera_dy = shift[1] * (h / 120.0)
        else:
            self._camera_dx = 0.0
            self._camera_dy = 0.0
            
        self._prev_frame_gray = small

    def evaluate_person(
        self,
        track_id: int,
        bbox: Tuple[int, int, int, int],
        frame_bgr: np.ndarray,
        pose_res: Optional[PoseResult] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate liveness for a tracked person in current frame.
        Returns dict with liveness_score, is_mannequin, and status.
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        now = time.time()
        
        # Get or create state
        if track_id not in self.tracks:
            state = LivenessTrackState(track_id=track_id, last_bbox=bbox)
            self.tracks[track_id] = state
        else:
            state = self.tracks[track_id]
            
        state.frame_count += 1
        state.center_history.append((cx, cy, now))
        state.last_bbox = bbox
        
        # 1. Gross whole-body movement check
        # If person is moving across scene at > 2.5 px/frame relative to camera -> definitely living human!
        gross_motion = 0.0
        if len(state.center_history) >= 4:
            c_prev = state.center_history[-4]
            dx = (cx - c_prev[0]) - self._camera_dx
            dy = (cy - c_prev[1]) - self._camera_dy
            gross_motion = math.hypot(dx, dy) / 4.0

        if gross_motion > 2.5:
            # Active movement confirmed
            state.liveness_score = 0.85 * state.liveness_score + 0.15 * 0.98
            state.is_mannequin = False
            state.status = "REAL_HUMAN"
            return {
                "track_id": track_id,
                "liveness_score": round(state.liveness_score, 2),
                "is_mannequin": False,
                "status": state.status,
            }

        # 2. Biological Micro-Motion Analysis (For stationary or slow entities)
        # Extract upper body crop (head + torso: top 60% of bbox)
        ih, iw = frame_bgr.shape[:2]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        
        crop_y2 = min(ih, y1 + int(bh * 0.65))
        crop_y1 = max(0, y1)
        crop_x1 = max(0, x1)
        crop_x2 = min(iw, x2)
        
        micro_energy = 0.0
        if (crop_y2 - crop_y1 > 15) and (crop_x2 - crop_x1 > 10):
            crop = frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            # Standardize to 64x64 for normalized differencing
            norm_crop = cv2.resize(gray_crop, (64, 64), interpolation=cv2.INTER_AREA)
            norm_crop = cv2.equalizeHist(norm_crop)
            
            state.crop_buffer.append(norm_crop)
            
            if len(state.crop_buffer) >= 6:
                # Compute absolute frame difference between current and 5 frames ago
                ref_crop = state.crop_buffer[-6]
                diff = cv2.absdiff(norm_crop, ref_crop)
                # Compute normalized difference energy (0.0 to 1.0)
                diff_energy = float(np.mean(diff)) / 255.0
                
                # Living human breathing/sway creates typical normalized energy ~ 0.035 - 0.15
                # Rigid plastic mannequin has near-zero internal diff energy < 0.015
                state.micro_energy_avg = 0.80 * state.micro_energy_avg + 0.20 * diff_energy
                micro_energy = state.micro_energy_avg

        # 3. Pose articulation variance
        pose_variance = 0.0
        if pose_res and pose_res.keypoints is not None:
            kpts = pose_res.keypoints
            # Shoulders (5, 6), Elbows (7, 8), Wrists (9, 10)
            if len(kpts) > 8 and kpts[5][2] > 0.4 and kpts[7][2] > 0.4:
                # Calculate elbow angle
                v1 = (kpts[5][0] - kpts[7][0], kpts[5][1] - kpts[7][1])
                angle = math.atan2(v1[1], v1[0])
                state.pose_angles.append(angle)
                if len(state.pose_angles) >= 8:
                    pose_variance = float(np.std(state.pose_angles))

        # 4. Synthesize instantaneous liveness score
        # Rigid plastic mannequin / cardboard cutout has microscopic or zero energy (< 0.005)
        if micro_energy < 0.005 and pose_variance < 0.005:
            instant_liveness = 0.08  # Highly rigid plastic / frozen dummy
        elif micro_energy < 0.008:
            instant_liveness = 0.28  # Ambiguous static
        else:
            # Biological respiration / subtle micro-sway (> 0.008)
            instant_liveness = min(0.95, 0.50 + (micro_energy * 15.0) + (pose_variance * 5.0))

        # Exponential moving average filter
        state.liveness_score = 0.88 * state.liveness_score + 0.12 * instant_liveness

        # 5. Status Decision based on observation duration
        if state.frame_count >= MIN_EVAL_FRAMES:
            if state.liveness_score <= self.mannequin_threshold:
                if not state.is_mannequin:
                    logger.info(
                        f"[Liveness] Track #{track_id} confirmed as FAKE/MANNEQUIN "
                        f"(Score: {state.liveness_score:.2f}, Energy: {micro_energy:.4f})"
                    )
                state.is_mannequin = True
                state.status = "MANNEQUIN"
            elif state.liveness_score >= self.human_threshold:
                state.is_mannequin = False
                state.status = "REAL_HUMAN"
            else:
                state.status = "EVALUATING"
        else:
            state.status = "EVALUATING"

        return {
            "track_id": track_id,
            "liveness_score": round(state.liveness_score, 2),
            "is_mannequin": state.is_mannequin,
            "status": state.status,
            "micro_energy": round(micro_energy, 4),
        }

    def cleanup_inactive(self, active_track_ids: List[int]):
        """Remove tracks that are no longer being tracked."""
        active_set = set(active_track_ids)
        to_del = [tid for tid in self.tracks if tid not in active_set]
        for tid in to_del:
            del self.tracks[tid]


# Global singleton instance
human_liveness_detector = HumanLivenessDetector()
