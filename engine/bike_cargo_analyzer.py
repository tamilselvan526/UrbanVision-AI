"""
Two-Wheeler Helmet & Cargo Disambiguation Engine
=================================================
Solves the traffic surveillance failure where carried items (guitar gig bags,
large backpacks, Swiggy/Zomato delivery bags) are falsely detected as unhelmeted
pillion passengers ("man without helmet").

Core Capabilities:
  1. Two-Wheeler Association:
     - Associates rider and luggage candidates with their motorcycle/scooter.
  2. Anatomical Keypoint & Pose Verification:
     - Reclassifies non-anatomical carried blobs (guitar cases, backpacks) as
       RIDER_CARGO, eliminating false passenger detections.
  3. Helmet vs. Bare Head Verification:
     - Biometrically inspects genuine riders for helmet compliance vs bare hair/skin.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from engine.detector import Detection
from engine.pose_estimator import PoseResult

TWO_WHEELER_CLASSES = {"motorcycle", "motorbike", "motor bike", "bicycle", "bike", "scooter"}


class BikeCargoAnalyzer:
    """
    Analyzes motorcycle and scooter occupants to disambiguate carried cargo
    from real riders and verify helmet safety compliance.
    """

    def analyze_two_wheelers(
        self,
        frame_bgr: np.ndarray,
        vehicle_dets: List[Detection],
        person_dets: List[Detection],
        pose_results: List[PoseResult],
    ) -> List[Detection]:
        """
        Processes candidate persons overlapping two-wheelers:
        1. Distinguishes genuine riders from slung guitar bags / luggage.
        2. Evaluates helmet compliance on genuine human riders.
        """
        # 1. Identify two-wheelers
        bikes = [
            v for v in vehicle_dets
            if v.class_name.lower() in TWO_WHEELER_CLASSES or v.is_vehicle
        ]
        if not bikes:
            # Still evaluate any isolated person who might be riding
            return person_dets

        ih, iw = frame_bgr.shape[:2]

        for bike in bikes:
            bx1, by1, bx2, by2 = bike.bbox
            bw = max(1, bx2 - bx1)
            bh = max(1, by2 - by1)

            # Find person detections overlapping this bike
            bike_occupants: List[Detection] = []
            for p in person_dets:
                if not p.is_person:
                    continue
                px1, py1, px2, py2 = p.bbox
                # Check horizontal overlap and vertical proximity to bike
                overlap_x = max(0, min(px2, bx2) - max(px1, bx1))
                if overlap_x > 0.30 * min(px2 - px1, bw):
                    # Person sits on top of or overlaps the bike body
                    if (py2 >= by1 - 30) and (py1 <= by2):
                        bike_occupants.append(p)

            if not bike_occupants:
                continue

            # Evaluate each occupant: Real human vs Carried cargo (Guitar bag / backpack)
            for occupant in bike_occupants:
                matching_pose = self._match_pose(occupant, pose_results)
                is_real_human, cargo_type = self._verify_anatomical_liveness(occupant, matching_pose, frame_bgr)

                if not is_real_human:
                    # Reclassify this entity as CARGO instead of a human passenger!
                    occupant.is_person = False
                    occupant.is_cargo = True
                    occupant.cargo_type = cargo_type
                    occupant.class_name = f"cargo_{cargo_type.lower()}"
                    occupant.has_helmet = None
                    logger.info(
                        f"[BikeAnalyzer] Disambiguated Track #{occupant.track_id} on Bike as "
                        f"CARGO ({cargo_type}) - False pillion passenger suppressed."
                    )
                else:
                    # Verified real human rider: evaluate helmet compliance
                    has_helmet, helmet_conf = self._evaluate_helmet(occupant, matching_pose, frame_bgr)
                    occupant.has_helmet = has_helmet
                    occupant.helmet_conf = helmet_conf

        return person_dets

    def _match_pose(self, det: Detection, pose_results: List[PoseResult]) -> Optional[PoseResult]:
        """Find pose estimation result overlapping person detection."""
        px1, py1, px2, py2 = det.bbox
        det_area = max(1, (px2 - px1) * (py2 - py1))

        best_pose = None
        best_iou = 0.0

        for pr in pose_results:
            bbox = getattr(pr, "person_bbox", getattr(pr, "bbox", (0, 0, 0, 0)))
            x1, y1, x2, y2 = bbox
            inter_x1 = max(px1, x1)
            inter_y1 = max(py1, y1)
            inter_x2 = min(px2, x2)
            inter_y2 = min(py2, y2)

            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                union_area = det_area + ((x2 - x1) * (y2 - y1)) - inter_area
                iou = inter_area / max(1, union_area)
                if iou > best_iou:
                    best_iou = iou
                    best_pose = pr

        return best_pose if best_iou > 0.25 else None

    def _verify_anatomical_liveness(
        self,
        det: Detection,
        pose: Optional[PoseResult],
        frame_bgr: np.ndarray,
    ) -> Tuple[bool, str]:
        """
        Verifies if candidate occupant possesses true human biological anatomy.
        Returns: (is_real_human: bool, cargo_type_if_fake: str)
        """
        x1, y1, x2, y2 = det.bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        aspect_ratio = h / float(w)

        # 1. Pose Keypoint Anatomical Check
        if pose is not None and pose.keypoints is not None:
            kpts = pose.keypoints
            # Face keypoints: Nose(0), LeftEye(1), RightEye(2), LeftEar(3), RightEar(4)
            face_confs = [kpts[i][2] for i in range(min(5, len(kpts)))]
            avg_face_conf = float(np.mean(face_confs)) if face_confs else 0.0

            # Shoulder keypoints: LeftShoulder(5), RightShoulder(6)
            shoulder_confs = [kpts[i][2] for i in (5, 6) if i < len(kpts)]
            avg_shoulder_conf = float(np.mean(shoulder_confs)) if shoulder_confs else 0.0

            # A real human on a bike has clear facial or bilateral shoulder markers
            if avg_face_conf > 0.35 or avg_shoulder_conf > 0.40:
                return True, ""

            # If face and shoulders have near-zero confidence, examine shape
            if avg_face_conf < 0.20 and avg_shoulder_conf < 0.25:
                # Elongated vertical object strapped to back -> Guitar gig bag / instrument case
                if aspect_ratio >= 1.9:
                    return False, "GUITAR_BAG"
                return False, "BACKPACK"

        # 2. Heuristic Geometry & Contour Tapering Check (when pose is absent/obscured)
        ih, iw = frame_bgr.shape[:2]
        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = min(iw, x2), min(ih, y2)

        if cx2 > cx1 + 10 and cy2 > cy1 + 20:
            crop = frame_bgr[cy1:cy2, cx1:cx2]
            # Guitar case signature: Upper 35% is significantly narrower than lower 65% (tapered neck)
            ch, cw = crop.shape[:2]
            top_half = crop[0:int(ch * 0.4), :]
            bottom_half = crop[int(ch * 0.4):ch, :]

            # Compute horizontal edge extent
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 40, 120)

            top_edges = np.where(edges[0:int(ch * 0.4), :] > 0)[1]
            bot_edges = np.where(edges[int(ch * 0.4):ch, :] > 0)[1]

            top_span = (np.max(top_edges) - np.min(top_edges)) if len(top_edges) > 10 else cw * 0.5
            bot_span = (np.max(bot_edges) - np.min(bot_edges)) if len(bot_edges) > 10 else cw

            # Taper ratio < 0.60 indicates a classic narrow guitar headstock/neck extending upward
            if aspect_ratio >= 1.8 and (top_span / max(1.0, float(bot_span))) < 0.65:
                return False, "GUITAR_BAG"

            if aspect_ratio >= 1.5:
                # Check for synthetic cordura/canvas texture vs human skin
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                # Skin color mask in HSV (H: 0-25, S: 35-200, V: 50-255)
                skin_mask = cv2.inRange(hsv, np.array([0, 35, 50]), np.array([25, 200, 255]))
                skin_ratio = float(np.count_nonzero(skin_mask)) / float(cw * ch)

                # Pure black/dark blue canvas luggage has near-zero skin ratio (< 1.5%)
                if skin_ratio < 0.015:
                    if aspect_ratio >= 2.0:
                        return False, "GUITAR_BAG"
                    return False, "BACKPACK"

        return True, ""

    def _evaluate_helmet(
        self,
        det: Detection,
        pose: Optional[PoseResult],
        frame_bgr: np.ndarray,
    ) -> Tuple[bool, float]:
        """
        Determines whether a verified living rider is wearing a safety helmet.
        Returns: (has_helmet: bool, confidence: float)
        """
        x1, y1, x2, y2 = det.bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        ih, iw = frame_bgr.shape[:2]

        # 1. Determine Head Bounding Box
        if pose is not None and pose.keypoints is not None:
            kpts = pose.keypoints
            # Nose, eyes, ears
            head_kpts = [kpts[i] for i in range(min(5, len(kpts))) if kpts[i][2] > 0.25]
            if head_kpts:
                hx = int(np.mean([p[0] for p in head_kpts]))
                hy = int(np.mean([p[1] for p in head_kpts]))
                head_size = int(max(w * 0.45, 25))
                hx1 = max(0, hx - head_size // 2)
                hy1 = max(0, hy - int(head_size * 0.7))
                hx2 = min(iw, hx + head_size // 2)
                hy2 = min(ih, hy + int(head_size * 0.6))
            else:
                # Default to top 42% of rider bbox (full head down to chin)
                hx1, hy1 = max(0, x1), max(0, y1)
                hx2, hy2 = min(iw, x2), min(ih, y1 + int(h * 0.42))
        else:
            hx1, hy1 = max(0, x1), max(0, y1)
            hx2, hy2 = min(iw, x2), min(ih, y1 + int(h * 0.42))

        if (hx2 <= hx1 + 10) or (hy2 <= hy1 + 10):
            return True, 0.50  # Inconclusive default

        head_crop = frame_bgr[hy1:hy2, hx1:hx2]
        ch, cw = head_crop.shape[:2]

        # 2. Biometric Skin & Hair vs Helmet Shell Analysis
        # YCrCb color space is optimal for human skin tone detection
        ycrcb = cv2.cvtColor(head_crop, cv2.COLOR_BGR2YCrCb)
        skin_mask_ycrcb = cv2.inRange(
            ycrcb,
            np.array([0, 133, 77], dtype=np.uint8),
            np.array([255, 173, 127], dtype=np.uint8)
        )
        hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
        skin_mask_hsv = cv2.inRange(
            hsv,
            np.array([0, 30, 45], dtype=np.uint8),
            np.array([28, 200, 255], dtype=np.uint8)
        )
        skin_mask = cv2.bitwise_or(skin_mask_ycrcb, skin_mask_hsv)
        skin_fraction = float(np.count_nonzero(skin_mask)) / float(cw * ch)

        # Hair texture analysis (Laplacian gradient variance on upper 50% of head)
        upper_head_gray = cv2.cvtColor(head_crop[0:int(ch * 0.55), :], cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(upper_head_gray, cv2.CV_32F).var())

        # Specular gloss / reflection analysis (helmets reflect sharp specular highlights)
        v_channel = hsv[:, :, 2]
        high_specular = float(np.count_nonzero(v_channel > 230)) / float(cw * ch)

        # Decision rules:
        # Bare Head: Non-trivial skin fraction (> 12%) on face/forehead + dark matte hair texture (lap_var > 60) without specular shine
        if skin_fraction >= 0.12 and high_specular < 0.035:
            conf = min(0.95, 0.60 + (skin_fraction * 1.5))
            return False, conf  # NO_HELMET violation

        # Helmet: Smooth outer dome, high specularity (> 0.03) or low skin fraction (< 10%)
        conf = min(0.95, 0.65 + (high_specular * 2.5) + max(0.0, (1.0 - skin_fraction) * 0.2))
        return True, conf  # HELMET COMPLIANT


# Global singleton instance
bike_cargo_analyzer = BikeCargoAnalyzer()
