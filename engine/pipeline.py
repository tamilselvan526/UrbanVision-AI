"""
Unified Crime Detection Pipeline
=================================
Orchestrates all 4 algorithms in a synchronized per-frame pipeline:
  Frame → Algorithm 1 (Detect) → Algorithm 2 (Track) → 
         Algorithm 3 (Pose) → Algorithm 4 (Threat) → Annotated Frame + Events
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from engine.detector import WeaponDetector, Detection
from engine.tracker import MultiObjectTracker
from engine.pose_estimator import PoseEstimator, PoseResult
from engine.threat_analyzer import ThreatAnalyzer, ThreatEvent
from engine.person_reid import person_reid
from engine.street_entity_profiler import street_entity_profiler
from engine.liveness_detector import human_liveness_detector
from engine.bike_cargo_analyzer import bike_cargo_analyzer
from engine.vehicle_tracker import vehicle_tracker
from engine.intelligence_db import intelligence_db, SNAPSHOTS_DIR


@dataclass
class PipelineConfig:
    """Configuration for the CrimeDetectionPipeline."""
    # Model paths (will auto-download if not found)
    person_model: str = "yolov8n.pt"
    weapon_model: Optional[str] = None
    pose_model: str = "yolov8n-pose.pt"
    tracker_config: str = "bytetrack.yaml"

    # Thresholds
    person_conf: float = 0.40
    weapon_conf: float = 0.35
    pose_conf: float = 0.35
    alert_threshold: float = 45.0

    # Display toggles
    show_boxes: bool = True
    show_trails: bool = True
    show_skeleton: bool = True
    show_threat_hud: bool = True
    show_kinematics: bool = True

    # Device
    device: Optional[str] = None  # None = auto (cuda if available)


@dataclass
class FrameResult:
    """Result of processing one video frame through the full pipeline."""
    frame_id: int = 0
    timestamp: float = field(default_factory=time.time)
    annotated_frame: Optional[np.ndarray] = None
    detections: List[Detection] = field(default_factory=list)
    pose_results: List[PoseResult] = field(default_factory=list)
    threat_events: List[ThreatEvent] = field(default_factory=list)
    fps: float = 0.0
    processing_time_ms: float = 0.0


class CrimeDetectionPipeline:
    """
    Master pipeline that runs all 4 algorithms per frame.
    
    Usage:
        pipeline = CrimeDetectionPipeline(config)
        result = pipeline.process_frame(bgr_frame)
        cv2.imshow("Crime Detection", result.annotated_frame)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        cfg = self.config

        # Initialize all 4 algorithm modules
        logger.info("=== Initializing Smart Crime Detection Pipeline ===")
        self.detector = WeaponDetector(
            person_model_path=cfg.person_model,
            weapon_model_path=cfg.weapon_model,
            person_conf=cfg.person_conf,
            weapon_conf=cfg.weapon_conf,
            device=cfg.device,
        )
        self.tracker = MultiObjectTracker(
            tracker_config=cfg.tracker_config,
            device=cfg.device,
        )
        self.pose_estimator = PoseEstimator(
            model_path=cfg.pose_model,
            conf=cfg.pose_conf,
            device=cfg.device,
        )
        self.threat_analyzer = ThreatAnalyzer(
            alert_threshold=cfg.alert_threshold,
        )

        self._frame_count = 0
        self._fps_history = []
        logger.success("=== Pipeline Ready — All 4 Algorithms Loaded ===")

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        """
        Process one BGR video frame through the full 4-algorithm pipeline.
        
        Returns:
            FrameResult with annotated frame and threat events.
        """
        t0 = time.perf_counter()
        self._frame_count += 1
        h, w = frame.shape[:2]

        # ── Algorithm 1: Weapon & Person Detection ─────────────────────
        detections = self.detector.detect(frame)

        # ── Algorithm 2: Multi-Object Tracking ─────────────────────────
        tracked_dets = self.tracker.update(
            frame=frame,
            detections=detections,
            person_model=self.detector.person_model,
        )
        all_tracks = self.tracker.get_all_tracks()

        # ── Algorithm 3: Pose Estimation ───────────────────────────────
        pose_results = self.pose_estimator.estimate(frame)

        # Match pose bounding boxes to tracked person IDs before computing motion kinematics
        tracked_persons = [d for d in tracked_dets if d.is_person and d.track_id >= 0]
        for pr in pose_results:
            tid = self.threat_analyzer._match_pose_to_person(pr, tracked_persons)
            if tid >= 0:
                pr.track_id = tid

        # Now update temporal motion features with valid track IDs
        self.pose_estimator.update_motion(pose_results)

        # Propagate safe action status from pose estimator to tracked persons
        pose_by_track = {pr.track_id: pr for pr in pose_results if pr.track_id >= 0}
        for det in tracked_dets:
            if det.is_person and det.track_id in pose_by_track:
                pr = pose_by_track[det.track_id]
                if getattr(pr, "is_safe_action", False):
                    det.is_safe_action = True
                    det.safe_action_label = getattr(pr, "safe_action_label", "HANDS RAISED (SAFE)")

        if getattr(self.threat_analyzer, "_demo_green_action", False):
            for det in tracked_dets:
                if det.is_person and not det.is_weapon:
                    det.is_safe_action = True
                    if not det.safe_action_label:
                        det.safe_action_label = "DEMO: HANDS RAISED (SAFE)"

        # ── Two-Wheeler Helmet & Cargo Disambiguation ──────────────────
        bike_cargo_analyzer.analyze_two_wheelers(
            frame_bgr=frame,
            vehicle_dets=tracked_dets,
            person_dets=[d for d in tracked_dets if d.is_person and d.track_id >= 0],
            pose_results=pose_results,
        )

        # ── Person Watchlist Re-ID Matching ────────────────────────────
        # For each tracked person, check visual appearance against enrolled gallery
        for det in tracked_dets:
            if det.is_person and det.track_id >= 0:
                px1, py1, px2, py2 = det.bbox
                pad = 8
                person_crop = frame[max(0, py1 - pad):min(h, py2 + pad), max(0, px1 - pad):min(w, px2 + pad)]
                if person_crop.size > 0:
                    match = person_reid.match_crop(person_crop)
                    if match:
                        matched_name, match_conf = match
                        det.target_match_name = matched_name
                        det.target_match_conf = match_conf

                        # Record sighting to intelligence database (rate limit: once per 30 frames per track)
                        if self._frame_count % 30 == 0:
                            s_name = f"target_{matched_name}_{det.track_id}_{int(time.time())}.jpg"
                            cv2.imwrite(str(SNAPSHOTS_DIR / "persons" / s_name), person_crop)
                            intelligence_db.record_sighting(
                                person_name=matched_name,
                                x=det.center[0],
                                y=det.center[1],
                                confidence=match_conf,
                                camera_source="Camera 01",
                                frame_id=self._frame_count,
                                snapshot_filename=s_name,
                            )

        # ── Human Liveness & Fake Mannequin Detection ──────────────────
        human_liveness_detector.update_frame(frame)
        pose_by_track = {pr.track_id: pr for pr in pose_results if pr.track_id >= 0}

        for det in tracked_dets:
            if det.is_person and det.track_id >= 0:
                p_pose = pose_by_track.get(det.track_id)
                live_info = human_liveness_detector.evaluate_person(
                    track_id=det.track_id,
                    bbox=det.bbox,
                    frame_bgr=frame,
                    pose_res=p_pose,
                )
                det.is_mannequin = live_info["is_mannequin"]
                det.liveness_score = live_info["liveness_score"]

        # ── Street Regulars & Routine Activity Profiling (Crowded Street AI) ───
        for det in tracked_dets:
            if det.is_person and det.track_id >= 0:
                p_pose = pose_by_track.get(det.track_id)
                reg_info = street_entity_profiler.process_person_detection(
                    track_id=det.track_id,
                    bbox=list(det.bbox),
                    frame_bgr=frame,
                    pose_res=p_pose,
                    is_mannequin=det.is_mannequin,
                )
                if reg_info.get("is_regular"):
                    det.regular_id = reg_info.get("regular_id")
                    det.regular_alias = reg_info.get("regular_alias")
                    det.regular_name = reg_info.get("custom_name")
                    det.regular_role = reg_info.get("role")
                    det.regular_conf = reg_info.get("role_confidence", 0.0)

        if self._frame_count % 90 == 0:
            street_entity_profiler.cleanup_inactive_tracks(max_idle_seconds=10.0)
            human_liveness_detector.cleanup_inactive([d.track_id for d in tracked_dets if d.track_id >= 0])

        # ── Vehicle Tracking & Number Plate Recognition (ANPR) ────────
        vehicle_dets = [d for d in tracked_dets if d.is_vehicle]
        if vehicle_dets:
            vehicle_tracker.update(frame, vehicle_dets, camera_source="Camera 01")

        # ── Algorithm 4: Spatio-Temporal Threat Analysis ────────────────
        threat_events = self.threat_analyzer.analyze(
            frame=frame,
            detections=tracked_dets,
            tracks=all_tracks,
            pose_results=pose_results,
        )

        # Escalate threat score for persons matched to the Watchlist
        for event in threat_events:
            matching_det = next((d for d in tracked_dets if d.track_id == event.person_track_id and d.target_match_name), None)
            if matching_det:
                event.action_tags.append(f"WATCHLIST: {matching_det.target_match_name.upper()}")
                event.threat_score = min(100.0, event.threat_score + 35.0)
                event.threat_label = f"TARGET: {matching_det.target_match_name.upper()}"

        # ── Compose annotated visualization ────────────────────────────
        vis = frame.copy()

        # Draw vehicles and license plates
        if vehicle_dets:
            vis = vehicle_tracker.draw(vis)

        if self.config.show_trails:
            vis = self.tracker.draw_trails(vis, tracked_dets)

        if self.config.show_boxes:
            vis = self.detector.draw(vis, tracked_dets)

        if self.config.show_kinematics:
            vis = self.tracker.draw_kinematics(vis, tracked_dets)

        if self.config.show_skeleton:
            vis = self.pose_estimator.draw_skeleton(vis, pose_results)

        if self.config.show_threat_hud:
            vis = self.threat_analyzer.draw_threat_overlay(vis, threat_events)

        # FPS counter
        t1 = time.perf_counter()
        proc_ms = (t1 - t0) * 1000
        fps = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0
        self._fps_history.append(fps)
        if len(self._fps_history) > 30:
            self._fps_history.pop(0)
        avg_fps = sum(self._fps_history) / len(self._fps_history)

        # FPS overlay
        cv2.putText(
            vis, f"FPS: {avg_fps:.1f}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
        )
        cv2.putText(
            vis, f"Frame: {self._frame_count}", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA
        )
        # Algorithm labels watermark
        cv2.putText(
            vis, "Alg1:YOLO  Alg2:ByteTrack  Alg3:Pose  Alg4:ThreatAI",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1, cv2.LINE_AA
        )

        return FrameResult(
            frame_id=self._frame_count,
            annotated_frame=vis,
            detections=tracked_dets,
            pose_results=pose_results,
            threat_events=threat_events,
            fps=avg_fps,
            processing_time_ms=proc_ms,
        )

    @property
    def incident_log(self) -> list:
        return self.threat_analyzer.incident_log
