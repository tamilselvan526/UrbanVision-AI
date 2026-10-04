"""
Unit and Integration Tests for Human Liveness & Fake Mannequin Detection
========================================================================
Tests:
  1. Moving human classification (gross motion -> REAL_HUMAN).
  2. Frozen static mannequin detection (zero internal micro-motion -> MANNEQUIN).
  3. Stationary living human with biological respiration (micro-motion energy -> REAL_HUMAN).
  4. Threat analyzer suppression: ensuring mannequins never trigger threat alerts.
"""

import sys
import time
import numpy as np
import cv2
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from engine.liveness_detector import HumanLivenessDetector
from engine.detector import Detection
from engine.threat_analyzer import ThreatAnalyzer


def test_moving_human_liveness():
    """Verify that any person with active physical movement is instantly classified as living."""
    detector = HumanLivenessDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Simulate person moving from x=100 to x=250 across 10 frames
    for i in range(12):
        x = 100 + i * 15
        bbox = (x, 100, x + 80, 300)
        res = detector.evaluate_person(track_id=1, bbox=bbox, frame_bgr=frame)

    assert res["is_mannequin"] is False
    assert res["liveness_score"] >= 0.85
    assert res["status"] == "REAL_HUMAN"


def test_static_mannequin_detection():
    """Verify that a rigid, completely frozen display figure is detected as MANNEQUIN after observation window."""
    detector = HumanLivenessDetector()
    # Create static synthetic mannequin image (e.g. plastic clothing dummy)
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    # Give mannequin some static painted textures
    cv2.rectangle(frame, (200, 100), (280, 300), (180, 160, 140), -1)
    cv2.circle(frame, (240, 130), 20, (200, 180, 160), -1)

    bbox = (200, 100, 280, 300)

    # Feed 35 identical frames (completely frozen plastic dummy)
    for _ in range(35):
        detector.update_frame(frame)
        res = detector.evaluate_person(track_id=99, bbox=bbox, frame_bgr=frame)

    assert res["is_mannequin"] is True
    assert res["liveness_score"] <= 0.30
    assert res["status"] == "MANNEQUIN"


def test_stationary_living_human_with_respiration():
    """Verify that a human standing still with subtle biological chest breathing is recognized as living."""
    detector = HumanLivenessDetector()
    bbox = (200, 100, 280, 300)

    for i in range(35):
        # Base frame
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        # Simulate breathing oscillation (sinusoidal expansion & subtle luminance micro-shift)
        breath_offset = int(3 * np.sin(i * 0.4))
        # Torso with subtle biological micro-changes
        torso_val = 140 + int(15 * np.sin(i * 0.5))
        cv2.rectangle(frame, (200 + breath_offset, 140), (280 - breath_offset, 250), (torso_val, 120, 110), -1)

        detector.update_frame(frame)
        res = detector.evaluate_person(track_id=42, bbox=bbox, frame_bgr=frame)

    # Living human should NOT be flagged as mannequin
    assert res["is_mannequin"] is False
    assert res["liveness_score"] >= 0.40


def test_threat_suppression_on_mannequin():
    """Verify that detected mannequins never produce assault/weapon alerts."""
    analyzer = ThreatAnalyzer(alert_threshold=0.30)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Detection 1: mannequin with knife/weapon assigned or nearby
    mannequin_det = Detection(
        track_id=5,
        class_name="person",
        bbox=(100, 100, 180, 300),
        confidence=0.85,
        is_mannequin=True,  # Tagged as mannequin
        liveness_score=0.15,
    )
    weapon_det = Detection(
        track_id=6,
        class_name="knife",
        bbox=(120, 150, 140, 190),
        confidence=0.80,
        is_weapon=True,
    )

    events = analyzer.analyze(
        frame=frame,
        detections=[mannequin_det, weapon_det],
        tracks={},
        pose_results=[],
    )

    # Since person 5 is a mannequin, all threats must be suppressed!
    assert len(events) == 0, f"Expected 0 threat events for mannequin, got {len(events)}"


if __name__ == "__main__":
    print("Testing moving human liveness...")
    test_moving_human_liveness()
    print("PASS: test_moving_human_liveness")

    print("Testing static mannequin detection...")
    test_static_mannequin_detection()
    print("PASS: test_static_mannequin_detection")

    print("Testing stationary living human with respiration...")
    test_stationary_living_human_with_respiration()
    print("PASS: test_stationary_living_human_with_respiration")

    print("Testing threat suppression on mannequin...")
    test_threat_suppression_on_mannequin()
    print("PASS: test_threat_suppression_on_mannequin")

    print("=== ALL 4 LIVENESS AI TESTS PASSED! ===")
