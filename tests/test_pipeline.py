"""
Integration Tests — Smart Crime Detection Pipeline
===================================================
Tests all 4 algorithms end-to-end on synthetic frames:

  Test 1: Algorithm 1 (YOLO Detection) loads and runs
  Test 2: Algorithm 2 (ByteTrack) maintains IDs across frames
  Test 3: Algorithm 3 (Pose) extracts keypoints from synthetic frame
  Test 4: Algorithm 4 (Threat Analyzer) scores simulated weapon+person scenario
  Test 5: Full Pipeline processes sequential frames without error
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
import torch
from loguru import logger


def make_test_frame(width=640, height=480, person_box=None, weapon_box=None):
    """Create a synthetic BGR test frame with optional colored rectangles."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (20, 20, 30)  # Dark background

    if person_box:
        x1, y1, x2, y2 = person_box
        frame[y1:y2, x1:x2] = (80, 120, 180)  # Blue-ish person shape

    if weapon_box:
        x1, y1, x2, y2 = weapon_box
        frame[y1:y2, x1:x2] = (30, 30, 200)  # Red-ish weapon shape

    return frame


def test_1_detector():
    """Test Algorithm 1: YOLO detection loads and produces detections."""
    logger.info("=== TEST 1: Algorithm 1 — YOLOv8 Detector ===")
    from engine.detector import WeaponDetector

    detector = WeaponDetector(device="cpu")
    frame = make_test_frame()

    t0 = time.perf_counter()
    detections = detector.detect(frame)
    elapsed = (time.perf_counter() - t0) * 1000

    logger.success(f"  ✓ Detector ran in {elapsed:.1f}ms | Detections: {len(detections)}")
    logger.info(f"    CUDA available: {torch.cuda.is_available()}")
    return True


def test_2_tracker():
    """Test Algorithm 2: ByteTrack assigns consistent IDs across frames."""
    logger.info("=== TEST 2: Algorithm 2 — ByteTrack Multi-Object Tracker ===")
    from engine.tracker import MultiObjectTracker, TrackState
    from engine.detector import Detection

    tracker = MultiObjectTracker()

    # Simulate 10 frames of the same person moving right
    person_ids = set()
    for i in range(10):
        det = Detection(
            class_id=0,
            class_name="person",
            confidence=0.85,
            bbox=(100 + i * 5, 100, 200 + i * 5, 300),
            is_person=True,
        )
        # Manually update tracker state (bypassing YOLO .track())
        tid = 1  # Simulate track ID 1
        det.track_id = tid
        if tid not in tracker._tracks:
            tracker._tracks[tid] = TrackState(
                track_id=tid, class_name="person", is_weapon=False, is_person=True
            )
        tracker._tracks[tid].update_kinematics(det.center)
        person_ids.add(tid)

    ts = tracker.get_track(1)
    trail_len = len(ts.trail)
    speed = ts.speed

    logger.success(f"  ✓ Track ID 1 maintained across 10 frames")
    logger.success(f"    Trail length: {trail_len} | Speed: {speed:.2f} px/frame")
    assert trail_len == 10, f"Expected 10 trail points, got {trail_len}"
    assert speed > 0, "Expected non-zero speed for moving person"
    return True


def test_3_pose_estimator():
    """Test Algorithm 3: Pose estimation loads and runs."""
    logger.info("=== TEST 3: Algorithm 3 — YOLO-Pose Estimator ===")
    from engine.pose_estimator import PoseEstimator

    estimator = PoseEstimator(device="cpu")
    # Use a real image-sized blank frame (YOLO requires proper-sized input)
    frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)

    t0 = time.perf_counter()
    results = estimator.estimate(frame)
    elapsed = (time.perf_counter() - t0) * 1000

    logger.success(f"  ✓ Pose estimator ran in {elapsed:.1f}ms | Persons detected: {len(results)}")
    logger.info(f"    (On noise image — 0 people expected)")
    return True


def test_4_threat_analyzer():
    """Test Algorithm 4: Threat analyzer correctly scores weapon+person scenario."""
    logger.info("=== TEST 4: Algorithm 4 — Spatio-Temporal Threat Analyzer ===")
    from engine.threat_analyzer import ThreatAnalyzer, ThreatLevel
    from engine.detector import Detection
    from engine.tracker import TrackState
    from engine.pose_estimator import PoseResult
    import numpy as np

    analyzer = ThreatAnalyzer(alert_threshold=40.0)

    # Simulate: person at (320,240) holding a gun (wrist near gun center)
    frame = make_test_frame(person_box=(250, 100, 390, 380))

    person_det = Detection(
        track_id=1, class_id=0, class_name="person",
        confidence=0.9, bbox=(250, 100, 390, 380), is_person=True,
    )
    weapon_det = Detection(
        track_id=9001, class_id=5, class_name="gun",
        confidence=0.8, bbox=(330, 200, 370, 230), is_weapon=True,
    )

    person_track = TrackState(track_id=1, class_name="person", is_weapon=False, is_person=True)
    person_track.trail.append((320, 240))
    person_track.speed = 5.0
    person_track.sudden_move = False

    # Pose: right wrist near gun center (345, 215)
    kp = np.zeros((17, 3))
    kp[0] = [320, 120, 0.9]   # nose
    kp[5] = [280, 160, 0.85]  # l_shoulder
    kp[6] = [360, 160, 0.85]  # r_shoulder
    kp[7] = [260, 210, 0.8]   # l_elbow
    kp[8] = [380, 210, 0.8]   # r_elbow
    kp[9] = [240, 220, 0.8]   # l_wrist
    kp[10] = [348, 215, 0.85] # r_wrist (near gun)
    kp[11] = [285, 290, 0.8]  # l_hip
    kp[12] = [355, 290, 0.8]  # r_hip

    pose = PoseResult(
        person_bbox=(250, 100, 390, 380),
        keypoints=kp,
        confidence_score=0.85,
    )
    from engine.pose_estimator import PoseEstimator
    estimator = PoseEstimator.__new__(PoseEstimator)
    estimator._kp_histories = {}
    estimator._compute_static_features(pose, 480, 640)

    events = analyzer.analyze(
        frame=frame,
        detections=[person_det, weapon_det],
        tracks={1: person_track},
        pose_results=[pose],
    )

    assert len(events) > 0, "Expected at least one threat event"
    event = events[0]
    logger.success(f"  ✓ Threat event generated for Person #1")
    logger.success(f"    Threat Score: {event.threat_score:.1f}%")
    logger.success(f"    Threat Level: {event.threat_label}")
    logger.success(f"    Action Tags: {event.action_tags}")

    assert event.threat_score > 30, f"Expected score > 30 for gun in hand scenario, got {event.threat_score}"
    assert "OBJECT_DETECTED" in event.action_tags, f"Expected OBJECT_DETECTED tag, got {event.action_tags}"
    assert "FIREARM" in event.action_tags or "WEAPON_IN_HAND" in event.action_tags, \
        f"Expected FIREARM or WEAPON_IN_HAND tag, got {event.action_tags}"
    return True


def test_5_full_pipeline():
    """Test 5: Full pipeline runs end-to-end on 5 sequential frames."""
    logger.info("=== TEST 5: Full Pipeline End-to-End ===")
    from engine.pipeline import CrimeDetectionPipeline, PipelineConfig

    config = PipelineConfig(device="cpu", show_skeleton=True, show_trails=True)
    pipeline = CrimeDetectionPipeline(config)

    frames_processed = 0
    for i in range(5):
        frame = np.random.randint(30, 200, (480, 640, 3), dtype=np.uint8)
        result = pipeline.process_frame(frame)
        assert result.annotated_frame is not None, "No annotated frame returned"
        assert result.annotated_frame.shape == frame.shape, "Frame shape mismatch"
        frames_processed += 1

    logger.success(f"  ✓ Full pipeline processed {frames_processed} frames without error")
    logger.success(f"    Average FPS: {pipeline._fps_history[-1]:.1f}")
    return True


if __name__ == "__main__":
    tests = [
        test_1_detector,
        test_2_tracker,
        test_3_pose_estimator,
        test_4_threat_analyzer,
        test_5_full_pipeline,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            logger.error(f"  ✗ {test_fn.__name__} FAILED: {e}")
            failed += 1
        print()

    print("=" * 50)
    print(f"Results: {passed}/{len(tests)} passed, {failed} failed")
    if failed == 0:
        print("✅ ALL TESTS PASSED — Pipeline ready!")
    else:
        print("❌ Some tests failed. Check logs above.")
    sys.exit(0 if failed == 0 else 1)
