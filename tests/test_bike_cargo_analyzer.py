"""
Test Suite for Two-Wheeler Helmet & Cargo Disambiguation AI
============================================================
Verifies:
  1. Guitar gig bag / backpack on a motorcycle rider is disambiguated as CARGO,
     preventing false pillion passenger detection and false "no helmet" fines.
  2. Genuine rider wearing helmet is verified as HELMET OK.
  3. Genuine unhelmeted rider is correctly flagged with NO HELMET VIOLATION.
"""

import sys
import time
import numpy as np
import cv2
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from engine.bike_cargo_analyzer import BikeCargoAnalyzer
from engine.detector import Detection
from engine.pose_estimator import PoseResult


def test_guitar_bag_on_bike_disambiguation():
    """Verify that an elongated guitar case on a rider's back is reclassified as CARGO and not an unhelmeted human."""
    analyzer = BikeCargoAnalyzer()
    frame = np.full((480, 640, 3), 100, dtype=np.uint8)

    # 1. Motorcycle detection
    bike_det = Detection(
        track_id=10,
        class_name="motorcycle",
        bbox=(100, 200, 380, 450),
        confidence=0.88,
        is_vehicle=True,
    )

    # 2. Genuine Rider detection (with real human pose)
    rider_det = Detection(
        track_id=1,
        class_name="person",
        bbox=(140, 80, 250, 320),
        confidence=0.85,
        is_person=True,
    )

    # Mock real pose keypoints for the rider (high confidence on face and shoulders)
    # Nose, L_eye, R_eye, L_ear, R_ear, L_sh, R_sh
    rider_kpts = np.zeros((17, 3), dtype=np.float32)
    rider_kpts[0] = [190, 110, 0.85]  # Nose
    rider_kpts[1] = [185, 105, 0.82]  # L Eye
    rider_kpts[2] = [195, 105, 0.82]  # R Eye
    rider_kpts[5] = [165, 145, 0.88]  # L Shoulder
    rider_kpts[6] = [215, 145, 0.88]  # R Shoulder

    rider_pose = PoseResult(
        person_bbox=(140, 80, 250, 320),
        confidence_score=0.85,
        keypoints=rider_kpts,
        track_id=1,
    )

    # 3. Guitar Bag carried behind rider (narrow, tall, aspect ratio > 2.2:1, NO anatomical face keypoints)
    guitar_det = Detection(
        track_id=2,
        class_name="person",  # Legacy YOLO misclassified bag as person!
        bbox=(190, 50, 250, 280),  # Width=60, Height=230 -> Aspect ratio = 3.83:1
        confidence=0.72,
        is_person=True,
    )

    # Guitar pose has zero or phantom face confidence (< 0.15)
    guitar_kpts = np.zeros((17, 3), dtype=np.float32)
    guitar_kpts[0] = [220, 70, 0.08]
    guitar_kpts[1] = [218, 68, 0.05]
    guitar_kpts[5] = [210, 120, 0.10]

    guitar_pose = PoseResult(
        person_bbox=(190, 50, 250, 280),
        confidence_score=0.40,
        keypoints=guitar_kpts,
        track_id=2,
    )

    # Run analyzer
    analyzer.analyze_two_wheelers(
        frame_bgr=frame,
        vehicle_dets=[bike_det],
        person_dets=[rider_det, guitar_det],
        pose_results=[rider_pose, guitar_pose],
    )

    # Assertions:
    # 1. Guitar bag must NOT be a person!
    assert guitar_det.is_person is False, "Guitar bag should NOT be classified as a person!"
    assert guitar_det.is_cargo is True, "Guitar bag should be tagged as CARGO!"
    assert guitar_det.cargo_type == "GUITAR_BAG", f"Expected GUITAR_BAG, got {guitar_det.cargo_type}"
    # 2. Guitar bag must NOT have a helmet violation assigned to it
    assert guitar_det.has_helmet is None, "Guitar bag should not have helmet compliance evaluation!"

    # 3. Rider should remain a person
    assert rider_det.is_person is True
    assert rider_det.is_cargo is False


def test_helmet_compliance_evaluation():
    """Verify that real riders are tested for helmets vs bare head."""
    analyzer = BikeCargoAnalyzer()

    # Synthetic frame with bare head (skin tone + dark matte hair)
    frame_bare = np.full((300, 300, 3), 100, dtype=np.uint8)
    # Lower face skin tone (BGR for Indian/olive skin: e.g. B=95, G=130, R=180)
    cv2.rectangle(frame_bare, (120, 50), (180, 90), (95, 130, 180), -1)
    # Dark hair on top with texture
    cv2.rectangle(frame_bare, (120, 20), (180, 50), (20, 20, 20), -1)
    # Add texture noise to hair
    noise = np.random.randint(0, 50, (30, 60, 3), dtype=np.uint8)
    frame_bare[20:50, 120:180] = np.clip(frame_bare[20:50, 120:180] + noise, 0, 255)

    det_bare = Detection(track_id=1, bbox=(100, 10, 200, 180), is_person=True)
    has_helmet_bare, conf_bare = analyzer._evaluate_helmet(det_bare, None, frame_bare)
    assert has_helmet_bare is False, "Bare head with skin and hair should be flagged as NO HELMET!"

    # Synthetic frame with helmet (bright glossy shell with high specular shine)
    frame_helmet = np.full((300, 300, 3), 100, dtype=np.uint8)
    # Smooth helmet dome
    cv2.circle(frame_helmet, (150, 50), 35, (30, 30, 180), -1)
    # Specular bright gloss spot
    cv2.circle(frame_helmet, (145, 40), 8, (255, 255, 255), -1)

    det_helmet = Detection(track_id=2, bbox=(100, 10, 200, 180), is_person=True)
    has_helmet_helm, conf_helm = analyzer._evaluate_helmet(det_helmet, None, frame_helmet)
    assert has_helmet_helm is True, "Glossy helmet dome should be evaluated as HELMET OK!"


if __name__ == "__main__":
    print("Testing guitar bag disambiguation on motorcycle...")
    test_guitar_bag_on_bike_disambiguation()
    print("PASS: test_guitar_bag_on_bike_disambiguation")

    print("Testing helmet compliance evaluation...")
    test_helmet_compliance_evaluation()
    print("PASS: test_helmet_compliance_evaluation")

    print("=== ALL BIKE CARGO & HELMET AI TESTS PASSED! ===")
