"""
Test Suite for Street Entity Profiler & Routine Activity Discovery
===================================================================
Tests:
  1. Database CRUD operations for street regulars.
  2. Crowd-mode upper-body weighted Re-ID embedding extraction.
  3. Spatio-temporal kinematic role classification:
     - STREET_VENDOR
     - SHOPKEEPER
     - COMMUTER
     - SUSPICIOUS_LOITERER
  4. Auto-enrollment and recognition of recurring entities.
"""

import sys
import time
import numpy as np
import cv2
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from engine.intelligence_db import intelligence_db
from engine.person_reid import person_reid
from engine.street_entity_profiler import StreetEntityProfiler, TrackSession, STATIONARY_RADIUS_PX


def test_db_street_regulars():
    """Test inserting, updating, fetching, and deleting street regulars in SQLite."""
    alias = f"TEST_REG_{int(time.time())}"
    dummy_emb = np.random.randn(392).astype(np.float32)
    dummy_emb /= np.linalg.norm(dummy_emb)

    # 1. Insert new regular
    reg_id = intelligence_db.add_or_update_regular(
        alias=alias,
        embedding=dummy_emb,
        detected_role="STREET_VENDOR",
        role_confidence=0.85,
        dwell_delta=120.0,
        anchor=(100, 200),
        active_hour=14,
        snapshot_filename="test_snapshot.jpg",
        custom_name="Test Vendor Chai",
    )
    assert reg_id > 0

    # 2. Retrieve by ID
    reg = intelligence_db.get_regular_by_id(reg_id)
    assert reg is not None
    assert reg["alias"] == alias
    assert reg["custom_name"] == "Test Vendor Chai"
    assert reg["detected_role"] == "STREET_VENDOR"
    assert reg["anchor_x"] == 100
    assert reg["anchor_y"] == 200
    assert 14 in reg["active_hours"]

    # 3. Update existing regular stats (simulate next sighting)
    new_emb = np.random.randn(392).astype(np.float32)
    new_emb /= np.linalg.norm(new_emb)
    reg_id_updated = intelligence_db.add_or_update_regular(
        alias=alias,
        embedding=new_emb,
        detected_role="STREET_VENDOR",
        role_confidence=0.90,
        dwell_delta=60.0,
        anchor=(105, 202),
        active_hour=15,
    )
    assert reg_id_updated == reg_id

    reg_after = intelligence_db.get_regular_by_id(reg_id)
    assert reg_after["total_appearances"] == 2
    assert reg_after["total_dwell_seconds"] == 180.0
    assert 15 in reg_after["active_hours"]

    # 4. Update custom name
    ok = intelligence_db.update_regular_custom_name(reg_id, "Master Chef Suresh")
    assert ok is True
    assert intelligence_db.get_regular_by_id(reg_id)["custom_name"] == "Master Chef Suresh"

    # 5. Load all embeddings
    all_embs = intelligence_db.load_all_regular_embeddings()
    match = next((r for r in all_embs if r["id"] == reg_id), None)
    assert match is not None
    assert match["embedding"].shape[0] == 392

    # 6. Delete regular
    del_ok = intelligence_db.delete_regular(reg_id)
    assert del_ok is True
    assert intelligence_db.get_regular_by_id(reg_id) is None


def test_crowd_mode_embedding():
    """Verify crowd occlusion upper-body weighted embedding."""
    # Create synthetic dummy person image (128x256 BGR)
    # Head: top 30%, Torso: 30-65%, Legs: bottom 35%
    img = np.zeros((256, 128, 3), dtype=np.uint8)
    # Give torso distinctive bright orange color
    img[75:160, :] = (0, 140, 255)
    # Give head skin-like tone
    img[0:75, :] = (130, 180, 220)
    # Give legs dark blue jeans
    img[160:256, :] = (120, 40, 20)

    emb = person_reid.extract_embedding(img, crowd_mode=True)
    assert emb.shape[0] == 392
    # Verify L2 normalization
    norm = np.linalg.norm(emb)
    assert abs(norm - 1.0) < 1e-4

    # Now occlude lower limbs completely (black / cart occlusion)
    img_occluded = img.copy()
    img_occluded[160:256, :] = 0  # Legs fully hidden behind street crowd / cart

    emb_occluded = person_reid.extract_embedding(img_occluded, crowd_mode=True)
    # In crowd_mode, cosine similarity should remain very high (> 0.85) because legs have low weight!
    sim = float(np.dot(emb, emb_occluded))
    assert sim >= 0.85, f"Expected similarity >= 0.85 under leg occlusion, got {sim:.3f}"


def test_role_activity_classification():
    """Verify autonomous role inference based on motion kinematics."""
    profiler = StreetEntityProfiler(match_threshold=0.74)

    # 1. Test Vendor / Stationary worker
    vendor_session = TrackSession(
        track_id=10,
        first_seen=time.time() - 25.0,  # 25 seconds ago
        anchor_x=300,
        anchor_y=400,
        stationary_frames=50,
        upper_gesture_frames=6,
    )
    vendor_session.frame_count = 60
    profiler._classify_role(vendor_session)
    assert vendor_session.classified_role == "STREET_VENDOR"
    assert vendor_session.role_confidence > 0.70

    # 2. Test Commuter (Fast transit, low dwell)
    commuter_session = TrackSession(
        track_id=11,
        first_seen=time.time() - 8.0,  # 8 seconds ago
        anchor_x=100,
        anchor_y=100,
    )
    commuter_session.frame_count = 30
    # Simulate movement from (100, 100) to (350, 120) over 6 seconds -> speed = ~41 px/s
    commuter_session.positions.append((100, 100, time.time() - 6.0))
    commuter_session.positions.append((350, 120, time.time()))
    profiler._classify_role(commuter_session)
    assert commuter_session.classified_role in ("COMMUTER", "DELIVERY_WORKER")
    assert commuter_session.role_confidence > 0.70

    # 3. Test Suspicious Loiterer (High dwell, low stationarity / wandering)
    loiterer_session = TrackSession(
        track_id=12,
        first_seen=time.time() - 35.0,  # 35 seconds ago
        anchor_x=500,
        anchor_y=500,
        stationary_frames=10,  # only 10 stationary out of 60
    )
    loiterer_session.frame_count = 60
    profiler._classify_role(loiterer_session)
    assert loiterer_session.classified_role == "SUSPICIOUS_LOITERER"
    assert loiterer_session.role_confidence > 0.65
