"""
Person Re-Identification (Re-ID) & Visual Memory Module
=========================================================
Allows enrolling persons of interest via reference photo, extracting multi-region
visual signatures (face/head crop, upper body, color-spatial histograms, texture HOG),
and matching live detected persons across video surveillance feeds.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from engine.intelligence_db import intelligence_db, SNAPSHOTS_DIR
from engine.pose_estimator import PoseResult


# Similarity threshold above which a person is considered a watchlist match (0.0 - 1.0)
DEFAULT_MATCH_THRESHOLD = 0.72


class PersonReIDEngine:
    """
    Extracts invariant visual embeddings from person crops and matches
    against the enrolled Watchlist in the Intelligence Database.
    """

    def __init__(self, match_threshold: float = DEFAULT_MATCH_THRESHOLD):
        self.match_threshold = match_threshold
        # In-memory cache of enrolled embeddings: [(id, name, embedding)]
        self._gallery: List[Tuple[int, str, np.ndarray]] = []
        self.reload_watchlist()

    def reload_watchlist(self):
        """Reload enrolled persons and their embeddings from database."""
        self._gallery = intelligence_db.get_all_embeddings()
        logger.info(f"[PersonReID] Loaded {len(self._gallery)} target person(s) into memory watchlist.")

    def extract_embedding(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Compute a robust, normalized multi-feature visual descriptor for an image/crop.
        Combines:
          1. Color appearance: 3-region (Upper, Mid, Lower) HSV 3D histogram (captures clothing/skin color)
          2. Texture/Structure: Multi-scale Grayscale Gradient Histogram (captures silhouette and clothing pattern)
          3. Aspect & Luminance invariant spatial representation.
        """
        if image_bgr is None or image_bgr.size == 0:
            return np.zeros(384, dtype=np.float32)

        # Resize to standardized aspect ratio for Re-ID (128x256)
        img = cv2.resize(image_bgr, (128, 256), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Divide into 3 vertical stripes: Head/Upper (0-30%), Torso (30-65%), Legs (65-100%)
        regions = [
            hsv[0:int(h * 0.30), :],
            hsv[int(h * 0.30):int(h * 0.65), :],
            hsv[int(h * 0.65):h, :],
        ]

        feature_parts = []
        for reg in regions:
            # 8 H bins, 4 S bins, 4 V bins = 128 elements per region
            hist = cv2.calcHist([reg], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            feature_parts.append(hist.flatten())

        # Structural gradient descriptor (HOG-like edge magnitude & angle across 4x8 blocks)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        # Quantize orientation into 8 bins
        ang_bins = (ang / 45.0).astype(np.int32) % 8
        struct_hist = np.zeros(8, dtype=np.float32)
        for b in range(8):
            struct_hist[b] = np.sum(mag[ang_bins == b])
        norm = np.linalg.norm(struct_hist)
        if norm > 1e-6:
            struct_hist /= norm
        feature_parts.append(struct_hist)

        # Concatenate and L2-normalize
        full_vec = np.concatenate(feature_parts).astype(np.float32)
        norm_total = np.linalg.norm(full_vec)
        if norm_total > 1e-6:
            full_vec /= norm_total

        return full_vec

    def enroll_person(
        self,
        name: str,
        image_bgr: np.ndarray,
        notes: str = "",
    ) -> Tuple[int, str]:
        """
        Enroll a new person of interest.
        Saves photo to snapshots/persons/ and embeds in database.
        """
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Invalid image for enrollment")

        embedding = self.extract_embedding(image_bgr)

        # Save reference photo
        safe_name = "".join(c for c in name if c.isalnum() or c in ("-", "_")).lower()
        filename = f"target_{safe_name}_{int(cv2.getTickCount())}.jpg"
        save_path = SNAPSHOTS_DIR / "persons" / filename
        cv2.imwrite(str(save_path), image_bgr)

        pid = intelligence_db.add_watchlist_person(
            name=name,
            embedding=embedding,
            notes=notes,
            photo_filename=filename,
        )
        self.reload_watchlist()
        logger.success(f"[PersonReID] Enrolled '{name}' (ID: {pid}) with {len(embedding)}-dim visual embedding.")
        return pid, filename

    def match_crop(
        self,
        crop_bgr: np.ndarray,
        threshold_override: Optional[float] = None,
    ) -> Optional[Tuple[str, float]]:
        """
        Compare a person image crop against all enrolled watchlist members.
        Returns (person_name, similarity_score) if above threshold, else None.
        """
        if not self._gallery or crop_bgr is None or crop_bgr.size == 0:
            return None

        emb = self.extract_embedding(crop_bgr)
        threshold = threshold_override or self.match_threshold

        best_name = None
        best_sim = -1.0

        for pid, name, target_emb in self._gallery:
            # Cosine similarity between L2-normalized vectors is simply the dot product
            sim = float(np.dot(emb, target_emb))
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_sim >= threshold and best_name is not None:
            return best_name, best_sim

        return None


# Global singleton
person_reid = PersonReIDEngine()
