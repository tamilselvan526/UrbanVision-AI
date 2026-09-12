"""
Automatic Number Plate Recognition (ANPR / ALPR) Module
=========================================================
Locates license plates on vehicle crops (cars, motorcycles, trucks, buses)
using morphological feature localization and extracts alphanumeric registration
numbers via Tesseract OCR (5.5.0).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from loguru import logger

# Set Tesseract executable path if standard Windows location exists
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if Path(TESSERACT_EXE).exists():
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

# Minimum & maximum length for valid vehicle plate strings
MIN_PLATE_LEN = 4
MAX_PLATE_LEN = 13


class PlateRecognizer:
    """
    Analyzes vehicle crops, isolates license plate regions,
    and runs high-accuracy OCR text recognition.
    """

    def __init__(self):
        self.tesseract_available = Path(TESSERACT_EXE).exists()
        if self.tesseract_available:
            logger.success("[PlateRecognizer] Connected to local Tesseract OCR engine.")
        else:
            logger.warning(f"[PlateRecognizer] Tesseract binary not found at {TESSERACT_EXE}. OCR will use fallback heuristics.")

    def extract_plate_candidates(self, vehicle_crop: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Locate license plate candidates using morphological gradient and contour aspect ratios.
        Returns list of (candidate_crop, (x, y, w, h)).
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return []

        vh, vw = vehicle_crop.shape[:2]
        if vh < 40 or vw < 60:
            return []

        # License plates are predominantly located in the lower 60% of the vehicle
        roi_y1 = int(vh * 0.35)
        roi = vehicle_crop[roi_y1:vh, 0:vw]
        rh, rw = roi.shape[:2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Contrast enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)

        # Morphological Black-Hat to reveal dark/light rectangular license plates
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(contrast, cv2.MORPH_BLACKHAT, rect_kernel)

        # Sobel horizontal gradient to find vertical character edges
        grad_x = cv2.Sobel(blackhat, cv2.CV_32F, dx=1, dy=0, ksize=3)
        grad_x = np.absolute(grad_x)
        min_v, max_v = np.min(grad_x), np.max(grad_x)
        if max_v > min_v:
            grad_x = (255 * (grad_x - min_v) / (max_v - min_v)).astype(np.uint8)
        else:
            grad_x = grad_x.astype(np.uint8)

        # Blur and close characters into a solid plate blob
        blurred = cv2.GaussianBlur(grad_x, (5, 5), 0)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, close_kernel)
        _, thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: List[Tuple[np.ndarray, Tuple[int, int, int, int]]] = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / max(float(h), 1.0)
            area = w * h

            # License plates have aspect ratio between 1.8 and 5.8, and reasonable area
            if 1.6 <= aspect_ratio <= 6.2 and 400 < area < (rw * rh * 0.35):
                pad_x = int(w * 0.05)
                pad_y = int(h * 0.08)
                px1 = max(0, x - pad_x)
                py1 = max(0, y - pad_y)
                px2 = min(rw, x + w + pad_x)
                py2 = min(rh, y + h + pad_y)

                cand = roi[py1:py2, px1:px2]
                if cand.size > 0:
                    candidates.append((cand, (px1, py1 + roi_y1, px2 - px1, py2 - py1)))

        # Also add default lower-center region as fallback candidate if no contour found
        if not candidates:
            def_w = int(vw * 0.40)
            def_h = int(vh * 0.18)
            def_x = int((vw - def_w) / 2)
            def_y = int(vh * 0.65)
            def_crop = vehicle_crop[def_y:min(vh, def_y + def_h), def_x:min(vw, def_x + def_w)]
            if def_crop.size > 0:
                candidates.append((def_crop, (def_x, def_y, def_w, def_h)))

        return candidates

    def clean_plate_text(self, raw_text: str) -> str:
        """Strip noise, newlines, and punctuation. Standardize to uppercase alphanumeric."""
        # Replace common OCR misreads (e.g. '|' to 'I', 'O' to '0' where appropriate)
        text = raw_text.upper()
        text = re.sub(r"[^A-Z0-9]", "", text)
        return text.strip()

    def recognize_plate(
        self,
        vehicle_crop: np.ndarray,
    ) -> Optional[Tuple[str, float, Tuple[int, int, int, int]]]:
        """
        Detect and read number plate from a vehicle crop.
        Returns: (plate_text, confidence, (x, y, w, h) bounding box relative to vehicle crop)
        """
        if not self.tesseract_available:
            return None

        candidates = self.extract_plate_candidates(vehicle_crop)
        best_plate = ""
        best_conf = 0.0
        best_bbox = (0, 0, 0, 0)

        for cand_img, bbox in candidates[:4]:
            if cand_img.shape[0] < 15 or cand_img.shape[1] < 30:
                continue

            # Scale up small plate crops for better OCR
            scale = max(1.0, 100.0 / cand_img.shape[0])
            scaled = cv2.resize(cand_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
            # Bilateral filter removes noise while keeping character edges sharp
            filtered = cv2.bilateralFilter(gray, 11, 17, 17)
            thresh = cv2.adaptiveThreshold(
                filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # Test both normal and inverted threshold
            for img_to_ocr in [filtered, thresh]:
                try:
                    # Config: PSM 7 (single line of text) with alphanumeric whitelist
                    data = pytesseract.image_to_data(
                        img_to_ocr,
                        config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                        output_type=pytesseract.Output.DICT,
                    )
                    words = []
                    confs = []
                    for i, w in enumerate(data.get("text", [])):
                        clean = self.clean_plate_text(w)
                        if clean:
                            words.append(clean)
                            try:
                                confs.append(float(data["conf"][i]))
                            except Exception:
                                confs.append(50.0)

                    joined = "".join(words)
                    avg_conf = float(np.mean(confs)) if confs else 0.0

                    if MIN_PLATE_LEN <= len(joined) <= MAX_PLATE_LEN and avg_conf > best_conf:
                        best_plate = joined
                        best_conf = avg_conf
                        best_bbox = bbox
                except Exception as e:
                    logger.debug(f"[PlateRecognizer] OCR error on candidate: {e}")

        if best_plate and best_conf > 30.0:
            logger.info(f"[PlateRecognizer] Detected Plate: '{best_plate}' (Conf: {best_conf:.1f}%)")
            return best_plate, best_conf, best_bbox

        return None


# Global singleton
plate_recognizer = PlateRecognizer()
