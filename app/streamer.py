"""
Video Frame Streamer
====================
Handles frame production from webcam, video file, or RTSP stream.
Runs in a background thread feeding frames to the pipeline.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np
from loguru import logger


class VideoStreamer:
    """
    Thread-safe video frame producer.
    
    Supports:
      - Webcam (source=0 or integer)
      - Video file (source="path/to/video.mp4")
      - RTSP IP camera (source="rtsp://...")
    """

    def __init__(self, source=0, max_queue_size: int = 4):
        self.source = source
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self.fps: float = 0.0
        self.width: int = 0
        self.height: int = 0
        self.running: bool = False
        self.frame_count: int = 0

    def start(self):
        """Open capture device and start background frame-grab thread."""
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"[Streamer] Started: {self.source} ({self.width}x{self.height} @ {self.fps:.0f}fps)")

    def stop(self):
        """Stop the frame-grab thread and release resources."""
        self._stop_event.set()
        self.running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()
        logger.info("[Streamer] Stopped.")

    def _capture_loop(self):
        """Background thread: continuously grab frames and push to queue."""
        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret:
                # For video files: loop back to start
                if isinstance(self.source, str) and self.source != "0":
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    logger.warning("[Streamer] No frame received. Stopping.")
                    break

            self.frame_count += 1
            # Drop oldest frame if queue full (prioritize freshness)
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                pass

        self.running = False

    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get the next frame (blocks until available or timeout)."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def is_running(self) -> bool:
        return self.running and not self._stop_event.is_set()
