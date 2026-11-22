"""
FastAPI Web Server — Smart Crime Detection Dashboard
====================================================
Provides:
  - GET  /           → Serve dashboard UI (index.html)
  - POST /start      → Start video stream (webcam/file/rtsp)
  - POST /stop       → Stop stream
  - GET  /status     → Stream status JSON
  - GET  /incidents  → All incident alerts JSON
  - WS   /ws/stream  → WebSocket: JPEG frames (base64) + live alert events
  - POST /upload     → Upload a video file for offline processing
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import shutil
import time
import threading
from pathlib import Path
from typing import List, Optional, Set

import cv2
import numpy as np
import uvicorn
from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form,
    BackgroundTasks, HTTPException
)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from engine.pipeline import CrimeDetectionPipeline, PipelineConfig
from engine.intelligence_db import intelligence_db, SNAPSHOTS_DIR
from engine.person_reid import person_reid
from engine.street_entity_profiler import street_entity_profiler
from app.streamer import VideoStreamer
from app.schemas import IncidentAlert, StreamStatus


# ── App Setup ──────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
UI_DIR = BASE_DIR / "ui"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Smart Crime Detection System",
    description="Real-time threat detection using 4 deep learning algorithms",
    version="1.0.0",
)

# Mount static files (CSS, JS)
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")

# Mount snapshots directory to serve target and vehicle thumbnails
if SNAPSHOTS_DIR.exists():
    app.mount("/snapshots", StaticFiles(directory=str(SNAPSHOTS_DIR)), name="snapshots")


# ── Global State ──────────────────────────────────────────────────────────

_pipeline: Optional[CrimeDetectionPipeline] = None
_streamer: Optional[VideoStreamer] = None
_active_clients: Set[WebSocket] = set()
_pipeline_lock = threading.Lock()
_state_lock = threading.Lock()
_latest_frame_jpg: Optional[bytes] = None
_latest_frame_id: int = -1
_current_source: str = "None"
_is_running: bool = False
_frame_count: int = 0
_current_fps: float = 0.0
_pipeline_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()
_alerts_history: list[dict] = []


# ── Pipeline Background Thread ─────────────────────────────────────────────

def _pipeline_loop():
    """Background thread: pull frames, run pipeline, update latest_frame."""
    global _latest_frame_jpg, _latest_frame_id, _frame_count, _current_fps, _is_running

    _stop_event.clear()
    logger.info("[Server] Pipeline loop started.")

    while not _stop_event.is_set():
        if _streamer is None or not _streamer.is_running():
            time.sleep(0.01)
            continue

        frame = _streamer.get_frame(timeout=0.1)
        if frame is None:
            continue

        if _pipeline is None:
            continue

        try:
            result = _pipeline.process_frame(frame)
            frame_id = result.frame_id
            fps = result.fps

            # JPEG encode annotated frame
            _, jpg = cv2.imencode(
                ".jpg", result.annotated_frame,
                [cv2.IMWRITE_JPEG_QUALITY, 75]
            )
            jpg_bytes = jpg.tobytes()

            # Process new alerts
            new_alerts = []
            if result.threat_events:
                high_events = [
                    e for e in result.threat_events if e.threat_score >= 45.0
                ]
                if high_events:
                    alerts = [IncidentAlert.from_threat_event(e) for e in high_events]
                    new_alerts = [a.model_dump() for a in alerts]

            with _state_lock:
                _latest_frame_jpg = jpg_bytes
                _latest_frame_id = frame_id
                _frame_count = frame_id
                _current_fps = fps
                if new_alerts:
                    _alerts_history.extend(new_alerts)
                    if len(_alerts_history) > 200:
                        del _alerts_history[:-200]

        except Exception as e:
            logger.error(f"[Pipeline] Error processing frame: {e}")

    logger.info("[Server] Pipeline loop stopped.")


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the main dashboard HTML."""
    index_path = UI_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>Dashboard UI not found. Run the server from project root.</h1>", status_code=404)
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.post("/start")
async def start_stream(
    source: str = "0",
    background_tasks: BackgroundTasks = None,
):
    """
    Start the detection pipeline.
    source: "0" for webcam, file path, or RTSP URL.
    """
    global _pipeline, _streamer, _is_running, _current_source, _pipeline_thread

    if _is_running:
        await stop_stream()

    # Resolve source
    if source.isdigit():
        resolved = int(source)
    elif source.startswith("rtsp://") or source.startswith("http://"):
        resolved = source
    else:
        # File path
        fpath = UPLOAD_DIR / source
        if fpath.exists():
            resolved = str(fpath)
        elif Path(source).exists():
            resolved = source
        else:
            raise HTTPException(400, f"Video source not found: {source}")

    _current_source = str(resolved)

    # Initialize pipeline
    with _pipeline_lock:
        config = PipelineConfig(device=None)  # auto GPU
        _pipeline = CrimeDetectionPipeline(config)

        _streamer = VideoStreamer(source=resolved)
        _streamer.start()

    _is_running = True
    _stop_event.clear()

    # Start pipeline in background thread
    _pipeline_thread = threading.Thread(target=_pipeline_loop, daemon=True)
    _pipeline_thread.start()

    logger.info(f"[Server] Stream started: {_current_source}")
    return {"status": "started", "source": _current_source}


@app.post("/stop")
async def stop_stream():
    """Stop the detection pipeline."""
    global _is_running, _latest_frame_jpg, _latest_frame_id

    _stop_event.set()
    _is_running = False

    if _streamer:
        _streamer.stop()

    with _state_lock:
        _latest_frame_jpg = None
        _latest_frame_id = -1

    logger.info("[Server] Stream stopped.")
    return {"status": "stopped"}


@app.get("/status", response_model=StreamStatus)
async def get_status():
    """Get current stream status."""
    incident_count = len(_pipeline.incident_log) if _pipeline else 0
    return StreamStatus(
        running=_is_running,
        source=_current_source,
        fps=round(_current_fps, 1),
        frame_count=_frame_count,
        incident_count=incident_count,
    )


@app.get("/incidents")
async def get_incidents():
    """Return all recorded incident alerts."""
    if not _pipeline:
        return {"incidents": []}
    alerts = [
        IncidentAlert.from_threat_event(e).model_dump()
        for e in _pipeline.incident_log[-100:]
    ]
    return {"incidents": alerts}


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file for processing."""
    safe_name = Path(file.filename).name
    dest = UPLOAD_DIR / safe_name
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    logger.info(f"[Server] Uploaded: {dest}")
    return {"filename": safe_name, "path": str(dest)}


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint streaming annotated JPEG frames as base64
    plus real-time alert events as JSON messages.
    """
    await websocket.accept()
    _active_clients.add(websocket)
    logger.info(f"[WS] Client connected. Total: {len(_active_clients)}")

    last_sent_frame_id = -1
    sent_waiting = False
    with _state_lock:
        last_alert_idx = len(_alerts_history)

    try:
        while True:
            await asyncio.sleep(1 / 30)  # ~30 fps push rate

            frame_bytes = None
            frame_id = -1
            fps = 0.0
            pending_alerts = []

            with _state_lock:
                total_alerts = len(_alerts_history)
                if total_alerts > last_alert_idx:
                    pending_alerts = _alerts_history[last_alert_idx:total_alerts]
                    last_alert_idx = total_alerts

                if _latest_frame_id != last_sent_frame_id and _latest_frame_jpg is not None:
                    frame_bytes = _latest_frame_jpg
                    frame_id = _latest_frame_id
                    fps = _current_fps

            # Send alerts first
            if pending_alerts:
                msg = json.dumps({
                    "type": "alerts",
                    "alerts": pending_alerts,
                })
                await websocket.send_text(msg)

            # Send frame if updated
            if frame_bytes is not None:
                last_sent_frame_id = frame_id
                sent_waiting = False
                b64 = base64.b64encode(frame_bytes).decode("utf-8")
                msg = json.dumps({
                    "type": "frame",
                    "data": b64,
                    "fps": round(fps, 1),
                    "frame_id": frame_id,
                })
                await websocket.send_text(msg)
            elif last_sent_frame_id == -1 and not sent_waiting:
                await websocket.send_text(json.dumps({"type": "waiting"}))
                sent_waiting = True

    except WebSocketDisconnect:
        logger.info("[WS] Client disconnected.")
    except Exception as e:
        logger.error(f"[WS] Error: {e}")
    finally:
        _active_clients.discard(websocket)


@app.post("/demo_action")
async def toggle_demo_action(enabled: Optional[bool] = None):
    """Toggle or set demo safe action (hands raised / green frame)."""
    global _pipeline
    with _pipeline_lock:
        if not _pipeline:
            config = PipelineConfig(device=None)
            _pipeline = CrimeDetectionPipeline(config)
        current = _pipeline.threat_analyzer.toggle_demo_green_action(enabled)

    return {
        "status": "ok",
        "demo_green_action": current,
        "action_name": "HANDS RAISED / WAVING (SAFE ACTION)",
        "frame_indicator": "GREEN FRAME ACTIVE" if current else "INACTIVE",
    }


# ── Watchlist & Person Re-ID Endpoints ─────────────────────────────────────

@app.post("/api/watchlist/enroll")
async def enroll_watchlist_person(
    name: str = Form(...),
    notes: str = Form(""),
    file: UploadFile = File(...),
):
    """Enroll a new target person with reference photo for visual Re-ID matching."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(400, "Invalid image format")

        pid, photo_fn = person_reid.enroll_person(name=name, image_bgr=img_bgr, notes=notes)
        return {
            "status": "success",
            "id": pid,
            "name": name,
            "photo_url": f"/snapshots/persons/{photo_fn}",
        }
    except Exception as e:
        logger.error(f"[API] Error enrolling person: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/watchlist/list")
async def list_watchlist():
    """List all enrolled persons of interest."""
    persons = intelligence_db.get_watchlist()
    for p in persons:
        p["photo_url"] = f"/snapshots/persons/{p['photo_filename']}" if p.get("photo_filename") else ""
    return {"watchlist": persons}


@app.delete("/api/watchlist/{person_id}")
async def delete_watchlist_person(person_id: int):
    """Remove a person from the watchlist."""
    success = intelligence_db.delete_watchlist_person(person_id)
    person_reid.reload_watchlist()
    return {"status": "deleted" if success else "not_found"}


@app.get("/api/sightings/person/{person_name}")
async def get_person_sightings(person_name: str):
    """Retrieve all chronological sightings and movement history for a person."""
    sightings = intelligence_db.get_sightings_for_person(person_name)
    for s in sightings:
        s["snapshot_url"] = f"/snapshots/persons/{s['snapshot_filename']}" if s.get("snapshot_filename") else ""
    return {"person": person_name, "sightings": sightings}


@app.get("/api/sightings/recent")
async def get_recent_sightings():
    """Retrieve recent watchlist sightings."""
    sightings = intelligence_db.get_recent_sightings(50)
    for s in sightings:
        s["snapshot_url"] = f"/snapshots/persons/{s['snapshot_filename']}" if s.get("snapshot_filename") else ""
    return {"sightings": sightings}


# ── Vehicle & License Plate (ANPR) Endpoints ───────────────────────────────

@app.get("/api/vehicles/search")
async def search_vehicles(plate: str = ""):
    """Search registered vehicles by full or partial license plate number."""
    results = intelligence_db.search_vehicles(query=plate)
    for v in results:
        v["snapshot_url"] = f"/snapshots/vehicles/{v['snapshot_filename']}" if v.get("snapshot_filename") else ""
    return {"query": plate, "count": len(results), "vehicles": results}


@app.get("/api/vehicles/recent")
async def get_recent_vehicles():
    """Retrieve recent vehicle passes with plate numbers and movement direction."""
    vehicles = intelligence_db.get_recent_vehicles(50)
    for v in vehicles:
        v["snapshot_url"] = f"/snapshots/vehicles/{v['snapshot_filename']}" if v.get("snapshot_filename") else ""
    return {"vehicles": vehicles}


# ── Street Regulars & Routine Profiling Endpoints ─────────────────────────

@app.get("/api/regulars/list")
async def get_street_regulars():
    """Retrieve all discovered street regulars with detected roles and statistics."""
    regulars = intelligence_db.get_all_regulars(100)
    for r in regulars:
        r["snapshot_url"] = f"/snapshots/regulars/{r['snapshot_filename']}" if r.get("snapshot_filename") else ""
    return {"regulars": regulars}


@app.post("/api/regulars/{regular_id}/rename")
async def rename_street_regular(regular_id: int, custom_name: str = Form(...)):
    """Assign or edit a custom identifier name for a street regular (e.g. 'Ramesh Chaiwala')."""
    success = intelligence_db.update_regular_custom_name(regular_id, custom_name)
    if success:
        street_entity_profiler.reload_regulars_from_db()
        return {"status": "success", "regular_id": regular_id, "custom_name": custom_name}
    raise HTTPException(404, "Street regular not found")


@app.delete("/api/regulars/{regular_id}")
async def delete_street_regular(regular_id: int):
    """Remove a regular profile from database."""
    success = intelligence_db.delete_regular(regular_id)
    if success:
        street_entity_profiler.reload_regulars_from_db()
        return {"status": "deleted"}
    raise HTTPException(404, "Street regular not found")


# ── Entry point (when called directly) ────────────────────────────────────

def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    uvicorn.run("app.main:app", host=host, port=port, reload=reload)
