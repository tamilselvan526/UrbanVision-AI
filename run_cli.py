"""
CLI Video Processor — Smart Crime Detection
===========================================
Process a video file (or webcam) directly in a window
without needing the browser dashboard.

Usage:
    python run_cli.py                               # Webcam
    python run_cli.py --source video.mp4            # Video file
    python run_cli.py --source rtsp://...           # IP camera
    python run_cli.py --source video.mp4 --save     # Save annotated output

Keyboard controls:
    Q / ESC  : Quit
    P        : Pause/Resume
    S        : Save current frame snapshot
    B        : Toggle bounding boxes
    T        : Toggle trails
    K        : Toggle skeleton
    H        : Toggle threat HUD
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
from loguru import logger
from engine.pipeline import CrimeDetectionPipeline, PipelineConfig


def main():
    parser = argparse.ArgumentParser(description="Smart Crime Detection CLI")
    parser.add_argument("--source", default="0", help="Video source: 0 (webcam), path, or RTSP URL")
    parser.add_argument("--save", action="store_true", help="Save annotated output video")
    parser.add_argument("--output", default="output_detected.mp4", help="Output video filename")
    parser.add_argument("--no-boxes", action="store_true", help="Disable bounding boxes")
    parser.add_argument("--no-trails", action="store_true", help="Disable motion trails")
    parser.add_argument("--no-skeleton", action="store_true", help="Disable pose skeleton")
    parser.add_argument("--device", default=None, help="Force device: cuda or cpu")
    args = parser.parse_args()

    # Resolve source
    source = args.source
    if source.isdigit():
        source = int(source)
    elif not source.startswith("rtsp://") and not source.startswith("http://"):
        p = Path(source)
        if not p.exists():
            logger.error(f"Video file not found: {source}")
            sys.exit(1)
        source = str(p)

    # Pipeline config
    config = PipelineConfig(
        show_boxes=not args.no_boxes,
        show_trails=not args.no_trails,
        show_skeleton=not args.no_skeleton,
        show_threat_hud=True,
        show_kinematics=True,
        device=args.device,
    )

    pipeline = CrimeDetectionPipeline(config)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open source: {source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
        logger.info(f"Saving to: {args.output}")

    logger.info("=== Smart Crime Detection CLI ===")
    logger.info(f"Source: {source} | {w}x{h} @ {fps:.0f}fps")
    logger.info("Controls: Q=Quit, P=Pause, S=Snapshot, B/T/K/H=Toggle overlays")

    paused = False
    frame_count = 0
    snapshot_count = 0

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(source, str):  # Loop video file
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                result = pipeline.process_frame(frame)
                out_frame = result.annotated_frame
                frame_count += 1

                if writer:
                    writer.write(out_frame)

                cv2.imshow("Smart Crime Detection System", out_frame)

                # Print high-threat events to console
                for event in result.threat_events:
                    if event.threat_score >= 45.0:
                        logger.warning(
                            f"[ALERT] Person #{event.person_track_id} | "
                            f"{event.threat_label} | Score: {event.threat_score:.0f}% | "
                            f"Tags: {event.action_tags}"
                        )

            # Key handling
            key = cv2.waitKey(1 if not paused else 100) & 0xFF
            if key in (ord("q"), 27):   # Q or ESC
                break
            elif key == ord("p"):
                paused = not paused
                logger.info("PAUSED" if paused else "RESUMED")
            elif key == ord("s"):
                fname = f"snapshot_{snapshot_count:04d}.jpg"
                cv2.imwrite(fname, result.annotated_frame)
                logger.info(f"Snapshot saved: {fname}")
                snapshot_count += 1
            elif key == ord("b"):
                config.show_boxes = not config.show_boxes
            elif key == ord("t"):
                config.show_trails = not config.show_trails
            elif key == ord("k"):
                config.show_skeleton = not config.show_skeleton
            elif key == ord("h"):
                config.show_threat_hud = not config.show_threat_hud

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        logger.info(f"=== Session Complete ===")
        logger.info(f"Frames processed: {frame_count}")
        logger.info(f"Total incidents: {len(pipeline.incident_log)}")
        if pipeline.incident_log:
            logger.warning("=== INCIDENT SUMMARY ===")
            for inc in pipeline.incident_log:
                logger.warning(
                    f"  [{time.strftime('%H:%M:%S', time.localtime(inc.timestamp))}] "
                    f"Person #{inc.person_track_id} | {inc.threat_label} | "
                    f"Score: {inc.threat_score:.0f}% | Weapon: {inc.weapon_class}"
                )


if __name__ == "__main__":
    main()
