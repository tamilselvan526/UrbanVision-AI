"""
Launch the Smart Crime Detection Web Dashboard
===============================================
Usage:
    python run_server.py                    # Default: webcam, port 8000
    python run_server.py --port 8080        # Custom port
    python run_server.py --host 0.0.0.0    # LAN accessible

Then open: http://127.0.0.1:8000
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Smart Crime Detection Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    args = parser.parse_args()

    logger.info(f"Starting Smart Crime Detection Dashboard at http://{args.host}:{args.port}")
    logger.info("Open your browser and navigate to the URL above.")
    logger.info("Select a video source (webcam/file/RTSP) and click START.")

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
