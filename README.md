# Smart City AI: Intelligent Surveillance & Crime Detection System

A high-performance, GPU-accelerated tactical surveillance and forensic intelligence system engineered with **deep learning algorithms**, **biomechanical action recognition**, **Person Re-Identification (Re-ID)**, and **Automatic Number Plate Recognition (ANPR)**.

---

## 🚀 Core Capabilities

### 1. Multi-Modal Weapon & Object Detection (Algorithm 1)
- **Primary Detection**: YOLOv8 deep neural network detecting persons and firearms (handguns, rifles, shotguns, blades).
- **Improvised Weapons**: Contextual detection of improvised weapons held in hand (bottles, baseball bats, scissors, knives, rods, umbrellas).
- **Vehicle Detection**: Real-time bounding and classification of cars, motorcycles, buses, and trucks.

### 2. Multi-Object Movement Tracking (Algorithm 2)
- **ByteTrack Integration**: High-accuracy multi-object tracking utilizing Kalman filtering and Hungarian association.
- **Kinematics & Trajectories**: Continuous path rendering, EMA velocity vectors, speed calculations, and sudden motion/lunge detection.

### 3. Biomechanical Pose & Violent Action Recognition (Algorithm 3)
- **17-Keypoint YOLO-Pose**: Real-time joint extraction (head, shoulders, elbows, wrists, hips, knees, ankles).
- **Punch / Strike Kinematics**: Wrist velocity tracking and sudden acceleration detection.
- **Stabbing Thrust Analysis**: Rapid elbow extension and forward trajectory vector analysis.
- **Aggressive Stare & Confrontation**: Face orientation vectors and opposing gaze alignment.
- **Combat Altercation Engine**: Detects mutual combat stances, boxing/sparring guards, and active street altercations (scoring up to 99% CRITICAL THREAT).

### 4. Spatio-Temporal Threat Reasoning (Algorithm 4)
- Fuses detections, tracking kinematics, and pose mechanics into a composite **Crime Threat Score (0–100%)**.
- Threat Levels: `CLEAR`, `SUSPICIOUS`, `THREAT DETECTED`, `ACTIVE THREAT`, `CRITICAL THREAT`.
- Emits contextual action tags: `DUAL_COMBAT_STANCE`, `MUTUAL_COMBAT`, `STRIKES_EXCHANGED`, `AIMING_AT_PERSON`, `WEAPON_IN_HAND`, `VICTIM_DOWN_NEARBY`.

### 5. Person Memory & Watchlist (Person Re-ID)
- **Subject Enrollment**: Upload suspect photos with alias and case notes.
- **Appearance Descriptors**: 392-dimensional multi-region spatial color-histogram and gradient structure vector.
- **Live Identification**: Real-time cosine similarity matching across camera feeds (`[TARGET: <NAME> <CONF>%]`).
- **Forensic Sighting Logs**: Automatically records timestamps, camera IDs, exact coordinates \((X, Y)\), and photographic snapshots in an SQLite database.

### 6. Automatic Number Plate Recognition (ANPR / ALPR)
- **Plate Localization**: Morphological Top-Hat/Black-Hat transforms, Sobel vertical gradients, and contour aspect ratio filtering.
- **Tesseract OCR Integration**: High-accuracy alphanumeric plate text extraction.
- **Movement Registry & Trajectory Search**: Search any plate number (e.g. `DL08`, `MH02`) to view where the vehicle went, timestamps, speed, direction, and evidence crops.

### 7. Retro Terminal Web Dashboard
- Amber phosphor CRT terminal aesthetic (no neon, aged blueprint layout, scanlines, and tactile controls).
- Navigation tabs: `[F1] Live Feed`, `[F2] Person Watchlist`, `[F3] Plate & Vehicle Search`.

---

## 🛠️ Architecture

```
lawfix/
├── engine/
│   ├── detector.py          # YOLOv8 Person, Weapon & Vehicle Detector
│   ├── tracker.py           # ByteTrack Multi-Object Motion Tracker
│   ├── pose_estimator.py    # YOLO-Pose 17-Keypoint & Strike Velocity Engine
│   ├── threat_analyzer.py   # Spatio-Temporal Combat Reasoning Engine
│   ├── person_reid.py       # Watchlist Re-ID & Visual Appearance Descriptor
│   ├── plate_recognizer.py  # License Plate Localization & Tesseract OCR
│   ├── vehicle_tracker.py   # Vehicle Movement Kinematics & Registry
│   ├── intelligence_db.py   # SQLite Intelligence & Sightings Database
│   └── pipeline.py          # Master Orchestrator Chaining All Algorithms
├── app/
│   ├── main.py              # FastAPI Web & WebSocket Server
│   ├── streamer.py          # Thread-Safe Video Frame Producer
│   └── schemas.py           # Pydantic Schemas & Alert Serializers
├── ui/
│   ├── index.html           # Retro CRT Surveillance Terminal Interface
│   ├── style.css            # Amber Phosphor CRT Stylesheet
│   └── app.js               # WebSocket Streaming, Canvas Rendering & Tabs
├── tests/
│   ├── test_pipeline.py     # Main 4-Algorithm Pipeline Verification Tests
│   └── test_reid_plate.py   # Person Re-ID and ANPR Unit Tests
├── run_server.py            # Web Dashboard Launcher
└── run_cli.py               # CLI Video / Webcam Processor
```

---

## ⚡ Quick Start

### 1. Prerequisites
- Python 3.12+
- NVIDIA GPU with CUDA support (e.g., RTX 4050 / 3060 / 4070)
- Tesseract OCR installed (for Windows: `C:\Program Files\Tesseract-OCR\tesseract.exe`)

### 2. Installation

```powershell
# Clone the repository
git clone https://github.com/saaqibA21/smartcityai.git
cd smartcityai

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install PyTorch with CUDA support (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install system dependencies
pip install -r requirements.txt
```

### 3. Launch Web Dashboard

```powershell
python run_server.py
```
Open **`http://127.0.0.1:8000`** in your browser.

### 4. Run CLI Offline Video Processor

```powershell
# Process video and save annotated output
python run_cli.py --source "path/to/video.mp4" --save --output "output_detected.mp4"

# Run on live webcam
python run_cli.py
```

### 5. Run Test Suites

```powershell
python tests/test_pipeline.py
python tests/test_reid_plate.py
```

---

## 📄 License
MIT License
