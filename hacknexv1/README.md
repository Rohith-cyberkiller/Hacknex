# Cross-Modal Deepfake Forensics

**Module:** Member 3 — Cross-Modal Synchronization and Multimodal Fusion Lead  
**Target:** Talking-head deepfake detection via audiovisual temporal synchronization, mouth-motion acoustics correlation, and calibrated quality-aware multimodal decision fusion.

---

## 📌 Technology Stack Matrix

| Layer | Component | Description |
|---|---|---|
| **Language** | Python 3.11 | Modern typed Python runtime |
| **Video/Audio Extraction** | FFmpeg, OpenCV | Container demuxing, video decoding, 16kHz PCM audio extraction |
| **Face & Mouth Tracking** | MediaPipe Face Mesh | 468-point 3D landmark tracking, sub-pixel lip geometry & OpenCV fallback |
| **Audio Features** | librosa, SoundFile, SciPy | Frame-synchronized RMS energy, spectral envelopes, dynamic range |
| **ML / Fusion** | PyTorch, NumPy, scikit-learn | Quality-aware attention fusion, temperature scaling calibration |
| **Lip-Sync** | Heuristic Correlation / SyncNet | Correlation first; drop-in Wav2Lip / SyncNet discriminator adapter |
| **API** | FastAPI, Uvicorn, Pydantic | Typed asynchronous endpoints (`/analyze/video`, `/health`, `/report/{id}`) |
| **Dashboard** | Streamlit, Plotly | Interactive forensic UI, dark timeline charts, and report exports |
---

## 📚 Implementation Reference Repositories

This module incorporates architectural patterns and validation protocols inspired by the following state-of-the-art repositories:

1. **[joonson/syncnet_python](https://github.com/joonson/syncnet_python)**:
   - Audiovisual synchronization architecture (Chung & Zisserman, ACCV 2016).
   - Dual acoustic (20-frame mel-spectrogram) and visual (5-frame cropped mouth) Euclidean embedding projections.
   - Cross-shift offset search ($[-15, +15]$ frames) implemented in [`core/models/syncnet_architecture.py`](file:///core/models/syncnet_architecture.py) and [`core/models/syncnet.py`](file:///core/models/syncnet.py).
2. **[Rudrabha/Wav2Lip](https://github.com/Rudrabha/Wav2Lip)**:
   - Lip-sync expert discriminator loss and cosine similarity metric for talking-head articulation alignment.
3. **[SCLBD/DeepfakeBench](https://github.com/SCLBD/DeepfakeBench)**:
   - Unified deepfake evaluation protocol and temperature scaling probability calibration.
   - NLL optimization, Expected Calibration Error (ECE), and Brier score evaluation implemented in [`core/calibration/trainer.py`](file:///core/calibration/trainer.py).
4. **[ondyari/FaceForensics](https://github.com/ondyari/FaceForensics)**:
   - Benchmark verification harness evaluating pristine (`original_sequences/youtube`) vs manipulated subsets (`Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`).
   - Automated evaluation harness in [`core/datasets/faceforensics.py`](file:///core/datasets/faceforensics.py).

---

## 📌 Architecture Overview

```
                      ┌───────────────────────────────────────┐
                      │    Uploaded Talking-Head Video        │
                      │       (MP4, MOV, AVI, WebM)           │
                      └──────────────────┬────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
    ┌───────────────────────────┐                 ┌───────────────────────────┐
    │     Video Subsystem       │                 │      Audio Subsystem      │
    │  - Frame Subsampling      │                 │  - FFmpeg Audio Extract   │
    │  - Primary Face Tracking  │                 │  - 16 kHz Mono PCM Audio  │
    │  - Mouth Region Cropping  │                 │  - Speech Acoustic Energy │
    │  - Optical Motion Energy  │                 │  - Voice Activity (VAD)   │
    │  - Visual Quality (Q_v)   │                 │  - Audio Quality (Q_a)    │
    └────────────┬──────────────┘                 └─────────────┬─────────────┘
                 │                                              │
                 └───────────────────────┬──────────────────────┘
                                         ▼
                     ┌──────────────────────────────────────┐
                     │          LipSyncModel Adapter        │
                     │  - HeuristicLipSyncModel (Default)   │
                     │  - PretrainedSyncNetAdapter (Ext.)   │
                     │  ----------------------------------- │
                     │  * Cross-Correlation Peak & Offset   │
                     │  * Drift Timeline & Suspicious Segs  │
                     │  * Lip-Sync Score (0 - 100)          │
                     └───────────────────┬──────────────────┘
                                         │
┌────────────────────────┐               │                ┌────────────────────────┐
│ Member 1: Visual Lead  │               │                │ Member 2: Audio Lead   │
│ - Visual Fake Prob     ├───────────────┼───────────────►│ - Audio Fake Prob      │
│ - Visual Quality (Q_v) │               │                │ - Audio Quality (Q_a)  │
└────────────────────────┘               │                └────────────────────────┘
                                         ▼
                     ┌──────────────────────────────────────┐
                     │    Quality-Aware Multimodal Fusion   │
                     │  - Dynamic Quality Attenuation       │
                     │  - Joint AV Cross-Modal Attention    │
                     │  - Raw Fused Probability             │
                     └───────────────────┬──────────────────┘
                                         ▼
                     ┌──────────────────────────────────────┐
                     │   Temperature Scaling Calibration    │
                     │   T-scaled Logits -> Calibrated Fake │
                     │   Authenticity Score = 100 - Fake    │
                     └───────────────────┬──────────────────┘
                                         ▼
                     ┌──────────────────────────────────────┐
                     │     Calibrated Forensic Report       │
                     │  - Classification: REAL/SUSP/FAKE   │
                     │  - Confidence: low/med/high          │
                     │  - Evidence Strings & Drift Windows  │
                     │  - Modal Contribution Weights        │
                     └──────────────────────────────────────┘
```

---

## 🚀 Key Features

1. **Audiovisual Synchronization Analysis**:
   - Computes dense optical motion on cropped mouth patches.
   - Extracts frame-aligned acoustic speech envelope using RMS energy and vocal tract smoothing.
   - Computes cross-correlation lag to estimate audio-video offset in milliseconds.
   - Isolates temporal drift segments where speech occurs with stationary lips, or lips move during silence.
2. **Quality-Aware Attention Fusion**:
   - Modulates modality contributions based on respective stream fidelity ($Q_v$, $Q_a$).
   - When audio quality is noisy or clipped, audio detector influence is attenuated; when video is blurry or poorly tracked, visual detector influence is reduced.
   - Evaluates joint cross-modal reliability.
3. **Probability Calibration**:
   - Mitigates detector overconfidence via temperature scaling on logits.
   - Safe fallback default ($T=1.15$) with support for loading custom-calibrated parameters from JSON artifacts.
4. **Honest Forensic Representation**:
   - Adheres to forensic integrity: never claims heuristic outputs are neural representations.
   - Clearly flags `"analysis_mode": "heuristic"` when the heuristic baseline is active.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.11
- FFmpeg (bundled automatically via `imageio-ffmpeg` or available on system PATH)

### 1. Clone & Create Virtual Environment

```bash
git clone <repo-url>
cd hacknexv1

# Create Python 3.11 virtual environment
py -3.11 -m venv .venv

# Activate environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate environment (Linux / macOS)
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy `.env.example` to `.env` if custom overrides are needed:
```bash
cp .env.example .env
```

---

## 🖥️ Running the Services

### Start the FastAPI Backend

Run the backend on port 8000:
```bash
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Health check verification:
```bash
curl http://localhost:8000/health
```

### Start the Streamlit Forensic Dashboard

Open a separate terminal and launch Streamlit on port 8501:
```bash
.venv\Scripts\python.exe -m streamlit run frontend/app.py --server.port 8501
```

Access the dashboard in your browser at: **`http://localhost:8501`**

---

## 🔌 Team Integration Guide (Members 1 & 2)

Member 3's backend accepts optional scores from Member 1 (Visual Detector) and Member 2 (Audio Detector) to produce the unified calibrated authenticity report.

### REST API Integration

#### Endpoint: `POST /analyze/video`

**Headers:** `Content-Type: multipart/form-data`

| Parameter | Type | Required | Description |
|---|---|---|---|
| `video` | File | Yes | Video file (`.mp4`, `.mov`, `.avi`, `.webm`) |
| `visual_fake_probability` | float | No | Member 1 fake probability (0.0 to 100.0 or 0.0 to 1.0) |
| `visual_quality` | float | No | Member 1 stream quality metric (0.0 to 1.0) |
| `audio_fake_probability` | float | No | Member 2 fake probability (0.0 to 100.0 or 0.0 to 1.0) |
| `audio_quality` | float | No | Member 2 stream quality metric (0.0 to 1.0) |
| `sample_fps` | int | No | Sampling frame rate (default: 20) |
| `model_type` | string | No | `"heuristic"` (default) or `"pretrained"` |

#### Python Client Example for Member 1 & 2

```python
import requests

url = "http://localhost:8000/analyze/video"

files = {
    "video": open("interview_clip.mp4", "rb")
}

data = {
    # Member 1 Visual Detector outputs:
    "visual_fake_probability": 38.5,
    "visual_quality": 0.88,
    
    # Member 2 Audio Detector outputs:
    "audio_fake_probability": 22.0,
    "audio_quality": 0.92,
    
    # Settings:
    "sample_fps": 20,
    "model_type": "heuristic"
}

response = requests.post(url, files=files, data=data)
report = response.json()

print(f"Verdict: {report['classification']}")
print(f"Authenticity Score: {report['authenticity_score']}%")
print(f"Timing Offset: {report['lip_sync']['estimated_offset_ms']} ms")
```

#### cURL Example

```bash
curl -X POST "http://localhost:8000/analyze/video" \
  -F "video=@interview_clip.mp4" \
  -F "visual_fake_probability=40.0" \
  -F "visual_quality=0.85" \
  -F "audio_fake_probability=30.0" \
  -F "audio_quality=0.90"
```

---

## 📄 Output Forensic Report Specification

```json
{
  "analysis_id": "8482436f-b25b-426b-a25e-04f7c16474fe",
  "classification": "SUSPICIOUS",
  "authenticity_score": 62.4,
  "final_fake_probability": 37.6,
  "confidence": "medium",
  "lip_sync": {
    "score": 61.0,
    "estimated_offset_ms": 180,
    "correlation": 0.42,
    "analysis_mode": "heuristic",
    "suspicious_segments": [
      {
        "start_sec": 4.2,
        "end_sec": 7.8,
        "severity": "high",
        "reason": "Mouth motion does not align with speech energy"
      }
    ]
  },
  "fusion": {
    "visual_contribution": 0.35,
    "audio_contribution": 0.25,
    "lip_sync_contribution": 0.40,
    "visual_quality": 0.9,
    "audio_quality": 0.8
  },
  "evidence": [
    "Detected an estimated 180 ms audio-video offset.",
    "Lip movement was weakly correlated with spoken-audio energy between 4.2 and 7.8 seconds."
  ],
  "limitations": [
    "This score is probabilistic forensic evidence, not proof of manipulation.",
    "Cross-modal synchronisation can be influenced by broadcast compression, encoding lag, or non-frontal head poses.",
    "Quality ratings applied: visual quality 0.90, audio quality 0.80.",
    "Cross-modal synchrony evaluated in Heuristic analysis mode (optical flow & acoustic envelopes)."
  ]
}
```

---

## 🧪 Running Unit & Integration Tests

The project includes unit and integration tests covering fusion, calibration, timeline detection, and end-to-end pipeline execution with synthetic signals:

```bash
.venv\Scripts\python.exe -m pytest -v
```

All 21 unit tests run and pass without requiring external pretrained models or manual video downloads.

---

## 📦 Benchmark Datasets via Kaggle CLI

The repository includes a dedicated dataset module [`core/datasets/downloader.py`](file:///core/datasets/downloader.py) supporting the Kaggle CLI:

### 1. Authenticate Kaggle CLI
Place your `kaggle.json` API token in `~/.kaggle/kaggle.json` or set your environment variables:
```bash
export KAGGLE_USERNAME="your-username"
export KAGGLE_KEY="your-api-key"
```

### 2. Download Datasets
```bash
# Download FakeAVCeleb dataset
.venv\Scripts\python.exe -m core.datasets.downloader --dataset fakeavceleb --output-dir data/datasets

# Download Deepfake Detection Challenge (DFDC) samples
.venv\Scripts\python.exe -m core.datasets.downloader --dataset dfdc --output-dir data/datasets

# Download VoxCeleb subset
.venv\Scripts\python.exe -m core.datasets.downloader --dataset voxceleb --output-dir data/datasets
---

## 📈 Probability Calibration Training (DeepfakeBench Protocol)

Train and fit the temperature scaling calibrator on prediction scores using Negative Log-Likelihood (NLL) optimization:

```bash
# Run calibration training and update calibrator.json
.venv\Scripts\python.exe -m core.calibration.trainer --save-path core/calibration/artifacts/calibrator.json
```

---

## 🔬 FaceForensics++ Verification Protocol

To verify the cross-modal forensic pipeline against FaceForensics++ pristine and manipulated subsets:

```bash
# Display official FaceForensics++ academic download guidelines
.venv\Scripts\python.exe -m core.datasets.faceforensics --instructions

# Run automated verification benchmark on downloaded FaceForensics++ samples
.venv\Scripts\python.exe -m core.datasets.faceforensics --dataset-dir data/faceforensics --max-samples 15
```

Outputs a comprehensive verification report to `data/verification_report.json` containing AUC, EER, precision/recall, and per-manipulation category statistics.

---

## 🔬 Model Adapter Integration (SyncNet / Wav2Lip)

The system exposes an abstract `LipSyncModel` interface in [`core/models/base.py`](file:///core/models/base.py).

- **`HeuristicLipSyncModel`**: Fully working default implementation analyzing dense optical flow and acoustic envelopes.
- **`PretrainedSyncNetAdapter`**: Drop-in wrapper located at [`core/models/syncnet.py`](file:///core/models/syncnet.py). To enable neural inference:
  1. Install PyTorch (`pip install torch torchvision`).
  2. Set `SYNCNET_CHECKPOINT_PATH=path/to/syncnet_weights.pth` in `.env`.
  3. When PyTorch or checkpoint weights are not present, the adapter automatically falls back to heuristic analysis and explicitly marks `"analysis_mode": "heuristic"`.

---

## ⚠️ Known Forensic Boundaries & Limitations

1. **Probabilistic Nature**: Authenticity and fake probabilities are statistical forensic indicators, not absolute proof of manipulation.
2. **Transmission & Encoding Latency**: Hardware capture latency, Bluetooth microphones, or VoIP re-encoding can introduce benign audio-video offsets of ±60ms to ±120ms.
3. **Acoustic Edge Cases**: Whispering, heavy background noise, or dubbing with matching cadence can decrease correlation reliability.
4. **Extreme Angles**: Non-frontal head orientations (> 45° yaw) degrade 2D mouth region tracking.
