# Visual Forensics & Spatial Deepfake Detection

**Visual Forensics** is an end-to-end, modular computer vision and deep learning forensics system engineered to detect facial manipulation, AI-generated imagery, and deepfakes through spatial artifact inspection, frequency-domain analysis (FFT & SRM), explainable Grad-CAM heatmaps, and post-processing robustness hardening.

---

## Roadmap & Phase Status

| Phase | Scope & Objective | Status |
| :--- | :--- | :--- |
| **Phase 1** | **Scaffolding & Architecture Foundation**<br>• Clean Python project structure & typed contracts (`DetectionResult`, `AlignedFace`)<br>• Public API skeleton (`visual_detector.py`)<br>• Deterministic mock pipeline logging future stages<br>• Zero-dependency CLI demo (`demo.py`) with Grad-CAM overlay preview | **COMPLETED** |
| **Phase 2** | **Face Extraction & Preprocessing Pipeline**<br>• Real face detection with MTCNN (`facenet-pytorch`)<br>• 5-point facial landmark localization (eyes, nose, mouth corners)<br>• Similarity transform alignment (affine warping to standard canonical reference)<br>• Standardized cropped face tensors (256×256) with ImageNet normalization<br>• Multi-face & robust no-face handling (graceful non-crashing execution) | **COMPLETED** |
| **Phase 3** | **Dual-Branch Multimodal Detector & Fusion**<br>• Branch A: Spatial RGB CNN (ImageNet-pretrained EfficientNet-B0 backbone)<br>• Branch B: Frequency Domain (2D FFT log-magnitude spectrum + 3-scale SRM high-pass residual filter bank)<br>• Score Fusion Layer: Weighted per-face combination ($0.6 \times \text{RGB} + 0.4 \times \text{Freq}$) & conservative multi-face minimum aggregation | **COMPLETED** |
| **Phase 4** | **Grad-CAM Explainability & Heatmap Compositing**<br>• Backprop gradient attribution hooked onto EfficientNet-B0 final convolutional features<br>• Class-targeted activation mapping (targeted toward fake class index 1 for anomaly highlighting)<br>• Per-face crop normalization and multi-face frame compositing via spatial max-pooling | **COMPLETED** |
| **Phase 5** | **Post-Processing Robustness & Degradation Hardening**<br>• Social-media re-encoding simulation (JPEG compression $Q=30..90$, Gaussian blur, downscale-upscale cycles)<br>• PSNR (Peak Signal-to-Noise Ratio) distortion tracking<br>• Robustness evaluation engine (`evaluate_robustness`) & CLI benchmark table | **COMPLETED** |

---

## Architectural Deep Dive

```
                             ┌────────────────────────────────────────┐
                             │       Raw Input Frame (BGR / RGB)      │
                             └───────────────────┬────────────────────┘
                                                 │
                                                 ▼
                        ┌──────────────────────────────────────────────────┐
                        │      Stage 1: Face Detection & Alignment         │
                        │ • MTCNN face detection (bounding box & confidence)│
                        │ • 5-point landmark extraction (eyes, nose, mouth)│
                        │ • Similarity transform alignment (256×256 crop)  │
                        │ • ImageNet tensor normalization [3, 256, 256]    │
                        └─────────┬──────────────────────────────┬─────────┘
                                  │                              │
         Normalized RGB Tensor    │                              │ Aligned RGB Crop
                                  ▼                              ▼
     ┌────────────────────────────────────────┐   ┌────────────────────────────────────────┐
     │      Stage 2A: Spatial RGB Branch      │   │     Stage 2B: Frequency Domain Branch  │
     │ • Backbone: EfficientNet-B0            │   │ • 2D FFT Log-Magnitude Spectrum       │
     │ • ImageNet pretrained texture features │   │ • 3-scale SRM High-Pass Filter Bank    │
     │ • Linear classification head           │   │ • Dual-head CNN feature classification │
     └───────────────────┬────────────────────┘   └───────────────────┬────────────────────┘
                         │                                            │
               RGB Score │                                 Freq Score │
                         └───────────────────┬────────────────────────┘
                                             │
                                             ▼
                        ┌──────────────────────────────────────────────────┐
                        │          Stage 2C: Decision Score Fusion         │
                        │ • Per-face: 0.60 * RGB + 0.40 * Freq             │
                        │ • Multi-face: Minimum aggregation (worst-case)   │
                        └────────────────────┬─────────────────────────────┘
                                             │
                                             ▼
                        ┌──────────────────────────────────────────────────┐
                        │         Stage 3: Grad-CAM Explainability         │
                        │ • Backprop gradients w.r.t. fake class (idx=1)   │
                        │ • Channel-weighted feature map pooling           │
                        │ • Spatial mapping & multi-face max compositing   │
                        └────────────────────┬─────────────────────────────┘
                                             │
                                             ▼
                        ┌──────────────────────────────────────────────────┐
                        │         Stage 4: Calibration & Hardening         │
                        │ • Score calibration to [0.0, 1.0]                │
                        │ • Post-processing & compression resilience checks│
                        └──────────────────────────────────────────────────┘
```

---

## Repository Structure

```
hacknex/
├── README.md               # Complete architecture, technical specs & usage guide
├── requirements.txt        # Pinned core dependencies (torch, torchvision, facenet-pytorch, cv2)
├── visual_detector.py      # Main public API (VisualDetector, detect_visual_authenticity, evaluate_robustness)
├── demo.py                 # Multi-stage CLI demo with degradation simulation & robustness benchmark
├── src/                    # Internal engine & forensic stages
│   ├── __init__.py         # Package exports across all stages
│   ├── types.py            # Typed dataclasses (DetectionResult, AlignedFace, BoundingBox, FacialLandmarks)
│   ├── face_extraction.py  # Stage 1: MTCNN detector, similarity alignment, 256x256 crops & normalization
│   ├── rgb_branch.py       # Stage 2A: EfficientNet-B0 spatial classifier & feature map extraction
│   ├── frequency_branch.py # Stage 2B: 2D FFT magnitude spectrum & 3-scale SRM filter bank
│   ├── fusion.py           # Stage 2C: Score fusion (weighted combination & multi-face aggregation)
│   ├── gradcam.py          # Stage 3: Grad-CAM backprop attribution & multi-face compositing
│   ├── degradation.py      # Stage 5: JPEG compression, Gaussian blur, downscale-upscale, PSNR
│   ├── pipeline.py         # End-to-end 4-stage pipeline orchestrator
│   └── utils.py            # Image I/O, validation, crop exporting, and overlay rendering
├── samples/                # Benchmark test assets
│   ├── portrait_face.jpg   # Sample single-face photo
│   ├── multi_faces.jpg     # Sample multi-face photo (2 faces)
│   ├── sample.jpg          # No-face test image
│   └── README.md           # Sample asset guide
├── outputs/                # Generated visual outputs & Grad-CAM overlays
│   ├── crops/              # Exported aligned 256x256 face crops
│   └── demo_overlay.jpg    # Visual overlay with bounding boxes, landmarks & Grad-CAM heatmap
└── tests/                  # Complete automated test suite (31 tests)
    ├── __init__.py
    ├── test_visual_detector.py # End-to-end detector tests (single, multi, no-face, contracts)
    ├── test_dual_branch.py     # RGB branch, frequency branch (FFT/SRM), and fusion tests
    ├── test_gradcam.py         # Grad-CAM attribution, hook lifecycle & heatmap compositing
    └── test_degradation.py     # JPEG, Gaussian blur, downscaling, PSNR & robustness tests
```

---

## Installation

```bash
# 1. Clone repository
git clone <repo-url>
cd hacknex

# 2. Set up virtual environment
python -m venv venv
.\venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt
```

---

## CLI Demo Usage

### 1. Run Complete Multimodal Detection on Single Face
```bash
python demo.py samples/portrait_face.jpg --save-crops
```
*Output: Detects face with MTCNN, extracts 5 landmarks, runs RGB + Frequency dual-branch inference, generates Grad-CAM heatmap, saves aligned crop in `outputs/crops/`, and writes annotated overlay in `outputs/demo_overlay.jpg`.*

### 2. Run Multi-Face Detection
```bash
python demo.py samples/multi_faces.jpg --save-crops
```
*Output: Detects and aligns multiple faces independently, scores each face through the dual-branch network, composites multi-face Grad-CAM heatmaps, and applies conservative minimum-score aggregation.*

### 3. Test Graceful No-Face Handling
```bash
python demo.py samples/sample.jpg
```
*Output: Logs `[INFO] No human faces detected`, sets verdict to `NO_FACE_DETECTED`, authenticity score to `1.0`, and exits cleanly without crashing.*

### 4. Test Simulated Social-Media JPEG Compression
```bash
python demo.py samples/portrait_face.jpg --degrade jpeg_q50
```
*Output: Simulates lossy WhatsApp/Instagram JPEG re-encoding at quality 50, displays PSNR distortion (dB), and evaluates score stability.*

### 5. Run Full Robustness Benchmark Table
```bash
python demo.py samples/portrait_face.jpg --robustness-benchmark
```
*Output: Evaluates input against 10 compression, blur, and social-media degradation profiles:*
```
==============================================================================
  PHASE 5: SOCIAL-MEDIA COMPRESSION & DEGRADATION ROBUSTNESS BENCHMARK
==============================================================================
Profile                | PSNR (dB)  | Score    | Delta    | Faces  | Verdict
------------------------------------------------------------------------------
clean                  | Inf (Clean)| 0.5075   | +0.0000  | 1      | REAL
jpeg_q85               | 41.87      | 0.5098   | +0.0023  | 1      | REAL
jpeg_q70               | 39.94      | 0.5098   | +0.0023  | 1      | REAL
jpeg_q50               | 38.36      | 0.5082   | +0.0007  | 1      | REAL
jpeg_q30               | 36.41      | 0.5109   | +0.0033  | 1      | REAL
blur_k3                | 41.71      | 0.5044   | -0.0032  | 1      | REAL
blur_k5                | 39.36      | 0.5095   | +0.0019  | 1      | REAL
downscale_0.5          | 39.08      | 0.5059   | -0.0016  | 1      | REAL
social_media_mild      | 40.33      | 0.5076   | +0.0000  | 1      | REAL
social_media_heavy     | 36.62      | 0.5146   | +0.0070  | 1      | REAL
==============================================================================
```

---

## Public Python API

```python
from visual_detector import VisualDetector, detect_visual_authenticity, evaluate_robustness

# 1. Simple functional detection
result = detect_visual_authenticity("samples/portrait_face.jpg")

print(f"Prediction Verdict         : {result.predicted_label}")
print(f"Visual Authenticity Score  : {result.visual_authenticity_score:.4f}")
print(f"Manipulation Probability   : {result.manipulation_probability:.4f}")
print(f"Detected Faces             : {len(result.faces)}")

for idx, face in enumerate(result.faces):
    print(f"  Face #{idx+1} Bounding Box : {face.box.as_tuple()}")
    print(f"  Face #{idx+1} Crop Shape   : {face.crop_bgr.shape}")          # (256, 256, 3)
    print(f"  Face #{idx+1} Tensor Shape : {face.normalized_tensor.shape}")  # torch.Size([3, 256, 256])

# Access dual-branch breakdown from metadata
print(f"RGB Scores                 : {result.metadata['rgb_scores']}")
print(f"Frequency Scores           : {result.metadata['freq_scores']}")
print(f"Fused Per-Face Scores      : {result.metadata['per_face_fused']}")
print(f"Grad-CAM Heatmap Shape     : {result.gradcam_heatmap_array.shape}")

# 2. Evaluate compression robustness
report = evaluate_robustness("samples/portrait_face.jpg")
for profile, stats in report.items():
    print(f"{profile}: Score={stats['score']:.4f}, PSNR={stats['psnr']} dB, Verdict={stats['label']}")
```

### Output Contract: `DetectionResult`

| Field | Type | Description |
| :--- | :--- | :--- |
| `visual_authenticity_score` | `float` | Score in $[0.0, 1.0]$. Higher = more authentic ($1.0$ = Real, $0.0$ = Fake). |
| `manipulation_probability` | `float` | Probability of manipulation: $1.0 - \text{score}$. |
| `predicted_label` | `str` | `"REAL"`, `"FAKE"`, or `"NO_FACE_DETECTED"`. |
| `bounding_boxes` | `List[Tuple[int, int, int, int]]` | Real detected face boxes `(x1, y1, x2, y2)` in original coordinates. |
| `faces` | `List[AlignedFace]` | Aligned face crops, 5-point landmarks, and normalized PyTorch tensors. |
| `gradcam_heatmap_array` | `np.ndarray` | 2D float array in $[0.0, 1.0]$ matching frame dimensions. |
| `stage_logs` | `List[StageExecutionLog]` | Stage logs tracking execution metrics and detector outputs. |
| `metadata` | `Dict[str, Any]` | Detailed stage metrics (`rgb_scores`, `freq_scores`, `per_face_fused`, `weights_trained`). |

---

## Running Automated Tests

Run the complete test suite verifying face extraction, dual-branch models, Grad-CAM hooks, degradation simulations, and contract compatibility:

```bash
python -m unittest discover -s tests -v
```

**Test Suite Coverage (31 passing tests):**
- `test_visual_detector.py` (6 tests): Single-face, multi-face, no-face graceful exit, blank frame, contract compatibility, crop exporting.
- `test_dual_branch.py` (10 tests): EfficientNet-B0 forward pass, feature map extraction, 2D FFT magnitude spectrum, SRM 3-channel filter residuals, dual-branch scoring, fusion calculations, and multi-face aggregation modes.
- `test_gradcam.py` (5 tests): Grad-CAM hook registration & cleanup, heatmap range $[0.0, 1.0]$, multi-face batch generation, frame coordinate mapping, and spatial max-pooling compositing.
- `test_degradation.py` (10 tests): JPEG compression, Gaussian blur, downscale-upscale cycles, PSNR calculations, degradation preset suite, and robustness evaluation.

---

## Target Datasets for Future Fine-Tuning

1. **FaceForensics++ (FF++)**
   - Repositories: [ondyari/FaceForensics](https://github.com/ondyari/FaceForensics) & [PalakMah/FaceForensics](https://github.com/PalakMah/FaceForensics)
   - Benchmark deepfake manipulation methods: Deepfakes, Face2Face, FaceSwap, NeuralTextures.
2. **DeeperForensics-1.0**
   - Repository: [EndlessSora/DeeperForensics-1.0](https://github.com/EndlessSora/DeeperForensics-1.0)
   - High-quality wild scenarios with real-world distortions.
3. **GenImage**
   - Repository: [GenImage-Dataset/GenImage](https://github.com/GenImage-Dataset/GenImage)
   - Diffusion and GAN synthesis models: Stable Diffusion, Midjourney, StyleGAN, ADM, VQ-GAN.
