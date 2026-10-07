# Samples Directory

This directory stores test images and benchmark assets for evaluating the Visual Forensics detector.

## Included Test Samples
1. **`portrait_face.jpg`**: High-resolution single adult portrait.
   - Used for evaluating single-face detection, 5-point landmark extraction, similarity transform alignment, and 256x256 tensor generation.
2. **`multi_faces.jpg`**: Multi-person photograph containing two distinct faces.
   - Used for evaluating multi-face detection, per-face alignment, and confidence ranking.
3. **`sample.jpg`**: Abstract background without human faces.
   - Used for verifying graceful no-face handling without pipeline crashes (`NO_FACE_DETECTED`).

## Quick CLI Testing
```bash
# Test single face alignment and save crops
python demo.py samples/portrait_face.jpg --save-crops

# Test multi-face detection
python demo.py samples/multi_faces.jpg --save-crops

# Test graceful handling with no face
python demo.py samples/sample.jpg
```
