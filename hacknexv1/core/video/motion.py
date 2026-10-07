"""Mouth-motion energy calculation across frame sequences."""
import logging
from typing import List
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def compute_mouth_motion_energy(mouth_crops: List[np.ndarray]) -> np.ndarray:
    """Compute normalized mouth motion dynamics across sampled frames.
    
    Uses dense optical flow (Farneback) complemented by intensity differences
    on registered mouth crops to capture articulation velocity.
    
    Args:
        mouth_crops: List of grayscale mouth images (96x96).
        
    Returns:
        1D numpy array of normalized mouth motion energy in [0.0, 1.0] of length len(mouth_crops).
    """
    n = len(mouth_crops)
    if n == 0:
        return np.array([], dtype=np.float32)
    if n == 1:
        return np.array([0.0], dtype=np.float32)

    energies = np.zeros(n, dtype=np.float32)

    for i in range(1, n):
        prev_crop = mouth_crops[i - 1]
        curr_crop = mouth_crops[i]

        # 1. Compute Farneback Optical Flow on 96x96 mouth ROI
        try:
            flow = cv2.calcOpticalFlowFarneback(
                prev_crop,
                curr_crop,
                None,
                pyr_scale=0.5,
                levels=2,
                winsize=13,
                iterations=2,
                poly_n=5,
                poly_sigma=1.1,
                flags=0,
            )
            # Magnitude of velocity vectors
            mag = np.hypot(flow[..., 0], flow[..., 1])
            # Weighted focus on center of mouth (inner lips)
            h, w = mag.shape
            center_h_start, center_h_end = int(h * 0.2), int(h * 0.8)
            center_w_start, center_w_end = int(w * 0.2), int(w * 0.8)
            center_mag = mag[center_h_start:center_h_end, center_w_start:center_w_end]
            optical_energy = float(np.mean(center_mag))
        except Exception as e:
            # Fallback to absolute difference if optical flow encounters exception
            diff = cv2.absdiff(prev_crop, curr_crop)
            optical_energy = float(np.mean(diff) / 255.0)

        # 2. Complemented by normalized absolute difference
        diff_val = float(np.mean(cv2.absdiff(prev_crop, curr_crop))) / 255.0
        
        # Combined dynamic articulatory energy
        energies[i] = optical_energy * 0.7 + diff_val * 0.3

    # For the initial frame, set energy to the second frame's value to avoid artificial 0-drop
    energies[0] = energies[1] if n > 1 else 0.0

    # Moving average smoothing over 3 frames
    kernel = np.ones(3) / 3.0
    smoothed = np.convolve(energies, kernel, mode="same")

    # Min-max normalization with percentile clipping
    p5 = np.percentile(smoothed, 5)
    p95 = np.percentile(smoothed, 95)
    spread = p95 - p5

    if spread > 1e-5:
        norm_energies = np.clip((smoothed - p5) / spread, 0.0, 1.0)
    else:
        norm_energies = np.zeros_like(smoothed)

    return norm_energies.astype(np.float32)
