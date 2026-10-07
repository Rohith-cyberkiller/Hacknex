"""Frequency-domain branch for deepfake detection (Phase 3).

Analyzes high-frequency forensic traces that are invisible in the RGB domain
but highly discriminative for GAN / diffusion model artifacts.

Two complementary feature extraction methods:

1. 2D FFT Magnitude Spectrum
   - Computes the log-magnitude of the 2D Discrete Fourier Transform.
   - GANs produce characteristic spectral fingerprints (checkerboard patterns
     from transposed convolutions, periodic spectral peaks from upsampling).
   - The log-magnitude spectrum is treated as a single-channel image and
     processed through a lightweight CNN.

2. SRM (Spatial Rich Model) High-Pass Residual Filters
   - Applies 3 standard SRM kernels to extract noise residuals:
     a) 1st-order edge filter
     b) 2nd-order edge filter  
     c) 3rd-order SQUARE filter
   - These residuals capture subtle pixel-level inconsistencies from
     neural rendering that are suppressed in natural photographs.
   - The 3 residual channels are concatenated and processed through a
     small CNN classifier.

Fusion: FFT and SRM classifier logits are averaged before softmax.

Reference:
  - Fridrich & Kodovsky, "Rich Models for Steganalysis of Digital Images", 2012
  - Zhang et al., "Detecting and Simulating Artifacts in GAN Fake Images", 2019
  - Frank et al., "Leveraging Frequency Analysis for Deep Fake Image Recognition", 2020
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("FrequencyBranch")

# ─── SRM Filter Kernels ───────────────────────────────────────────────────────
# Standard Spatial Rich Model high-pass filters for noise residual extraction.

SRM_FILTER_1 = np.array([
    [ 0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0],
    [ 0,  1, -2,  1,  0],
    [ 0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0],
], dtype=np.float32)

SRM_FILTER_2 = np.array([
    [ 0,  0,  0,  0,  0],
    [ 0,  0,  1,  0,  0],
    [ 0,  1, -4,  1,  0],
    [ 0,  0,  1,  0,  0],
    [ 0,  0,  0,  0,  0],
], dtype=np.float32)

SRM_FILTER_3 = np.array([
    [-1,  2, -2,  2, -1],
    [ 2, -6,  8, -6,  2],
    [-2,  8, -12, 8, -2],
    [ 2, -6,  8, -6,  2],
    [-1,  2, -2,  2, -1],
], dtype=np.float32) / 12.0


def compute_fft_magnitude(image_gray: np.ndarray) -> np.ndarray:
    """Compute the log-magnitude spectrum of the 2D FFT.

    Args:
        image_gray: Grayscale image (H, W) as float32 or uint8.

    Returns:
        np.ndarray: Log-magnitude spectrum (H, W) normalized to [0, 1].
    """
    if image_gray.dtype == np.uint8:
        image_gray = image_gray.astype(np.float32) / 255.0

    # 2D FFT → shift zero-frequency to center
    f_transform = np.fft.fft2(image_gray)
    f_shifted = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shifted)

    # Log scale to compress dynamic range
    log_mag = np.log1p(magnitude)

    # Normalize to [0, 1]
    min_val = log_mag.min()
    max_val = log_mag.max()
    if max_val > min_val:
        log_mag = (log_mag - min_val) / (max_val - min_val)
    else:
        log_mag = np.zeros_like(log_mag)

    return log_mag.astype(np.float32)


def apply_srm_filters(image_gray: np.ndarray) -> np.ndarray:
    """Apply 3 SRM high-pass filters to extract noise residuals.

    Args:
        image_gray: Grayscale image (H, W) as float32 [0, 1].

    Returns:
        np.ndarray: Stacked residual maps (3, H, W) as float32.
    """
    import cv2

    if image_gray.dtype == np.uint8:
        image_gray = image_gray.astype(np.float32) / 255.0

    residuals = []
    for kernel in [SRM_FILTER_1, SRM_FILTER_2, SRM_FILTER_3]:
        filtered = cv2.filter2D(image_gray, -1, kernel, borderType=cv2.BORDER_REFLECT)
        residuals.append(filtered)

    return np.stack(residuals, axis=0).astype(np.float32)


def prepare_frequency_features(rgb_crop: np.ndarray) -> dict:
    """Extract both FFT magnitude and SRM residual features from an RGB face crop.

    Args:
        rgb_crop: Face crop (H, W, 3) in RGB format, uint8.

    Returns:
        dict with keys:
            'fft_tensor': (1, H, W) tensor of log-magnitude spectrum
            'srm_tensor': (3, H, W) tensor of SRM noise residuals
    """
    import cv2

    gray = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY)

    fft_mag = compute_fft_magnitude(gray)
    fft_tensor = torch.from_numpy(fft_mag).unsqueeze(0).float()

    srm_residuals = apply_srm_filters(gray.astype(np.float32) / 255.0 if gray.dtype == np.uint8 else gray)
    srm_tensor = torch.from_numpy(srm_residuals).float()

    return {"fft_tensor": fft_tensor, "srm_tensor": srm_tensor}


class FrequencyClassifier(nn.Module):
    """Lightweight CNN classifier for frequency-domain features.

    Processes either single-channel FFT magnitude or 3-channel SRM residuals.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, C, H, W) frequency feature tensor.

        Returns:
            torch.Tensor: Logits (B, 2).
        """
        feat = self.conv_block(x)
        return self.fc(feat)


class FrequencyBranch:
    """Inference wrapper for the frequency-domain branch.

    Manages FFT and SRM feature extraction, classification, and scoring.
    """

    def __init__(self, device: str = "cpu", weights_path: Optional[str] = None) -> None:
        self.device = torch.device(device)
        self.fft_classifier = FrequencyClassifier(in_channels=1, num_classes=2)
        self.srm_classifier = FrequencyClassifier(in_channels=3, num_classes=2)
        self._weights_loaded = False

        if weights_path is not None:
            try:
                state = torch.load(weights_path, map_location=self.device)
                self.fft_classifier.load_state_dict(state.get("fft", state))
                self.srm_classifier.load_state_dict(state.get("srm", state))
                self._weights_loaded = True
                logger.info("Loaded trained frequency branch weights from: %s", weights_path)
            except Exception as err:
                logger.warning("Could not load frequency weights: %s", err)

        self.fft_classifier.to(self.device).eval()
        self.srm_classifier.to(self.device).eval()

        if not self._weights_loaded:
            logger.info(
                "Frequency Branch initialized with untrained CNN heads. "
                "Fine-tune on deepfake dataset for production accuracy."
            )

    @torch.no_grad()
    def score_faces(self, face_crops_rgb: List[np.ndarray]) -> List[float]:
        """Compute frequency-domain authenticity scores for face crops.

        Args:
            face_crops_rgb: List of (H, W, 3) RGB uint8 face crops.

        Returns:
            List[float]: Per-face authenticity scores in [0.0, 1.0] (1.0 = real).
        """
        if not face_crops_rgb:
            return []

        fft_batch = []
        srm_batch = []

        for crop in face_crops_rgb:
            features = prepare_frequency_features(crop)
            fft_batch.append(features["fft_tensor"])
            srm_batch.append(features["srm_tensor"])

        fft_input = torch.stack(fft_batch).to(self.device)
        srm_input = torch.stack(srm_batch).to(self.device)

        fft_logits = self.fft_classifier(fft_input)
        srm_logits = self.srm_classifier(srm_input)

        # Average logits before softmax (late fusion of FFT + SRM)
        combined_logits = (fft_logits + srm_logits) / 2.0
        probs = F.softmax(combined_logits, dim=1)

        # Column 0 = P(real)
        real_probs = probs[:, 0].cpu().tolist()
        return real_probs
