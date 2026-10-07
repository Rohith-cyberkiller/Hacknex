"""Image degradation utilities for compression hardening (Phase 5).

Simulates real-world social-media re-encoding and lossy compression that
images undergo when shared on platforms like Instagram, WhatsApp, Facebook,
and Twitter/X.

Supported degradation types:
  1. JPEG Compression: Quality levels 30-95 (social media typically uses Q=60-85)
  2. Gaussian Blur: Kernel sizes 3-9 (simulates camera shake / platform downscaling)
  3. Downscale-Upscale: Factors 0.25-0.75 (bicubic down, bilinear up)
  4. Combined: Sequential application of multiple degradations

These utilities serve two purposes:
  a) Training-time augmentation (Phase 5+ when fine-tuning on deepfake data)
  b) Evaluation: comparing model robustness on clean vs degraded inputs
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("Degradation")


@dataclass
class DegradationResult:
    """Result of applying a degradation to an image."""
    degraded_image: np.ndarray
    degradation_type: str
    params: Dict
    psnr: Optional[float] = None


def jpeg_compress(image: np.ndarray, quality: int = 75) -> np.ndarray:
    """Apply JPEG compression and re-decode.

    Args:
        image: BGR uint8 image.
        quality: JPEG quality (1-100). Lower = more compression artifacts.

    Returns:
        np.ndarray: Re-decoded BGR image after JPEG round-trip.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer)
    result_rgb = np.array(compressed)
    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)


def gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply Gaussian blur.

    Args:
        image: BGR uint8 image.
        kernel_size: Odd integer kernel size.

    Returns:
        np.ndarray: Blurred image.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def downscale_upscale(image: np.ndarray, scale_factor: float = 0.5) -> np.ndarray:
    """Simulate resolution degradation via downscale + upscale.

    Args:
        image: BGR uint8 image.
        scale_factor: Factor to downscale (e.g., 0.5 = half resolution).

    Returns:
        np.ndarray: Image restored to original resolution after downscaling.
    """
    h, w = image.shape[:2]
    small_h = max(1, int(h * scale_factor))
    small_w = max(1, int(w * scale_factor))
    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return restored


def compute_psnr(original: np.ndarray, degraded: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio between original and degraded images."""
    mse = np.mean((original.astype(np.float64) - degraded.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return float("inf")
    return float(10.0 * np.log10(255.0 ** 2 / mse))


# ─── Preset degradation profiles ──────────────────────────────────────────────

DEGRADATION_PROFILES = {
    "clean": [],
    "jpeg_q30": [("jpeg", {"quality": 30})],
    "jpeg_q50": [("jpeg", {"quality": 50})],
    "jpeg_q70": [("jpeg", {"quality": 70})],
    "jpeg_q85": [("jpeg", {"quality": 85})],
    "blur_k3": [("blur", {"kernel_size": 3})],
    "blur_k5": [("blur", {"kernel_size": 5})],
    "blur_k7": [("blur", {"kernel_size": 7})],
    "downscale_0.5": [("downscale", {"scale_factor": 0.5})],
    "downscale_0.25": [("downscale", {"scale_factor": 0.25})],
    "social_media_mild": [
        ("jpeg", {"quality": 85}),
        ("downscale", {"scale_factor": 0.75}),
    ],
    "social_media_heavy": [
        ("downscale", {"scale_factor": 0.5}),
        ("jpeg", {"quality": 50}),
        ("blur", {"kernel_size": 3}),
    ],
}


def apply_degradation_profile(
    image: np.ndarray,
    profile_name: str,
) -> DegradationResult:
    """Apply a named degradation profile to an image.

    Args:
        image: BGR uint8 image.
        profile_name: Key from DEGRADATION_PROFILES.

    Returns:
        DegradationResult: Degraded image with metadata.
    """
    steps = DEGRADATION_PROFILES.get(profile_name, [])
    result = image.copy()

    for step_type, params in steps:
        if step_type == "jpeg":
            result = jpeg_compress(result, **params)
        elif step_type == "blur":
            result = gaussian_blur(result, **params)
        elif step_type == "downscale":
            result = downscale_upscale(result, **params)

    psnr = compute_psnr(image, result) if profile_name != "clean" else float("inf")

    return DegradationResult(
        degraded_image=result,
        degradation_type=profile_name,
        params={"steps": steps},
        psnr=psnr,
    )


def apply_all_profiles(image: np.ndarray) -> Dict[str, DegradationResult]:
    """Apply all degradation profiles and return results.

    Args:
        image: BGR uint8 image.

    Returns:
        Dict mapping profile name → DegradationResult.
    """
    results = {}
    for name in DEGRADATION_PROFILES:
        results[name] = apply_degradation_profile(image, name)
    return results
