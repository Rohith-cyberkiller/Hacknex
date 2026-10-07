"""Score fusion module for dual-branch deepfake detection (Phase 3).

Combines the RGB spatial branch and frequency-domain branch scores
into a single image-level visual authenticity score.

Fusion Strategy:
  Per-face fusion: weighted average of RGB and frequency scores.
    fused_face = w_rgb * rgb_score + w_freq * freq_score
    Default weights: w_rgb=0.6, w_freq=0.4
    Rationale: RGB spatial features tend to be more discriminative for
    face-swap artifacts, while frequency features add complementary
    sensitivity to GAN checkerboard patterns and synthesis fingerprints.

  Image-level aggregation: minimum of per-face fused scores.
    If ANY face in the image is suspicious, the image is flagged.
    This is the conservative approach used in FaceForensics++ evaluation.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Fusion")


class ScoreFusion:
    """Fuses RGB and frequency branch scores into a single authenticity score."""

    def __init__(
        self,
        rgb_weight: float = 0.6,
        freq_weight: float = 0.4,
        aggregation: str = "min",
    ) -> None:
        """Initialize fusion module.

        Args:
            rgb_weight: Weight for RGB branch score.
            freq_weight: Weight for frequency branch score.
            aggregation: Multi-face aggregation method ('min', 'mean', 'max').
        """
        assert abs(rgb_weight + freq_weight - 1.0) < 1e-6, "Weights must sum to 1.0"
        self.rgb_weight = rgb_weight
        self.freq_weight = freq_weight
        self.aggregation = aggregation

    def fuse_per_face(
        self,
        rgb_scores: List[float],
        freq_scores: List[float],
    ) -> List[float]:
        """Compute per-face fused authenticity scores.

        Args:
            rgb_scores: Per-face RGB branch scores [0, 1].
            freq_scores: Per-face frequency branch scores [0, 1].

        Returns:
            List[float]: Per-face fused scores [0, 1].
        """
        assert len(rgb_scores) == len(freq_scores), "Score lists must have equal length"
        fused = []
        for rgb_s, freq_s in zip(rgb_scores, freq_scores):
            f = self.rgb_weight * rgb_s + self.freq_weight * freq_s
            fused.append(float(np.clip(f, 0.0, 1.0)))
        return fused

    def aggregate_image_score(self, per_face_scores: List[float]) -> float:
        """Aggregate per-face scores into a single image-level score.

        Args:
            per_face_scores: Fused scores for each detected face.

        Returns:
            float: Image-level authenticity score [0, 1].
        """
        if not per_face_scores:
            return 1.0  # No face → no facial manipulation detected

        if self.aggregation == "min":
            return float(min(per_face_scores))
        elif self.aggregation == "mean":
            return float(np.mean(per_face_scores))
        elif self.aggregation == "max":
            return float(max(per_face_scores))
        else:
            return float(min(per_face_scores))

    def fuse(
        self,
        rgb_scores: List[float],
        freq_scores: List[float],
    ) -> Tuple[float, List[float], Dict]:
        """Full fusion pipeline: per-face → image-level.

        Returns:
            Tuple of:
                - image_score: float, image-level authenticity [0, 1]
                - per_face_scores: list of per-face fused scores
                - details: dict with diagnostic info
        """
        per_face = self.fuse_per_face(rgb_scores, freq_scores)
        image_score = self.aggregate_image_score(per_face)

        details = {
            "rgb_scores": rgb_scores,
            "freq_scores": freq_scores,
            "per_face_fused": per_face,
            "rgb_weight": self.rgb_weight,
            "freq_weight": self.freq_weight,
            "aggregation": self.aggregation,
            "image_score": image_score,
        }

        return image_score, per_face, details
