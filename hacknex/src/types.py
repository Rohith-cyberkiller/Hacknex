"""Data types and result schemas for Visual Forensics Deepfake Detection.

This module defines the typed output contract for the visual detector,
ensuring strict type safety and structured outputs across Phase 1 stubs,
Phase 2 real face extraction & alignment, and future ML model stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch


class InputColorSpace(str, Enum):
    """Color space format for incoming raw frames."""
    BGR = "BGR"
    RGB = "RGB"


@dataclass(frozen=True)
class FacialLandmarks:
    """5-point facial landmark coordinates (in pixels).

    Standard landmarks:
    - left_eye: (x, y)
    - right_eye: (x, y)
    - nose: (x, y)
    - left_mouth: (x, y)
    - right_mouth: (x, y)
    """
    left_eye: Tuple[float, float]
    right_eye: Tuple[float, float]
    nose: Tuple[float, float]
    left_mouth: Tuple[float, float]
    right_mouth: Tuple[float, float]

    def as_array(self) -> np.ndarray:
        """Return 5x2 numpy array of landmark coordinates."""
        return np.array(
            [
                self.left_eye,
                self.right_eye,
                self.nose,
                self.left_mouth,
                self.right_mouth,
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class BoundingBox:
    """Bounding box coordinates for detected faces or manipulated regions.

    Coordinates are defined in pixels as (x1, y1, x2, y2):
    - x1, y1: top-left corner
    - x2, y2: bottom-right corner
    """
    x1: int
    y1: int
    x2: int
    y2: int
    label: str = "face"
    confidence: float = 1.0

    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Return coordinates as (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)

    def width(self) -> int:
        """Width of bounding box in pixels."""
        return max(0, self.x2 - self.x1)

    def height(self) -> int:
        """Height of bounding box in pixels."""
        return max(0, self.y2 - self.y1)


@dataclass
class AlignedFace:
    """Preprocessed and geometrically aligned face crop ready for model inference.

    Attributes:
        box: Detected face bounding box in original image coordinates.
        confidence: Detection confidence score from face detector (0.0 to 1.0).
        landmarks: 5-point facial landmarks if detected.
        crop_bgr: Aligned face image in BGR format (H, W, 3) uint8.
        crop_rgb: Aligned face image in RGB format (H, W, 3) uint8.
        normalized_tensor: PyTorch tensor (3, H, W) normalized with ImageNet mean/std.
        target_size: Resolution of the aligned square crop (e.g., 256 for 256x256).
    """
    box: BoundingBox
    confidence: float
    crop_bgr: np.ndarray
    crop_rgb: np.ndarray
    landmarks: Optional[FacialLandmarks] = None
    normalized_tensor: Optional[torch.Tensor] = None
    target_size: int = 256


@dataclass
class StageExecutionLog:
    """Execution metadata and simulation log for individual pipeline stages."""
    stage_name: str
    status: str
    description: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionResult:
    """Primary output contract for visual forensics detection.

    Attributes:
        visual_authenticity_score: Probability that the image/frame is authentic (real).
            Value in [0.0, 1.0], where 1.0 = highly authentic / real, 0.0 = deepfake / manipulated.
        bounding_boxes: List of bounding boxes (x1, y1, x2, y2) for suspected manipulation / detected faces.
        gradcam_heatmap_array: 2D numpy array (H, W) in range [0.0, 1.0] suitable for UI overlay.
        faces: Structured list of real detected & aligned face crops (Phase 2 extension).
        stage_logs: List of logged stages for auditing and explainability.
        metadata: Arbitrary metadata (e.g., input resolution, frame index, model version).
    """
    visual_authenticity_score: float
    bounding_boxes: List[Tuple[int, int, int, int]]
    gradcam_heatmap_array: np.ndarray
    faces: List[AlignedFace] = field(default_factory=list)
    stage_logs: List[StageExecutionLog] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def face_detected(self) -> bool:
        """True if one or more faces were detected in the input."""
        return len(self.bounding_boxes) > 0

    @property
    def manipulation_probability(self) -> float:
        """Probability of manipulation: 1.0 - visual_authenticity_score."""
        return float(np.clip(1.0 - self.visual_authenticity_score, 0.0, 1.0))

    @property
    def is_manipulated(self) -> bool:
        """True if the manipulation probability exceeds 0.5 (or authenticity < 0.5)."""
        return self.visual_authenticity_score < 0.5

    @property
    def predicted_label(self) -> str:
        """Human-readable prediction label ('REAL', 'FAKE', or 'NO_FACE_DETECTED')."""
        if not self.face_detected:
            return "NO_FACE_DETECTED"
        return "FAKE" if self.is_manipulated else "REAL"

    def summary(self) -> str:
        """Formatted human-readable summary string."""
        return (
            f"Prediction: {self.predicted_label}\n"
            f"Visual Authenticity Score: {self.visual_authenticity_score:.4f} (1.0 = Real, 0.0 = Fake)\n"
            f"Manipulation Probability: {self.manipulation_probability:.4f}\n"
            f"Detected Faces: {len(self.bounding_boxes)} face(s) {self.bounding_boxes}\n"
            f"Grad-CAM Heatmap Shape: {self.gradcam_heatmap_array.shape}"
        )


# Type aliases for flexible inputs
ImageInput = Union[str, bytes, np.ndarray]
VideoInput = List[np.ndarray]
