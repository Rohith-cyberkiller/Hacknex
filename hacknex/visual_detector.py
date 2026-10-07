"""Main public module for Visual Forensics Deepfake Detection.

This module provides the primary public API for deepfake and image manipulation detection.
It accepts image file paths, numpy arrays (BGR/RGB), or lists of video frames,
and returns a structured, typed `DetectionResult`.

Phase 2 Scope:
  - Real face detection, landmark extraction, and alignment via MTCNN.
  - Generates standardized aligned face crops (256x256) and ImageNet-normalized PyTorch tensors.
  - Multi-face and no-face handling without crashing.
  - Real detected bounding boxes returned in original image coordinates.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Union
import numpy as np

from src.pipeline import ForensicsPipeline
from src.types import AlignedFace, DetectionResult, ImageInput, VideoInput
from src.utils import load_image, validate_image_array

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("VisualDetector")


class VisualDetector:
    """Public Visual Forensics Detector class.

    Example:
        >>> detector = VisualDetector()
        >>> result = detector.detect("samples/portrait_face.jpg")
        >>> print("Authenticity:", result.visual_authenticity_score)
        >>> print("Real Face Bounding Boxes:", result.bounding_boxes)
        >>> print("Extracted Aligned Faces:", len(result.faces))
    """

    def __init__(
        self,
        device: str = "cpu",
        confidence_threshold: float = 0.70,
        target_size: int = 256,
        max_faces: int = 5,
        verbose: bool = True,
    ) -> None:
        """Initialize the visual detector with face extraction configuration.

        Args:
            device: Computing device ('cpu' or 'cuda').
            confidence_threshold: Minimum face detection confidence (0.0 to 1.0).
            target_size: Square resolution for cropped/aligned faces (e.g. 256 for 256x256).
            max_faces: Maximum number of faces to process per image.
            verbose: If True, logs step-by-step progress for each pipeline stage.
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.target_size = target_size
        self.max_faces = max_faces
        self.verbose = verbose

        self.pipeline = ForensicsPipeline(
            device=device,
            confidence_threshold=confidence_threshold,
            target_size=target_size,
            max_faces=max_faces,
            verbose=verbose,
        )

    def detect(
        self,
        input_data: Union[str, Path, np.ndarray, Sequence[np.ndarray]],
        color_space: str = "BGR",
    ) -> DetectionResult:
        """Detect visual authenticity on an image path, numpy frame, or video sequence.

        Args:
            input_data: One of:
                - File path string or Path object to an image file.
                - 2D/3D numpy array representing an image frame.
                - Sequence/List of numpy arrays representing video frames.
            color_space: 'BGR' (OpenCV default) or 'RGB' for input numpy arrays.

        Returns:
            DetectionResult: Dataclass containing:
                - visual_authenticity_score (float in [0.0, 1.0], 1.0 = Real)
                - bounding_boxes (list of real detected face (x1, y1, x2, y2) tuples)
                - gradcam_heatmap_array (2D float numpy array [0.0, 1.0])
                - faces (list of AlignedFace objects with 256x256 crops and tensors)
                - stage_logs and execution metadata

        Raises:
            FileNotFoundError: If input is a filepath that does not exist.
            ValueError: If input image dimensions are invalid or empty.
            TypeError: If input_data type is unsupported.
        """
        if isinstance(input_data, (str, Path)):
            frame = load_image(input_data)
            return self._detect_single_frame(frame)

        if isinstance(input_data, np.ndarray):
            frame = load_image(input_data)
            if color_space.upper() == "RGB" and frame.ndim == 3 and frame.shape[2] == 3:
                import cv2
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return self._detect_single_frame(frame)

        if isinstance(input_data, (list, tuple)):
            if len(input_data) == 0:
                raise ValueError("Input video frames list is empty.")
            return self.detect_video_frames(input_data, color_space=color_space)

        raise TypeError(
            f"Unsupported input type '{type(input_data)}'. "
            "Expected filepath (str/Path), numpy frame (np.ndarray), or list of frames."
        )

    def _detect_single_frame(self, frame: np.ndarray) -> DetectionResult:
        """Run the forensics pipeline on a validated single BGR frame."""
        return self.pipeline.run(frame)

    def detect_video_frames(
        self,
        frames: Sequence[np.ndarray],
        color_space: str = "BGR",
    ) -> DetectionResult:
        """Analyze a sequence of video frames and aggregate authenticity scores.

        Args:
            frames: Sequence of numpy image frames.
            color_space: 'BGR' or 'RGB'.

        Returns:
            DetectionResult: Aggregated detection result across all frames.
        """
        if not frames:
            raise ValueError("Frames list cannot be empty.")

        frame_results: List[DetectionResult] = []
        for i, frame in enumerate(frames):
            if self.verbose:
                logger.info("Processing frame %d/%d...", i + 1, len(frames))
            res = self.detect(frame, color_space=color_space)
            frame_results.append(res)

        # Aggregate metrics across frames
        scores = [r.visual_authenticity_score for r in frame_results]
        min_score = min(scores)
        mean_score = float(np.mean(scores))

        # Use the most suspicious frame's heatmap and face detections
        most_suspicious_idx = int(np.argmin(scores))
        lead_result = frame_results[most_suspicious_idx]

        return DetectionResult(
            visual_authenticity_score=min_score,
            bounding_boxes=lead_result.bounding_boxes,
            gradcam_heatmap_array=lead_result.gradcam_heatmap_array,
            faces=lead_result.faces,
            stage_logs=lead_result.stage_logs,
            metadata={
                "frame_count": len(frames),
                "mean_authenticity_score": mean_score,
                "min_authenticity_score": min_score,
                "most_suspicious_frame_idx": most_suspicious_idx,
                "pipeline_version": "0.5.0-phase5",
            },
        )

    def evaluate_robustness(
        self,
        input_data: Union[str, Path, np.ndarray],
        profiles: Optional[Sequence[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate detection stability under simulated social-media degradation.

        Args:
            input_data: Image path or numpy frame.
            profiles: Optional subset of profile names to test. Defaults to standard suite.

        Returns:
            Dict mapping profile_name -> dict with psnr, score, score_delta, label, faces_detected.
        """
        from src.degradation import apply_degradation_profile, DEGRADATION_PROFILES

        frame = load_image(input_data)
        if profiles is None:
            profiles = ["clean", "jpeg_q70", "jpeg_q50", "blur_k3", "downscale_0.5", "social_media_heavy"]

        # Run clean baseline
        clean_result = self._detect_single_frame(frame)
        clean_score = clean_result.visual_authenticity_score

        robustness_report: Dict[str, Dict[str, Any]] = {}
        for p_name in profiles:
            deg_res = apply_degradation_profile(frame, p_name)
            deg_det = self._detect_single_frame(deg_res.degraded_image)
            score_delta = deg_det.visual_authenticity_score - clean_score

            robustness_report[p_name] = {
                "psnr": deg_res.psnr,
                "score": deg_det.visual_authenticity_score,
                "score_delta": round(score_delta, 4),
                "label": deg_det.predicted_label,
                "faces_detected": len(deg_det.faces),
            }

        return robustness_report


# Top-level functional API
_default_detector: Optional[VisualDetector] = None


def detect_visual_authenticity(
    input_data: Union[str, Path, np.ndarray, Sequence[np.ndarray]],
    verbose: bool = True,
    device: str = "cpu",
    confidence_threshold: float = 0.70,
    target_size: int = 256,
    max_faces: int = 5,
    color_space: str = "BGR",
) -> DetectionResult:
    """Convenience top-level function for detecting visual authenticity with face alignment.

    Args:
        input_data: Image path, numpy frame, or list of frames.
        verbose: Whether to log pipeline stages.
        device: 'cpu' or 'cuda'.
        confidence_threshold: Minimum face detection confidence.
        target_size: Square resolution for cropped/aligned faces.
        max_faces: Max faces to detect.
        color_space: Color format of input array ('BGR' or 'RGB').

    Returns:
        DetectionResult with score, real detected bounding boxes, heatmap, and aligned faces.

    Example:
        >>> from visual_detector import detect_visual_authenticity
        >>> result = detect_visual_authenticity("samples/portrait_face.jpg")
        >>> print(f"Detected {len(result.faces)} face(s)")
    """
    global _default_detector
    if (
        _default_detector is None
        or _default_detector.verbose != verbose
        or _default_detector.device != device
        or _default_detector.target_size != target_size
    ):
        _default_detector = VisualDetector(
            device=device,
            confidence_threshold=confidence_threshold,
            target_size=target_size,
            max_faces=max_faces,
            verbose=verbose,
        )
    return _default_detector.detect(input_data, color_space=color_space)


def evaluate_robustness(
    input_data: Union[str, Path, np.ndarray],
    profiles: Optional[Sequence[str]] = None,
    device: str = "cpu",
) -> Dict[str, Dict[str, Any]]:
    """Evaluate detection robustness under simulated social-media degradations."""
    detector = VisualDetector(device=device, verbose=False)
    return detector.evaluate_robustness(input_data, profiles=profiles)


__all__ = [
    "VisualDetector",
    "detect_visual_authenticity",
    "evaluate_robustness",
    "DetectionResult",
    "AlignedFace",
]

