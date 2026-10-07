"""Face extraction, landmark alignment, and preprocessing pipeline for Phase 2.

This module provides production-grade facial preprocessing for visual forensics:
  1. Face Detection via MTCNN (facenet-pytorch): extracts bounding boxes and confidences.
  2. Facial Landmark Localization: extracts 5 facial keypoints (eyes, nose, mouth corners).
  3. Similarity Transform Alignment: warps faces to canonical upright pose via Umeyama/affine transform.
  4. Resizing & Standardization: crops to fixed model input dimensions (default 256x256).
  5. Tensor Normalization: prepares PyTorch RGB tensors with ImageNet mean/std for Phase 3 CNNs.

Architectural Justification:
  MTCNN is selected as primary for Phase 2 because:
    - Pretrained weights are self-contained and pre-packaged with zero external download hurdles.
    - CPU-optimized inference requires no GPU dependencies.
    - Standard 5-point landmark prediction aligns directly with FaceForensics++ benchmark protocols.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
import torch

from src.types import AlignedFace, BoundingBox, FacialLandmarks

logger = logging.getLogger("FaceExtraction")

# Standard canonical 5-point facial landmark reference template (ArcFace standard for 112x112)
ARCFACE_SRC_112 = np.array(
    [
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose tip
        [41.5493, 92.3655],  # left mouth corner
        [70.7299, 92.2041],  # right mouth corner
    ],
    dtype=np.float32,
)

# Standard ImageNet normalization parameters for deep learning backbones
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_canonical_landmarks(target_size: int = 256) -> np.ndarray:
    """Scale the canonical 5-point facial reference landmarks to target crop resolution."""
    scale = target_size / 112.0
    return (ARCFACE_SRC_112 * scale).astype(np.float32)


def align_face_similarity_transform(
    image: np.ndarray,
    landmarks: np.ndarray,
    target_size: int = 256,
) -> np.ndarray:
    """Align a face using 5-point landmarks and a 2D similarity transform.

    Computes an optimal rotation, translation, and scale mapping detected landmarks
    to the canonical reference template, eliminating head tilt and perspective bias.

    Args:
        image: Source image array (H, W, 3).
        landmarks: Detected 5x2 landmarks array.
        target_size: Target square crop size in pixels (e.g. 256).

    Returns:
        np.ndarray: Aligned face crop with shape (target_size, target_size, 3).
    """
    ref_landmarks = get_canonical_landmarks(target_size)

    # Estimate partial 2D affine transform (similarity transform: scale + rotation + translation)
    # RANSAC / LMEDS robust estimation
    transform_matrix, inliers = cv2.estimateAffinePartial2D(
        landmarks.astype(np.float32),
        ref_landmarks,
        method=cv2.LMEDS,
    )

    if transform_matrix is None:
        # Fallback to standard affine if partial estimate fails
        try:
            transform_matrix = cv2.getAffineTransform(
                landmarks[:3].astype(np.float32),
                ref_landmarks[:3],
            )
        except Exception:
            transform_matrix = None

    if transform_matrix is not None:
        aligned = cv2.warpAffine(
            image,
            transform_matrix,
            (target_size, target_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        return aligned

    # Fallback to simple crop and resize if transform cannot be computed
    return resize_crop_fallback(image, landmarks, target_size)


def resize_crop_fallback(
    image: np.ndarray,
    landmarks: Optional[np.ndarray],
    target_size: int = 256,
) -> np.ndarray:
    """Fallback face crop using landmarks or bounding extents when affine transform fails."""
    h, w = image.shape[:2]
    if landmarks is not None and len(landmarks) > 0:
        min_x, min_y = np.min(landmarks, axis=0)
        max_x, max_y = np.max(landmarks, axis=0)
        margin_x = (max_x - min_x) * 0.4
        margin_y = (max_y - min_y) * 0.4
        x1 = max(0, int(min_x - margin_x))
        y1 = max(0, int(min_y - margin_y))
        x2 = min(w, int(max_x + margin_x))
        y2 = min(h, int(max_y + margin_y))
        crop = image[y1:y2, x1:x2]
    else:
        crop = image

    if crop.size == 0:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)

    return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)


def crop_face_bounding_box(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
    margin_ratio: float = 0.20,
    target_size: int = 256,
) -> np.ndarray:
    """Crop face using bounding box with contextual margin.

    Args:
        image: Source image array.
        box: (x1, y1, x2, y2) bounding box coordinates.
        margin_ratio: Margin to expand box by to capture facial perimeter/hairline.
        target_size: Target square crop size.

    Returns:
        np.ndarray: Cropped and resized face image.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1

    mx = int(box_w * margin_ratio)
    my = int(box_h * margin_ratio)

    cx1 = max(0, x1 - mx)
    cy1 = max(0, y1 - my)
    cx2 = min(w, x2 + mx)
    cy2 = min(h, y2 + my)

    crop = image[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)

    return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)


def normalize_face_tensor(rgb_crop: np.ndarray) -> torch.Tensor:
    """Convert an RGB face crop [0, 255] into an ImageNet-normalized PyTorch tensor (3, H, W).

    Args:
        rgb_crop: Aligned face crop with shape (H, W, 3) in RGB format, uint8.

    Returns:
        torch.Tensor: Normalized tensor with shape (3, H, W) and dtype float32.
    """
    # Scale to [0.0, 1.0]
    img_float = rgb_crop.astype(np.float32) / 255.0
    # Apply ImageNet normalization: (x - mean) / std
    normalized = (img_float - IMAGENET_MEAN) / IMAGENET_STD
    # Transpose (H, W, C) -> (C, H, W)
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float()
    return tensor


class FaceDetector:
    """Production-grade face detection and landmark alignment pipeline.

    Encapsulates MTCNN with multi-face support, confidence filtering,
    similarity transform alignment, and tensor normalization.
    """

    def __init__(
        self,
        device: str = "cpu",
        confidence_threshold: float = 0.70,
        target_size: int = 256,
        max_faces: int = 5,
        min_face_size: int = 20,
    ) -> None:
        """Initialize the Face Detector.

        Args:
            device: 'cpu' or 'cuda'.
            confidence_threshold: Minimum detection confidence to accept face.
            target_size: Target square crop resolution (default 256 for 256x256).
            max_faces: Maximum number of faces to process per image (sorted by confidence).
            min_face_size: Minimum face size in pixels for MTCNN detection.
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.target_size = target_size
        self.max_faces = max_faces
        self.min_face_size = min_face_size

        logger.info(
            "Initializing MTCNN Face Detector (device=%s, target_size=%dx%d, min_face_size=%d)...",
            self.device,
            self.target_size,
            self.target_size,
            self.min_face_size,
        )

        try:
            from facenet_pytorch import MTCNN

            self._mtcnn = MTCNN(
                image_size=self.target_size,
                margin=0,
                min_face_size=self.min_face_size,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=False,
                select_largest=False,
                keep_all=True,
                device=self.device,
            )
            self._initialized = True
        except Exception as err:
            logger.warning("Could not initialize MTCNN: %s. Fallback mode enabled.", err)
            self._mtcnn = None
            self._initialized = False

    def detect_and_align(self, frame_bgr: np.ndarray) -> List[AlignedFace]:
        """Detect all faces in a BGR frame, extract landmarks, align, and normalize.

        Args:
            frame_bgr: Input image in OpenCV BGR format with shape (H, W, 3).

        Returns:
            List[AlignedFace]: List of aligned face objects (empty if no faces detected).
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        h, w = frame_bgr.shape[:2]

        # Convert to RGB for MTCNN
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if not self._initialized or self._mtcnn is None:
            logger.warning("Face detector uninitialized; returning empty faces list.")
            return []

        try:
            boxes, probs, landmarks = self._mtcnn.detect(frame_rgb, landmarks=True)
        except Exception as err:
            logger.error("MTCNN detection encountered error: %s", err)
            return []

        if boxes is None or len(boxes) == 0:
            logger.info("No faces detected in input frame.")
            return []

        # Pair detections and filter by confidence threshold
        valid_detections = []
        for i, box in enumerate(boxes):
            prob = float(probs[i]) if (probs is not None and probs[i] is not None) else 0.0
            if prob >= self.confidence_threshold:
                lms = landmarks[i] if (landmarks is not None and i < len(landmarks)) else None
                valid_detections.append((box, prob, lms))

        if not valid_detections:
            logger.info(
                "Faces detected by MTCNN, but none passed confidence threshold >= %.2f.",
                self.confidence_threshold,
            )
            return []

        # Sort detections by confidence descending
        valid_detections.sort(key=lambda item: item[1], reverse=True)

        # Limit to max_faces
        if self.max_faces is not None and len(valid_detections) > self.max_faces:
            valid_detections = valid_detections[: self.max_faces]

        aligned_faces: List[AlignedFace] = []

        for box_arr, prob, lms_arr in valid_detections:
            # Clamp coordinates to frame boundary
            x1 = max(0, int(round(box_arr[0])))
            y1 = max(0, int(round(box_arr[1])))
            x2 = min(w, int(round(box_arr[2])))
            y2 = min(h, int(round(box_arr[3])))

            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label="face", confidence=prob)

            facial_landmarks: Optional[FacialLandmarks] = None
            if lms_arr is not None and len(lms_arr) == 5:
                facial_landmarks = FacialLandmarks(
                    left_eye=(float(lms_arr[0][0]), float(lms_arr[0][1])),
                    right_eye=(float(lms_arr[1][0]), float(lms_arr[1][1])),
                    nose=(float(lms_arr[2][0]), float(lms_arr[2][1])),
                    left_mouth=(float(lms_arr[3][0]), float(lms_arr[3][1])),
                    right_mouth=(float(lms_arr[4][0]), float(lms_arr[4][1])),
                )
                # Geometric alignment via similarity transform
                crop_bgr = align_face_similarity_transform(
                    frame_bgr, lms_arr, target_size=self.target_size
                )
            else:
                # Fallback to bounding box crop with margin
                crop_bgr = crop_face_bounding_box(
                    frame_bgr, (x1, y1, x2, y2), target_size=self.target_size
                )

            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            normalized_tensor = normalize_face_tensor(crop_rgb)

            aligned_faces.append(
                AlignedFace(
                    box=bbox,
                    confidence=prob,
                    crop_bgr=crop_bgr,
                    crop_rgb=crop_rgb,
                    landmarks=facial_landmarks,
                    normalized_tensor=normalized_tensor,
                    target_size=self.target_size,
                )
            )

        logger.info(
            "Successfully extracted and aligned %d face(s) (target_size=%dx%d).",
            len(aligned_faces),
            self.target_size,
            self.target_size,
        )
        return aligned_faces
