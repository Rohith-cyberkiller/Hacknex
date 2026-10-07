"""Utility functions for image I/O, validation, visualization, and mock heatmaps.

Provides robust image loading, validation, alpha-blended Grad-CAM overlays,
facial landmark rendering, and face crop saving utilities for Phase 2.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

from src.types import AlignedFace


def load_image(input_source: Union[str, Path, np.ndarray]) -> np.ndarray:
    """Load and validate an image from a filepath or an existing numpy array.

    Args:
        input_source: File path (str or Path) or numpy image array (BGR or RGB).

    Returns:
        np.ndarray: BGR image array with shape (H, W, 3) and dtype uint8.

    Raises:
        FileNotFoundError: If the given file path does not exist.
        ValueError: If the input cannot be decoded or is not a valid image.
    """
    if isinstance(input_source, (str, Path)):
        file_path = Path(input_source)
        if not file_path.is_file():
            raise FileNotFoundError(f"Image file not found: {file_path.resolve()}")

        # Windows-safe unicode reading using numpy and cv2.imdecode
        try:
            image_data = np.fromfile(str(file_path), dtype=np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        except Exception as err:
            raise ValueError(f"Failed to read file '{file_path}': {err}") from err

        if image is None:
            raise ValueError(f"Could not decode image from file: {file_path}")

        return image

    if isinstance(input_source, np.ndarray):
        validate_image_array(input_source)
        # Normalize float images to uint8 [0, 255] if necessary
        if input_source.dtype in (np.float32, np.float64):
            if input_source.max() <= 1.0:
                return (np.clip(input_source, 0.0, 1.0) * 255.0).astype(np.uint8)
            return np.clip(input_source, 0.0, 255.0).astype(np.uint8)
        return input_source.copy()

    raise TypeError(
        f"Unsupported input type: {type(input_source)}. Expected file path or np.ndarray."
    )


def validate_image_array(frame: np.ndarray) -> None:
    """Validate that the given array is a valid 2D or 3D image.

    Args:
        frame: Candidate image array.

    Raises:
        ValueError: If image dimensions or properties are invalid.
    """
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(frame)}")

    if frame.size == 0:
        raise ValueError("Image array is empty (size 0).")

    if frame.ndim == 2:
        return

    if frame.ndim == 3:
        channels = frame.shape[2]
        if channels not in (1, 3, 4):
            raise ValueError(f"Expected 1, 3, or 4 color channels, got {channels}.")
        return

    raise ValueError(f"Image array must have 2 or 3 dimensions, got ndim={frame.ndim}.")


def generate_mock_heatmap(
    height: int,
    width: int,
    bounding_box: Optional[Tuple[int, int, int, int]] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a realistic mock 2D Grad-CAM heatmap array centered on face region."""
    if seed is not None:
        np.random.seed(seed)

    y_coords, x_coords = np.ogrid[:height, :width]

    if bounding_box is not None:
        x1, y1, x2, y2 = bounding_box
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        sigma_x = max(10.0, (x2 - x1) * 0.35)
        sigma_y = max(10.0, (y2 - y1) * 0.35)
    else:
        center_x = width / 2.0
        center_y = height / 2.0
        sigma_x = width * 0.25
        sigma_y = height * 0.25

    gaussian_main = np.exp(
        -(
            ((x_coords - center_x) ** 2) / (2.0 * sigma_x**2)
            + ((y_coords - center_y) ** 2) / (2.0 * sigma_y**2)
        )
    )

    sub_x = center_x + sigma_x * 0.3
    sub_y = center_y + sigma_y * 0.2
    gaussian_secondary = 0.35 * np.exp(
        -(
            ((x_coords - sub_x) ** 2) / (2.0 * (sigma_x * 0.5) ** 2)
            + ((y_coords - sub_y) ** 2) / (2.0 * (sigma_y * 0.5) ** 2)
        )
    )

    combined = gaussian_main + gaussian_secondary
    min_val = float(combined.min())
    max_val = float(combined.max())
    if max_val > min_val:
        heatmap = (combined - min_val) / (max_val - min_val)
    else:
        heatmap = np.zeros((height, width), dtype=np.float32)

    return heatmap.astype(np.float32)


def create_overlay_image(
    base_image: np.ndarray,
    heatmap: np.ndarray,
    bounding_boxes: List[Tuple[int, int, int, int]],
    authenticity_score: float,
    faces: Optional[List[AlignedFace]] = None,
    alpha: float = 0.40,
    colormap_type: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Create a presentation-ready visualization blending original image, heatmap, and face landmarks.

    Args:
        base_image: Original image in BGR format (H, W, 3).
        heatmap: 2D float array (H, W) in range [0.0, 1.0].
        bounding_boxes: List of (x1, y1, x2, y2) tuples.
        authenticity_score: Detection authenticity score in [0.0, 1.0].
        faces: Optional list of AlignedFace objects containing landmarks and confidences.
        alpha: Blending weight for heatmap (0.0 = original image, 1.0 = heatmap).
        colormap_type: OpenCV colormap ID.

    Returns:
        np.ndarray: BGR overlay image with annotations.
    """
    h, w = base_image.shape[:2]

    # Ensure heatmap matches image dimensions
    if heatmap.shape[:2] != (h, w):
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        heatmap_resized = heatmap

    # If no faces were detected, skip heatmap blend to keep clean view
    if len(bounding_boxes) == 0:
        overlay = base_image.copy()
    else:
        heatmap_uint8 = np.clip(heatmap_resized * 255.0, 0, 255).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap_type)
        overlay = cv2.addWeighted(colored_heatmap, alpha, base_image, 1.0 - alpha, 0)

    # Status color
    has_faces = len(bounding_boxes) > 0
    is_real = authenticity_score >= 0.5
    if not has_faces:
        box_color = (180, 180, 180)  # Neutral grey
        label_text = "NO FACE DETECTED"
    else:
        box_color = (46, 204, 113) if is_real else (52, 73, 235)  # Green vs Red
        label_text = f"{'REAL' if is_real else 'FAKE'}: {authenticity_score * 100:.1f}% Authenticity"

    # Draw face bounding boxes and landmarks
    for i, box in enumerate(bounding_boxes):
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        # Box outline
        cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, 2)

        # Confidence tag
        conf_str = ""
        if faces and i < len(faces):
            conf_str = f" {faces[i].confidence * 100:.1f}%"
        tag = f"Face #{i+1}{conf_str}"

        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tag_y1 = max(0, y1 - th - 6)
        cv2.rectangle(overlay, (x1, tag_y1), (x1 + tw + 8, y1), box_color, -1)
        cv2.putText(
            overlay,
            tag,
            (x1 + 4, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Draw 5-point landmarks if available
        if faces and i < len(faces) and faces[i].landmarks is not None:
            lms = faces[i].landmarks.as_array()
            for pt in lms:
                px, py = int(round(pt[0])), int(round(pt[1]))
                if 0 <= px < w and 0 <= py < h:
                    cv2.circle(overlay, (px, py), 3, (0, 255, 255), -1)  # Yellow dots
                    cv2.circle(overlay, (px, py), 4, (0, 0, 0), 1)

    # Top overlay banner
    banner_height = 42
    banner = np.zeros((banner_height, w, 3), dtype=np.uint8)
    banner[:] = (25, 25, 25)
    overlay_with_banner = np.vstack([banner, overlay])

    banner_subtitle = f"[Visual Forensics Phase 2] {label_text}"
    if has_faces:
        banner_subtitle += f" ({len(bounding_boxes)} face(s) aligned)"

    cv2.putText(
        overlay_with_banner,
        banner_subtitle,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        box_color,
        2,
        cv2.LINE_AA,
    )

    return overlay_with_banner


def save_aligned_face_crops(
    faces: List[AlignedFace],
    output_dir: Union[str, Path],
    prefix: str = "face",
) -> List[Path]:
    """Save cropped and aligned face images to disk for inspection.

    Args:
        faces: List of AlignedFace objects.
        output_dir: Destination directory.
        prefix: Filename prefix.

    Returns:
        List[Path]: Paths of saved crop images.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    for i, face in enumerate(faces):
        filename = f"{prefix}_{i+1}_aligned_{face.target_size}x{face.target_size}_conf{int(face.confidence*100)}.png"
        crop_path = output_dir / filename
        cv2.imwrite(str(crop_path), face.crop_bgr)
        saved_paths.append(crop_path)

    return saved_paths


def create_synthetic_sample_image(
    output_path: Union[str, Path],
    width: int = 640,
    height: int = 640,
) -> Path:
    """Create a synthetic test image with no real face (for no-face testing)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        factor = y / height
        b = int(45 + 30 * factor)
        g = int(35 + 25 * factor)
        r = int(30 + 20 * factor)
        img[y, :] = (b, g, r)

    cv2.putText(
        img,
        "No-Face Test Sample (Phase 2 Test)",
        (20, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_path), img)
    return output_path
