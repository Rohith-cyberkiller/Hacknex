"""Grad-CAM explainability for deepfake detection (Phase 4).

Implements Gradient-weighted Class Activation Mapping on the RGB branch
to produce pixel-level attribution heatmaps showing which facial regions
drive the real/fake classification decision.

How it works:
  1. Register forward and backward hooks on the last convolutional layer
     of EfficientNet-B0 (features[-1]).
  2. Forward-pass the face crop through the model.
  3. Backpropagate the gradient of the target class logit (fake class by
     default to highlight suspicious regions).
  4. Compute channel-wise importance weights = global average of gradients.
  5. Weighted combination of feature maps → ReLU → normalize to [0, 1].

The resulting heatmap is sized to the feature map resolution (8×8 for 256×256
input on EfficientNet-B0) and is upsampled to the face crop size and/or
mapped back to the original image coordinates.

Reference:
  Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks", ICCV 2017
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.rgb_branch import RGBClassifier

logger = logging.getLogger("GradCAM")


class GradCAM:
    """Grad-CAM heatmap generator for the RGB deepfake classifier."""

    def __init__(self, model: RGBClassifier, target_layer: Optional[str] = None) -> None:
        """Initialize Grad-CAM.

        Args:
            model: The RGBClassifier model instance.
            target_layer: Name of the target convolutional layer.
                          Defaults to the last block of EfficientNet features.
        """
        self.model = model
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None

        # Hook the last conv layer in EfficientNet features
        if target_layer is None:
            self._target_module = model.features[-1]
        else:
            self._target_module = dict(model.named_modules())[target_layer]

        self._forward_hook = self._target_module.register_forward_hook(self._save_activation)
        self._backward_hook = self._target_module.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Generate a Grad-CAM heatmap for a single face tensor.

        Args:
            input_tensor: (1, 3, H, W) or (3, H, W) ImageNet-normalized tensor.
            target_class: Class index to visualize. Default: 1 (fake class)
                          to highlight suspicious regions.
            target_size: Optional (H, W) to resize the heatmap.

        Returns:
            np.ndarray: 2D heatmap (H, W) with values in [0.0, 1.0].
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device).requires_grad_(True)

        # Forward pass
        self.model.eval()
        logits = self.model(input_tensor)

        if target_class is None:
            target_class = 1  # fake class

        # Backward pass for target class
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward(retain_graph=False)

        if self._gradients is None or self._activations is None:
            logger.warning("Grad-CAM hooks did not capture activations/gradients. Returning zero map.")
            h, w = input_tensor.shape[2], input_tensor.shape[3]
            sz = target_size if target_size else (h, w)
            return np.zeros(sz, dtype=np.float32)

        # Channel-wise importance weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)  # Only positive contributions

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Resize to target
        if target_size is not None:
            cam = cv2.resize(cam, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

        return cam.astype(np.float32)

    def cleanup(self):
        """Remove hooks."""
        self._forward_hook.remove()
        self._backward_hook.remove()


def generate_gradcam_for_faces(
    rgb_model: RGBClassifier,
    face_tensors: List[torch.Tensor],
    crop_size: int = 256,
) -> List[np.ndarray]:
    """Generate Grad-CAM heatmaps for a list of face tensors.

    Args:
        rgb_model: The RGB branch classifier.
        face_tensors: List of (3, H, W) normalized tensors.
        crop_size: Size of each face crop for heatmap resolution.

    Returns:
        List[np.ndarray]: Per-face 2D heatmaps (crop_size, crop_size) in [0, 1].
    """
    if not face_tensors:
        return []

    gradcam = GradCAM(rgb_model)
    heatmaps = []

    try:
        for tensor in face_tensors:
            hmap = gradcam.generate(tensor, target_class=1, target_size=(crop_size, crop_size))
            heatmaps.append(hmap)
    finally:
        gradcam.cleanup()

    return heatmaps


def map_heatmap_to_frame(
    heatmap: np.ndarray,
    face_box: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int],
) -> np.ndarray:
    """Map a face-crop-sized heatmap back to the full frame coordinates.

    Args:
        heatmap: 2D heatmap aligned to face crop (H_crop, W_crop).
        face_box: (x1, y1, x2, y2) of the face in the original frame.
        frame_shape: (frame_H, frame_W) of the original image.

    Returns:
        np.ndarray: Full-frame 2D heatmap (frame_H, frame_W) with values in [0, 1].
    """
    frame_h, frame_w = frame_shape
    full_heatmap = np.zeros((frame_h, frame_w), dtype=np.float32)

    x1, y1, x2, y2 = face_box
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_w, x2)
    y2 = min(frame_h, y2)

    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0 or box_h <= 0:
        return full_heatmap

    resized = cv2.resize(heatmap, (box_w, box_h), interpolation=cv2.INTER_LINEAR)
    full_heatmap[y1:y2, x1:x2] = resized

    return full_heatmap


def composite_heatmap_multi_face(
    face_heatmaps: List[np.ndarray],
    face_boxes: List[Tuple[int, int, int, int]],
    frame_shape: Tuple[int, int],
) -> np.ndarray:
    """Compose a single full-frame heatmap from multiple per-face heatmaps.

    Uses max-pooling so the most suspicious region across all faces is highlighted.

    Args:
        face_heatmaps: Per-face heatmaps (each sized to crop).
        face_boxes: Per-face (x1, y1, x2, y2) in frame coords.
        frame_shape: (H, W) of original frame.

    Returns:
        np.ndarray: Composite 2D heatmap (H, W) in [0, 1].
    """
    composite = np.zeros(frame_shape, dtype=np.float32)
    for hmap, box in zip(face_heatmaps, face_boxes):
        mapped = map_heatmap_to_frame(hmap, box, frame_shape)
        composite = np.maximum(composite, mapped)
    return composite
