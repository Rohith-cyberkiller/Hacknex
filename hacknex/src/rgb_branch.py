"""RGB spatial branch for deepfake detection (Phase 3).

Uses EfficientNet-B0 as the spatial backbone for analyzing RGB pixel-level
manipulation artifacts in aligned face crops.

Backbone Choice: EfficientNet-B0
  - Compact (5.3M params) and CPU-friendly inference (~15ms per face).
  - Pretrained ImageNet features capture fine texture irregularities (skin pores,
    specular reflections, compression blocky artifacts) that GANs struggle to reproduce.
  - Widely adopted in FaceForensics++ benchmarks with strong baselines.
  - Available via torchvision with zero external weight files.

Architecture:
  Input: (B, 3, 256, 256) ImageNet-normalized RGB tensor
  Backbone: EfficientNet-B0 feature extractor (frozen or fine-tunable)
  Head: AdaptiveAvgPool → Dropout(0.3) → Linear(1280, 256) → ReLU → Linear(256, 2)
  Output: softmax probability [p_real, p_fake]

Without trained deepfake weights, the backbone uses ImageNet features with a
randomly-initialized classification head.  The output is a real feature-based
signal (not a hardcoded stub), but accuracy is limited until fine-tuned on
FaceForensics++ or equivalent deepfake datasets.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("RGBBranch")


class RGBClassifier(nn.Module):
    """EfficientNet-B0 based binary classifier for real vs fake face detection.

    The backbone extracts rich spatial texture features; the classification head
    maps them to a two-class softmax distribution.
    """

    def __init__(self, pretrained_backbone: bool = True, dropout: float = 0.3) -> None:
        super().__init__()
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

        if pretrained_backbone:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            backbone = efficientnet_b0(weights=weights)
        else:
            backbone = efficientnet_b0(weights=None)

        # Extract feature layers (everything except the final classifier)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        feature_dim = 1280  # EfficientNet-B0 final channel count

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(256, 2),
        )

        # Freeze backbone by default (transfer learning mode)
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, 3, H, W), ImageNet-normalized.

        Returns:
            torch.Tensor: Logits (B, 2) for [real, fake].
        """
        features = self.features(x)
        pooled = self.avgpool(features).flatten(1)
        logits = self.classifier(pooled)
        return logits

    def get_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the final convolutional feature map (for Grad-CAM in Phase 4).

        Returns:
            torch.Tensor: Feature map (B, 1280, H', W').
        """
        return self.features(x)


class RGBBranch:
    """Inference wrapper for the RGB spatial branch.

    Manages model initialization, device placement, and batched scoring.
    """

    def __init__(self, device: str = "cpu", weights_path: Optional[str] = None) -> None:
        """Initialize the RGB branch.

        Args:
            device: 'cpu' or 'cuda'.
            weights_path: Optional path to trained deepfake classifier weights.
                          If None, uses ImageNet backbone + random head (placeholder).
        """
        self.device = torch.device(device)
        self.model = RGBClassifier(pretrained_backbone=True)
        self._weights_loaded = False

        if weights_path is not None:
            try:
                state = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state)
                self._weights_loaded = True
                logger.info("Loaded trained RGB branch weights from: %s", weights_path)
            except Exception as err:
                logger.warning("Could not load RGB weights (%s): %s. Using ImageNet backbone.", weights_path, err)

        self.model.to(self.device)
        self.model.eval()

        if not self._weights_loaded:
            logger.info(
                "RGB Branch initialized with ImageNet-pretrained EfficientNet-B0 backbone. "
                "Classification head is untrained (placeholder). Fine-tune on deepfake dataset for production accuracy."
            )

    @torch.no_grad()
    def score_faces(self, face_tensors: List[torch.Tensor]) -> List[float]:
        """Compute real-probability scores for a list of aligned face tensors.

        Args:
            face_tensors: List of (3, H, W) ImageNet-normalized tensors.

        Returns:
            List[float]: Per-face authenticity scores in [0.0, 1.0] (1.0 = real).
        """
        if not face_tensors:
            return []

        batch = torch.stack(face_tensors).to(self.device)
        logits = self.model(batch)
        probs = F.softmax(logits, dim=1)
        # Column 0 = P(real), Column 1 = P(fake)
        real_probs = probs[:, 0].cpu().tolist()
        return real_probs

    @torch.no_grad()
    def get_feature_maps(self, face_tensors: List[torch.Tensor]) -> torch.Tensor:
        """Extract final conv feature maps for Grad-CAM (Phase 4).

        Args:
            face_tensors: List of (3, H, W) ImageNet-normalized tensors.

        Returns:
            torch.Tensor: Feature maps (B, 1280, H', W').
        """
        if not face_tensors:
            return torch.empty(0)
        batch = torch.stack(face_tensors).to(self.device)
        return self.model.get_feature_map(batch)
