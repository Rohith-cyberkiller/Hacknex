"""PyTorch SyncNet architecture modeled after joonson/syncnet_python and Rudrabha/Wav2Lip.

References:
- Chung & Zisserman, "Out of time: automated lip sync in the wild", ACCV 2016.
  (https://github.com/joonson/syncnet_python)
- Prajwal et al., "A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild", ACM MM 2020.
  (https://github.com/Rudrabha/Wav2Lip)
"""
import logging
from typing import Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class Conv2dBlock(nn.Module):
        """Standard 2D Convolution with BatchNorm and ReLU."""
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class SyncNetAudioEncoder(nn.Module):
        """Acoustic encoder projecting mel-spectrogram slices into 512-dim embedding."""
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                Conv2dBlock(1, 64, kernel_size=3, stride=1, padding=1),
                Conv2dBlock(64, 64, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1)),

                Conv2dBlock(64, 128, kernel_size=3, stride=1, padding=1),
                Conv2dBlock(128, 128, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                Conv2dBlock(128, 256, kernel_size=3, stride=1, padding=1),
                Conv2dBlock(256, 256, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                Conv2dBlock(256, 512, kernel_size=3, stride=1, padding=1),
                Conv2dBlock(512, 512, kernel_size=3, stride=1, padding=1),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.fc = nn.Linear(512, 512)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (B, 1, Mel_bins, T_audio_frames)
            out = self.conv(x)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return F.normalize(out, p=2, dim=1)

    class SyncNetVisualEncoder(nn.Module):
        """Visual encoder projecting 5-frame cropped mouth window into 512-dim embedding."""
        def __init__(self):
            super().__init__()
            # 5 frames stacked across channel dimension (5 * 3 = 15 channels or 5 grayscale)
            self.conv = nn.Sequential(
                Conv2dBlock(15, 64, kernel_size=7, stride=2, padding=3),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                Conv2dBlock(64, 128, kernel_size=5, stride=1, padding=2),
                Conv2dBlock(128, 128, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                Conv2dBlock(128, 256, kernel_size=3, stride=1, padding=1),
                Conv2dBlock(256, 256, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                Conv2dBlock(256, 512, kernel_size=3, stride=1, padding=1),
                Conv2dBlock(512, 512, kernel_size=3, stride=1, padding=1),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.fc = nn.Linear(512, 512)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (B, 15, H, W) where H=W=96
            out = self.conv(x)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return F.normalize(out, p=2, dim=1)

    class SyncNet(nn.Module):
        """Full two-stream dual-encoder SyncNet architecture."""
        def __init__(self):
            super().__init__()
            self.audio_encoder = SyncNetAudioEncoder()
            self.visual_encoder = SyncNetVisualEncoder()

        def forward(self, audio_feat: torch.Tensor, visual_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Output normalized L2 embeddings for both audio and visual inputs."""
            a_emb = self.audio_encoder(audio_feat)
            v_emb = self.visual_encoder(visual_feat)
            return a_emb, v_emb

        def compute_distance(self, a_emb: torch.Tensor, v_emb: torch.Tensor) -> torch.Tensor:
            """Compute Euclidean distance between normalized embeddings (range 0.0 to 2.0)."""
            return F.pairwise_distance(v_emb, a_emb)

        def compute_cosine_similarity(self, a_emb: torch.Tensor, v_emb: torch.Tensor) -> torch.Tensor:
            """Cosine similarity in range [-1.0, 1.0]."""
            return F.cosine_similarity(v_emb, a_emb, dim=1)

else:
    # Safe stubs when PyTorch is absent
    SyncNet = None
    SyncNetAudioEncoder = None
    SyncNetVisualEncoder = None
