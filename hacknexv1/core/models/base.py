"""Abstract Base Class for Lip-Sync Models in Cross-Modal Deepfake Forensics."""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import numpy as np

from core.models.schemas import LipSyncResult


class LipSyncModel(ABC):
    """Abstract interface for audiovisual lip-sync synchronization detectors.
    
    Subclasses must implement `analyze` which correlates mouth-motion dynamics
    with speech-energy dynamics or deep multi-modal embedding projections.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the model or adapter."""
        pass

    @property
    @abstractmethod
    def analysis_mode(self) -> str:
        """Mode identifier: 'heuristic' or 'pretrained'."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Indicates if the model weights/runtime dependencies are available."""
        pass

    @abstractmethod
    def analyze(
        self,
        mouth_energy: np.ndarray,
        speech_energy: np.ndarray,
        timestamps: np.ndarray,
        fps: float,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> LipSyncResult:
        """Perform synchronization analysis.
        
        Args:
            mouth_energy: 1D array of normalized mouth motion magnitudes per frame.
            speech_energy: 1D array of normalized speech acoustic energy per frame.
            timestamps: 1D array of timestamp points (seconds) corresponding to frames.
            fps: Video sample rate in frames per second.
            extra_context: Optional metadata such as raw frames, audio waveform, or bounding boxes.
            
        Returns:
            LipSyncResult containing score (0-100), estimated offset (ms), correlation,
            suspicious segments, and analysis_mode.
        """
        pass
