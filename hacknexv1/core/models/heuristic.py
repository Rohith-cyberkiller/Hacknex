"""Heuristic lip-sync correlation model using optical motion and acoustic envelopes."""
import logging
from typing import Optional, Dict, Any
import numpy as np

from core.models.base import LipSyncModel
from core.models.schemas import LipSyncResult
from core.timeline.drift import (
    compute_cross_correlation_offset,
    detect_suspicious_segments,
    calculate_lip_sync_score,
)

logger = logging.getLogger(__name__)


class HeuristicLipSyncModel(LipSyncModel):
    """Production-grade heuristic model evaluating audiovisual synchronization.
    
    Extracts normalized optical mouth-motion dynamics and correlates them with
    acoustic speech envelopes. Detects timing lag and anomalies in speech-movement alignment.
    """

    @property
    def name(self) -> str:
        return "Heuristic Audiovisual Correlation Model"

    @property
    def analysis_mode(self) -> str:
        return "heuristic"

    def is_available(self) -> bool:
        return True

    def analyze(
        self,
        mouth_energy: np.ndarray,
        speech_energy: np.ndarray,
        timestamps: np.ndarray,
        fps: float,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> LipSyncResult:
        """Execute heuristic cross-modal synchronization analysis."""
        if len(timestamps) == 0:
            return LipSyncResult(
                score=50.0,
                estimated_offset_ms=0,
                correlation=0.0,
                analysis_mode=self.analysis_mode,
                suspicious_segments=[],
                timeline_series=[],
            )

        # 1. Compute cross-correlation peak and timing offset
        offset_ms, peak_corr = compute_cross_correlation_offset(
            mouth_energy=mouth_energy,
            speech_energy=speech_energy,
            fps=fps,
            max_lag_sec=0.5,
        )

        # 2. Detect localized temporal drift segments
        suspicious_segments, timeline_points = detect_suspicious_segments(
            mouth_energy=mouth_energy,
            speech_energy=speech_energy,
            timestamps=timestamps,
            fps=fps,
        )

        # 3. Derive aggregate lip-sync consistency score (0 to 100)
        total_duration = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 1.0
        score = calculate_lip_sync_score(
            peak_correlation=peak_corr,
            offset_ms=offset_ms,
            suspicious_segments=suspicious_segments,
            total_duration_sec=total_duration,
        )

        logger.info(
            f"HeuristicLipSync: offset={offset_ms}ms, corr={peak_corr:.3f}, "
            f"score={score:.1f}, suspicious_segs={len(suspicious_segments)}"
        )

        return LipSyncResult(
            score=round(float(score), 1),
            estimated_offset_ms=int(offset_ms),
            correlation=round(float(peak_corr), 2),
            analysis_mode=self.analysis_mode,
            suspicious_segments=suspicious_segments,
            timeline_series=timeline_points,
        )
