"""Timeline and drift analysis exports."""
from core.timeline.drift import (
    compute_cross_correlation_offset,
    detect_suspicious_segments,
    calculate_lip_sync_score,
)

__all__ = [
    "compute_cross_correlation_offset",
    "detect_suspicious_segments",
    "calculate_lip_sync_score",
]
