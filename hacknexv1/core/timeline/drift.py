"""Temporal drift analysis and audiovisual cross-correlation tools."""
import logging
from typing import List, Tuple, Dict, Any
import numpy as np

from core.models.schemas import SuspiciousSegment, TimelinePoint

logger = logging.getLogger(__name__)


def compute_cross_correlation_offset(
    mouth_energy: np.ndarray,
    speech_energy: np.ndarray,
    fps: float,
    max_lag_sec: float = 0.5,
) -> Tuple[int, float]:
    """Estimate the temporal offset in ms and maximum cross-correlation.
    
    A positive offset (ms) means video/mouth leads audio; negative means audio leads.
    
    Args:
        mouth_energy: 1D normalized mouth motion array.
        speech_energy: 1D normalized speech energy array.
        fps: Frames per second.
        max_lag_sec: Maximum temporal search window in seconds (default 500 ms).
        
    Returns:
        (offset_ms, peak_correlation):
            offset_ms: Estimated offset in milliseconds.
            peak_correlation: Pearson correlation coefficient at the optimal lag.
    """
    n = len(mouth_energy)
    if n < 4 or len(speech_energy) < 4:
        return 0, 0.0

    # Ensure zero-mean, unit-variance signals for standard normalized cross-correlation
    m_std = np.std(mouth_energy)
    s_std = np.std(speech_energy)
    
    if m_std < 1e-6 or s_std < 1e-6:
        # One or both signals have negligible variation (e.g. silent or still)
        return 0, 0.0

    m_norm = (mouth_energy - np.mean(mouth_energy)) / m_std
    s_norm = (speech_energy - np.mean(speech_energy)) / s_std

    max_lag_frames = int(round(max_lag_sec * fps))
    if max_lag_frames < 1:
        max_lag_frames = 1
    
    # Restrict lag search so we have enough overlapping points
    max_lag_frames = min(max_lag_frames, n // 2)

    lags = np.arange(-max_lag_frames, max_lag_frames + 1)
    correlations = []

    for lag in lags:
        if lag < 0:
            # mouth shifted earlier
            m_slice = m_norm[:lag]
            s_slice = s_norm[-lag:]
        elif lag > 0:
            m_slice = m_norm[lag:]
            s_slice = s_norm[:-lag]
        else:
            m_slice = m_norm
            s_slice = s_norm

        if len(m_slice) > 0:
            corr = float(np.mean(m_slice * s_slice))
        else:
            corr = 0.0
        correlations.append(corr)

    correlations = np.array(correlations)
    best_idx = int(np.argmax(correlations))
    best_lag = lags[best_idx]
    peak_corr = float(np.clip(correlations[best_idx], -1.0, 1.0))
    
    # Lag in frames to ms
    offset_sec = best_lag / max(fps, 1.0)
    offset_ms = int(round(offset_sec * 1000.0))

    return offset_ms, peak_corr


def detect_suspicious_segments(
    mouth_energy: np.ndarray,
    speech_energy: np.ndarray,
    timestamps: np.ndarray,
    fps: float,
    min_segment_sec: float = 0.4,
    speech_thresh: float = 0.20,
    mouth_thresh: float = 0.15,
) -> Tuple[List[SuspiciousSegment], List[TimelinePoint]]:
    """Identify periods of audiovisual desynchronization.
    
    Detects two primary anomalies:
    1. Speech is active, but lips are stationary (voiceover / puppetry mismatch).
    2. Lips are actively moving, but audio is completely silent.
    3. Severe local anti-correlation during dialogue.
    
    Args:
        mouth_energy: 1D normalized mouth motion array.
        speech_energy: 1D normalized speech energy array.
        timestamps: 1D timestamp array in seconds.
        fps: Frames per second.
        min_segment_sec: Minimum segment duration in seconds to filter micro-pauses.
        speech_thresh: Energy threshold for active voice.
        mouth_thresh: Energy threshold for active lip motion.
        
    Returns:
        (suspicious_segments, timeline_points)
    """
    n = len(timestamps)
    if n == 0:
        return [], []

    # Compute rolling window correlation (window size ~ 1.5 seconds)
    win_size = max(5, int(round(1.5 * fps)))
    is_anomaly = np.zeros(n, dtype=bool)
    reasons = [""] * n
    severities = ["low"] * n

    for i in range(n):
        s_val = speech_energy[i]
        m_val = mouth_energy[i]

        # Case 1: Speech present, mouth still
        if s_val > speech_thresh and m_val < (mouth_thresh * 0.5):
            is_anomaly[i] = True
            severities[i] = "high" if s_val > 0.45 else "medium"
            reasons[i] = "Mouth motion does not align with speech energy"

        # Case 2: Mouth moving heavily, audio silent
        elif m_val > (mouth_thresh * 1.8) and s_val < (speech_thresh * 0.4):
            is_anomaly[i] = True
            severities[i] = "medium" if m_val > 0.4 else "low"
            reasons[i] = "Mouth motion detected during silent audio segment"

        # Case 3: Local inverse correlation when both have nominal amplitude
        else:
            w_start = max(0, i - win_size // 2)
            w_end = min(n, i + win_size // 2 + 1)
            w_m = mouth_energy[w_start:w_end]
            w_s = speech_energy[w_start:w_end]
            if len(w_m) >= 5 and np.std(w_m) > 0.05 and np.std(w_s) > 0.05:
                loc_corr = np.corrcoef(w_m, w_s)[0, 1]
                if loc_corr < -0.35 and (np.mean(w_s) > speech_thresh):
                    is_anomaly[i] = True
                    severities[i] = "high" if loc_corr < -0.5 else "medium"
                    reasons[i] = "Negative local correlation between articulation and acoustic signal"

    # Group contiguous anomalies into segments
    segments: List[SuspiciousSegment] = []
    in_seg = False
    seg_start = 0

    for i in range(n):
        if is_anomaly[i] and not in_seg:
            in_seg = True
            seg_start = i
        elif not is_anomaly[i] and in_seg:
            in_seg = False
            seg_end = i - 1
            duration = timestamps[seg_end] - timestamps[seg_start]
            if duration >= min_segment_sec:
                # Aggregate reason and severity
                seg_reasons = [reasons[k] for k in range(seg_start, seg_end + 1) if reasons[k]]
                seg_sevs = [severities[k] for k in range(seg_start, seg_end + 1)]
                dominant_reason = max(set(seg_reasons), key=seg_reasons.count) if seg_reasons else "Audio-video sync mismatch"
                highest_sev = "high" if "high" in seg_sevs else ("medium" if "medium" in seg_sevs else "low")
                
                segments.append(
                    SuspiciousSegment(
                        start_sec=round(float(timestamps[seg_start]), 2),
                        end_sec=round(float(timestamps[seg_end]), 2),
                        severity=highest_sev,
                        reason=dominant_reason,
                    )
                )

    # Check trailing segment
    if in_seg:
        seg_end = n - 1
        duration = timestamps[seg_end] - timestamps[seg_start]
        if duration >= min_segment_sec:
            seg_reasons = [reasons[k] for k in range(seg_start, seg_end + 1) if reasons[k]]
            seg_sevs = [severities[k] for k in range(seg_start, seg_end + 1)]
            dominant_reason = max(set(seg_reasons), key=seg_reasons.count) if seg_reasons else "Audio-video sync mismatch"
            highest_sev = "high" if "high" in seg_sevs else ("medium" if "medium" in seg_sevs else "low")
            segments.append(
                SuspiciousSegment(
                    start_sec=round(float(timestamps[seg_start]), 2),
                    end_sec=round(float(timestamps[seg_end]), 2),
                    severity=highest_sev,
                    reason=dominant_reason,
                )
            )

    # Construct sampled timeline points
    timeline_points = [
        TimelinePoint(
            timestamp_sec=round(float(timestamps[i]), 3),
            mouth_energy=round(float(mouth_energy[i]), 4),
            speech_energy=round(float(speech_energy[i]), 4),
            is_suspicious=bool(is_anomaly[i]),
        )
        for i in range(n)
    ]

    return segments, timeline_points


def calculate_lip_sync_score(
    peak_correlation: float,
    offset_ms: int,
    suspicious_segments: List[SuspiciousSegment],
    total_duration_sec: float,
) -> float:
    """Calculate overall lip-sync consistency score (0 to 100, higher = more authentic).
    
    Natural talking head video has:
    - High positive correlation (> 0.40)
    - Small offset (|offset| < 80ms)
    - Few or no suspicious drift periods
    """
    # 1. Base score from peak correlation: map [-0.2, 0.8] to [15, 95]
    corr_clamped = np.clip(peak_correlation, -0.3, 0.85)
    base = 15.0 + ((corr_clamped + 0.3) / 1.15) * 80.0

    # 2. Offset penalty: offsets within [-60, +60] ms are normal human perceptual tolerance (ITU-R BT.1359)
    abs_offset = abs(offset_ms)
    if abs_offset > 80:
        offset_penalty = min(35.0, (abs_offset - 80) * 0.12)
    else:
        offset_penalty = 0.0

    # 3. Suspicious segment penalty based on ratio of anomalous duration
    if total_duration_sec > 0 and suspicious_segments:
        anomaly_duration = sum(s.end_sec - s.start_sec for s in suspicious_segments)
        anomaly_ratio = np.clip(anomaly_duration / total_duration_sec, 0.0, 1.0)
        segment_penalty = anomaly_ratio * 40.0
    else:
        segment_penalty = 0.0

    raw_score = base - offset_penalty - segment_penalty
    return float(np.clip(raw_score, 5.0, 98.0))
