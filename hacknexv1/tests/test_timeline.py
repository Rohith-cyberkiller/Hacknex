"""Unit tests for synthetic audiovisual signals, timing offset, and drift timeline detection."""
import pytest
import numpy as np

from core.timeline.drift import (
    compute_cross_correlation_offset,
    detect_suspicious_segments,
    calculate_lip_sync_score,
)


def test_synthetic_perfect_synchronization():
    """Identical sinusoidal envelopes should yield ~0 ms offset and high score."""
    fps = 20.0
    duration_sec = 6.0
    n = int(fps * duration_sec)
    t = np.linspace(0, duration_sec, n)

    # 1.5 Hz speech articulation rhythm
    carrier = 0.5 + 0.4 * np.sin(2 * np.pi * 1.5 * t)
    mouth = carrier.copy()
    speech = carrier.copy()

    offset_ms, peak_corr = compute_cross_correlation_offset(mouth, speech, fps=fps)
    assert abs(offset_ms) <= 50
    assert peak_corr > 0.90

    segments, points = detect_suspicious_segments(mouth, speech, t, fps=fps)
    assert len(segments) == 0
    assert len(points) == n

    score = calculate_lip_sync_score(peak_corr, offset_ms, segments, duration_sec)
    assert score > 80.0


def test_synthetic_lag_detection():
    """Known 200 ms artificial lag (4 frames at 20 fps) must be recovered."""
    fps = 20.0
    duration_sec = 8.0
    n = int(fps * duration_sec)
    t = np.linspace(0, duration_sec, n)

    base = 0.5 + 0.45 * np.sin(2 * np.pi * 1.0 * t)
    shift_frames = 4  # 4 / 20 = 0.20s = 200ms
    speech = base
    # Mouth leads speech by 4 frames
    mouth = np.roll(base, shift_frames)

    offset_ms, peak_corr = compute_cross_correlation_offset(mouth, speech, fps=fps)
    # The recovered offset should be around 200 ms (tolerance 50 ms)
    assert abs(offset_ms - 200) <= 50


def test_synthetic_speech_with_still_mouth_anomaly():
    """Mouth stationary while speech is active should trigger suspicious segments."""
    fps = 20.0
    duration_sec = 5.0
    n = int(fps * duration_sec)
    t = np.linspace(0, duration_sec, n)

    speech = np.full(n, 0.65)  # Constant loud voice
    mouth = np.full(n, 0.02)   # Stationary lips

    segments, points = detect_suspicious_segments(
        mouth_energy=mouth,
        speech_energy=speech,
        timestamps=t,
        fps=fps,
        speech_thresh=0.20,
        mouth_thresh=0.15,
    )

    assert len(segments) >= 1
    assert any("does not align with speech energy" in s.reason for s in segments)
    assert segments[0].severity in ["medium", "high"]


def test_synthetic_mouth_babble_during_silence():
    """Mouth moving during silence should trigger silent speech anomaly."""
    fps = 20.0
    duration_sec = 5.0
    n = int(fps * duration_sec)
    t = np.linspace(0, duration_sec, n)

    mouth = np.full(n, 0.70)   # Moving lips
    speech = np.full(n, 0.01)  # Complete silence

    segments, points = detect_suspicious_segments(
        mouth_energy=mouth,
        speech_energy=speech,
        timestamps=t,
        fps=fps,
        speech_thresh=0.20,
        mouth_thresh=0.15,
    )

    assert len(segments) >= 1
    assert any("silent audio segment" in s.reason for s in segments)
